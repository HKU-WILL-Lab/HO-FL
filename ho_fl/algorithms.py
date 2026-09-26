import torch
import torch.nn.functional as F

def optimizer_for(params, cfg):
    return torch.optim.AdamW(list(params.values()), lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps, weight_decay=cfg.weight_decay, amsgrad=cfg.amsgrad, foreach=cfg.foreach, fused=cfg.fused, maximize=False, capturable=False, differentiable=False)

def task_loss(wrapper, batch, tokens):
    logits = wrapper.forward(batch['input_ids'], batch['attention_mask'], token_ids=tokens)[:, -1].float()
    return F.cross_entropy(logits, batch['labels'], reduction='mean')

@torch.no_grad()
def restore(params, snapshot, debug=False):
    for name, param in params.items():
        param.copy_(snapshot[name])
    if debug:
        for name, param in params.items():
            assert torch.equal(param, snapshot[name]), name

@torch.no_grad()
def perturb(params, snapshot, directions, scale):
    for name, param in params.items():
        param.copy_(snapshot[name]).add_(directions[name], alpha=scale)

def hybrid_step(cfg, wrapper, params, prefix, batch, tokens, optimizer, directions, local_step):
    optimizer.zero_grad(set_to_none=True)
    p_count, mu = (cfg.q, cfg.mu)
    zo_params = prefix
    snapshot = {n: p.detach().clone() for n, p in zo_params.items()}
    accum = {n: torch.zeros_like(p) for n, p in zo_params.items()}
    scalars = []
    cut = max((int(n.split('.layers.')[1].split('.')[0]) for n in prefix)) + 1
    with torch.no_grad():
        hidden, context = wrapper.forward_prefix(batch['input_ids'], batch['attention_mask'], cut)
    h_cut = hidden.detach().requires_grad_(True)
    logits = wrapper.forward_suffix(h_cut, context, cut, last_only=True, token_ids=tokens)[:, -1].float()
    loss = F.cross_entropy(logits, batch['labels'], reduction='mean')
    loss.backward()
    lam = h_cut.grad.detach().float()
    assert lam is not None
    for p in range(p_count):
        u = directions.get(zo_params, local_step, p)
        try:
            with torch.no_grad():
                perturb(zo_params, snapshot, u, mu)
                h_plus, _ = wrapper.forward_prefix(batch['input_ids'], batch['attention_mask'], cut)
                v = torch.sum(lam * (h_plus.float() - hidden.float()))
                scalars.append(v.detach())
                for name in accum:
                    accum[name].add_(u[name] * (v / (p_count * mu)))
        finally:
            restore(zo_params, snapshot, False)
    for name, param in zo_params.items():
        param.grad = accum[name]
    optimizer.step()
    return (loss.detach(), scalars)
