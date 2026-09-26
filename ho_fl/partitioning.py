import numpy as np
import torch
from .randomness import stable_seed

def allocate(labels, alpha, seed, n=30):
    labels = np.asarray(labels, dtype=int)
    classes, size = (int(labels.max()) + 1, len(labels) // n)
    rng = np.random.default_rng(10000 + seed)
    if alpha is None:
        order = torch.randperm(len(labels), generator=torch.Generator().manual_seed(10000 + seed)).tolist()
        return ([{'indices': order[c * size:(c + 1) * size], 'repeated_fill_count': 0} for c in range(n)], list(range(n)))
    prior = np.bincount(labels, minlength=classes) / len(labels)
    pools = [rng.permutation(np.flatnonzero(labels == c)) for c in range(classes)]
    cursor = np.zeros(classes, dtype=int)
    client_order = rng.permutation(n).tolist()
    clients = [None] * n
    for cid in client_order:
        theta = rng.dirichlet(classes * alpha * prior)
        requested = rng.choice(classes, size=size, p=theta)
        indices, repeated = (np.empty(size, dtype=int), 0)
        for c in range(classes):
            positions = np.flatnonzero(requested == c)
            fresh = int(min(len(positions), len(pools[c]) - cursor[c]))
            indices[positions[:fresh]] = pools[c][cursor[c]:cursor[c] + fresh]
            cursor[c] += fresh
            shortage = len(positions) - fresh
            if shortage:
                indices[positions[fresh:]] = rng.choice(pools[c], size=shortage, replace=True)
                repeated += shortage
        clients[cid] = {'indices': indices.tolist(), 'repeated_fill_count': repeated, 'target_label_probabilities': theta.tolist()}
    return (clients, client_order)


def torch_manifest(cfg, labels):
    partition_rng = torch.Generator().manual_seed(10000 + cfg.seed)
    shuffled = torch.randperm(len(labels), generator=partition_rng).tolist()
    per_client = len(labels) // cfg.data.num_clients
    used = per_client * cfg.data.num_clients
    assignment_rng = torch.Generator().manual_seed(20000 + cfg.seed)
    assignment = torch.randperm(cfg.data.num_clients, generator=assignment_rng).tolist()
    cuts = [None] * cfg.data.num_clients
    for j, cid in enumerate(assignment):
        cuts[cid] = cfg.model.zo_layer_choices[j // 10]
    clients = []
    for cid in range(cfg.data.num_clients):
        indices = shuffled[cid * per_client:(cid + 1) * per_client]
        clients.append({'client_id': cid, 'indices': indices, 'mixed_L_zo': cuts[cid], 'label_counts': np.bincount([labels[i] for i in indices], minlength=2).tolist(), 'data_seed': stable_seed(cfg.seed, cfg.model.key, cid, 'data')})
    manifest = {'model_key': cfg.model.key, 'base_seed': cfg.seed, 'partition_seed': 10000 + cfg.seed, 'cut_assignment_seed': 20000 + cfg.seed, 'examples_per_client': per_client, 'unused_indices': shuffled[used:], 'clients': clients}
    return manifest

def squad_manifest(cfg, data):
    count = len(data['input_ids'])
    shuffled = torch.randperm(count, generator=torch.Generator().manual_seed(10000 + cfg.seed)).tolist()
    size = count // 30
    order = torch.randperm(30, generator=torch.Generator().manual_seed(20000 + cfg.seed)).tolist()
    cuts = [None] * 30
    for j, cid in enumerate(order):
        cuts[cid] = cfg.model.zo_layer_choices[j // 10]
    clients = []
    for cid in range(30):
        indices = shuffled[cid * size:(cid + 1) * size]
        clients.append({'client_id': cid, 'indices': indices, 'mixed_L_zo': cuts[cid], 'data_seed': stable_seed(cfg.seed, cfg.model.key, cid, 'data'), 'mean_answer_tokens': float(np.mean([len(data['answer_ids'][i]) for i in indices]))})
    manifest = {'dataset': 'squad', 'model_key': cfg.model.key, 'base_seed': cfg.seed, 'partition_seed': 10000 + cfg.seed, 'cut_assignment_seed': 20000 + cfg.seed, 'examples_per_client': size, 'unused_indices': shuffled[30 * size:], 'clients': clients}
    return manifest

def numpy_manifest(labels, domain, partition, seed):
    labels = np.asarray(labels, dtype=int)
    n, cuts, counts = (30, [3, 6, 9], [10, 10, 10])
    size, classes = (len(labels) // n, int(labels.max()) + 1)
    assignment = torch.randperm(n, generator=torch.Generator().manual_seed(20000 + seed)).tolist()
    client_cuts = [0] * n
    for cid, cut in zip(assignment, np.repeat(cuts, counts)):
        client_cuts[cid] = int(cut)
    rng = np.random.default_rng(10000 + seed)
    alpha = None
    assert domain == 'opt' and partition == 'iid'
    prior = np.bincount(labels, minlength=classes) / len(labels)
    pools = [np.flatnonzero(labels == c) for c in range(classes)]
    shuffled = rng.permutation(len(labels)) if alpha is None else None
    clients = []
    for cid in range(n):
        probabilities = prior if alpha is None else rng.dirichlet(alpha * classes * prior)
        if alpha is None:
            indices = shuffled[cid * size:(cid + 1) * size]
        clients.append({'client_id': cid, 'indices': indices.tolist(), 'mixed_L_zo': client_cuts[cid], 'label_counts': np.bincount(labels[indices], minlength=classes).tolist(), 'target_label_probabilities': probabilities.tolist(), 'unique_examples': len(set(indices.tolist())), 'data_seed': stable_seed(seed, domain, cid, 'data'), 'augmentation_seed': stable_seed(seed, domain, cid, 'augmentation')})
    flat = [i for c in clients for i in c['indices']]
    histogram = np.sum([c['label_counts'] for c in clients], axis=0)
    result = {'domain': domain, 'partition': partition, 'alpha': alpha, 'base_seed': seed, 'partition_seed': 10000 + seed, 'cut_assignment_seed': 20000 + seed, 'num_clients': n, 'examples_per_client': size, 'num_classes': classes, 'sampling_with_replacement_within_and_across_clients': alpha is not None, 'dirichlet_concentration': None if alpha is None else (alpha * classes * prior).tolist(), 'source_label_counts': np.bincount(labels, minlength=classes).tolist(), 'pooled_label_counts': histogram.tolist(), 'pooled_unique_examples': len(set(flat)), 'unused_indices': sorted(set(range(len(labels))) - set(flat)), 'clients': clients, 'label_counts_by_cut': {str(cut): np.sum([c['label_counts'] for c in clients if c['mixed_L_zo'] == cut], axis=0).tolist() for cut in cuts}}
    return result
