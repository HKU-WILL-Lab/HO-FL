import re

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import SPECS
from .split import OPTPartition, QwenPartition, SmolLM3Partition


def load_model(cfg):
    torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.use_deterministic_algorithms(False)
    identifier, revision, depth, _ = SPECS[cfg.model]
    source = cfg.model_path or identifier
    kwargs = dict(local_files_only=cfg.local_files_only, cache_dir=cfg.cache_dir)
    if not cfg.model_path:
        kwargs["revision"] = revision
    lm = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.bfloat16, attn_implementation="sdpa", **kwargs)
    assert lm.config.num_hidden_layers == depth
    lm.config.use_cache = False
    lm.gradient_checkpointing_disable()
    peft = get_peft_model(lm, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none", target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", init_lora_weights=True, use_rslora=False, use_dora=False, fan_in_fan_out=False))
    peft.to(cfg.device)
    params = dict(sorted((n, p) for n, p in peft.named_parameters() if p.requires_grad))
    for param in params.values():
        param.data = param.data.float()
    assert all("lora_" in n and p.dtype == torch.float32 for n, p in params.items())
    assert all(p.dtype == torch.bfloat16 for p in peft.parameters() if not p.requires_grad)
    peft.eval()
    wrapper = {"opt": OPTPartition, "qwen": QwenPartition, "smollm3": SmolLM3Partition}[cfg.model](peft.get_base_model())
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    tokenizer.padding_side = tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = [tokenizer.encode(v, add_special_tokens=False) for v in cfg.verbalizers]
    assert all(len(v) == 1 for v in ids), ids
    tokens = torch.tensor([v[0] for v in ids], device=cfg.device)
    return peft, wrapper, params, tokenizer, tokens


def block_index(name):
    match = re.search(r"\.layers\.(\d+)\.", name)
    assert match, name
    return int(match.group(1))


def partition_parameters(params, cut):
    return ({n: p for n, p in params.items() if block_index(n) < cut},
            {n: p for n, p in params.items() if block_index(n) >= cut})
