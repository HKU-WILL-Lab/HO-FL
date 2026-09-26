import argparse
import gc
from pathlib import Path

import torch

from .algorithms import hybrid_step, optimizer_for, restore, task_loss
from .config import Config
from .decomfl import CentralEstimator, estimator_step
from .io import write_json
from .model import load_model, partition_parameters
from .randomness import RoundDirections


def measure(method, length, cfg):
    model, wrapper, params, tokenizer, tokens = load_model(cfg)
    cut = {"inference": 0, "fedavg": 0, "decomfl": 28, "ho_small": 8, "ho_medium": 16, "ho_large": 24}[method]
    prefix, _ = partition_parameters(params, cut)
    batch = {"input_ids": torch.full((16, length), tokenizer.encode("good", add_special_tokens=False)[0], device=cfg.device, dtype=torch.long),
             "attention_mask": torch.ones((16, length), device=cfg.device, dtype=torch.long),
             "labels": torch.arange(16, device=cfg.device) % 2}
    original = {n: p.detach().cpu().clone() for n, p in params.items()}
    rows = []
    for repeat in range(-1, 3):
        restore(params, original)
        for p in params.values():
            p.grad = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(cfg.device)
        torch.cuda.reset_peak_memory_stats(cfg.device)
        optimizer = None if method == "inference" else optimizer_for(params, cfg)
        directions = None if method == "inference" else RoundDirections(41242)
        estimator = CentralEstimator(params, cfg.mu, cfg.q) if method == "decomfl" else None
        for e in range(1 if method == "inference" else 5):
            if method == "inference":
                with torch.no_grad():
                    value = wrapper.forward(batch["input_ids"], batch["attention_mask"], token_ids=tokens)
                scalars = []
            elif method == "fedavg":
                optimizer.zero_grad(set_to_none=True)
                value = task_loss(wrapper, batch, tokens)
                value.backward()
                optimizer.step()
                scalars = []
            elif method == "decomfl":
                with torch.no_grad():
                    value = task_loss(wrapper, batch, tokens).detach()
                perturbed_loss, scalars = estimator_step(estimator, wrapper, batch, tokens, optimizer, directions, e)
                del perturbed_loss
            else:
                value, scalars = hybrid_step(cfg, wrapper, params, prefix, batch, tokens, optimizer, directions, e)
            assert torch.isfinite(value).all()
            del value, scalars
        torch.cuda.synchronize(cfg.device)
        if repeat >= 0:
            rows.append({"repeat": repeat, "peak_allocated_bytes": torch.cuda.max_memory_allocated(cfg.device),
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved(cfg.device)})
        del optimizer, directions, estimator
    return {"model": "qwen", "method": method, "input_length": length, "batch_size": 16,
            "E": 0 if method == "inference" else 5, "q": 2 if cut else 0,
            "cut_layer": cut, "bottom_dimension": sum(p.numel() for p in prefix.values()),
            "total_trainable": sum(p.numel() for p in params.values()), "measurements": rows,
            "status": "COMPLETED", "no_server_gpu_copy": True,
            "scope": "client_only_inference_forward" if method == "inference" else "client_only_full_local_update_including_optimizer_initialization",
            "batch_kind": "synthetic_full_length_classification", "warmups": 1, "repetitions": 3,
            "extra_unperturbed_loss_forwards": 5 if method == "decomfl" else 0,
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["inference", "fedavg", "decomfl", "ho_small", "ho_medium", "ho_large"], required=True)
    parser.add_argument("--length", type=int, choices=[512, 1024], required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.device(args.device).type != "cuda":
        raise ValueError("CUDA peak-memory profiling requires a CUDA device")
    if torch.device(args.device).index is not None:
        torch.cuda.set_device(torch.device(args.device))
    cfg = Config(model="qwen", model_path=args.model_path, cache_dir=args.cache_dir,
                 device=args.device, local_files_only=args.local_files_only)
    try:
        row = measure(args.method, args.length, cfg)
    except torch.cuda.OutOfMemoryError:
        row = {"status": "OOM", "method": args.method, "input_length": args.length, "batch_reduced": False, "measurements": []}
    write_json(args.output, row)


if __name__ == "__main__":
    main()
