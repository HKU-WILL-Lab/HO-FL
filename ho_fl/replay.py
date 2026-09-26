import csv
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .algorithms import optimizer_for, restore
from .batches import ClientStream
from .config import parse_config
from .io import write_json
from .model import partition_parameters
from .randomness import client_directions, tensor_direction
from .train import Backend

ATOL, RTOL = 1e-7, 1e-5


def snapshot(params):
    return {n: p.detach().cpu().clone() for n, p in params.items()}


def optimizer_snapshot(params, optimizer):
    return {key: {n: optimizer.state[p][key].detach().cpu().clone() for n, p in params.items()}
            for key in ("exp_avg", "exp_avg_sq", "step")}


def subtract(a, b):
    return {n: a[n] - b[n] for n in a}


def replay_client(global_state, cut, packet, fo_delta, cfg):
    prefix, suffix = partition_parameters(global_state, cut)
    assert list(fo_delta) == list(suffix)
    params = {n: torch.nn.Parameter(p.detach().clone()) for n, p in prefix.items()}
    optimizer = optimizer_for(params, cfg)
    trace = []
    for e, step in enumerate(packet):
        assert step["step"] == e and len(step["directions"]) == cfg.q
        optimizer.zero_grad(set_to_none=True)
        before = snapshot(params)
        for param in params.values():
            param.grad = torch.zeros_like(param)
        for direction in step["directions"]:
            scalar = torch.tensor(direction["scalar"], dtype=torch.float32, device=cfg.device)
            for name, param in params.items():
                u = tensor_direction(name, param, direction["seed"])
                param.grad.add_(u * (scalar / (cfg.q * cfg.mu)))
        gradients = snapshot({n: p.grad for n, p in params.items()})
        optimizer.step()
        after = snapshot(params)
        trace.append({"gradient": gradients, "parameter": after, "step_delta": subtract(after, before),
                      "round_delta": subtract(after, snapshot(prefix)), **optimizer_snapshot(params, optimizer)})
    final = {n: p.detach().clone() if n in params else global_state[n] + fo_delta[n]
             for n, p in {**params, **suffix}.items()}
    final = {n: final[n] for n in global_state}
    return final, trace


def compare(reference, reconstructed, **identity):
    assert list(reference) == list(reconstructed)
    ref = torch.cat([v.detach().cpu().reshape(-1).double() for v in reference.values()])
    got = torch.cat([v.detach().cpu().reshape(-1).double() for v in reconstructed.values()])
    diff = got - ref
    ref_norm = float(torch.linalg.vector_norm(ref))
    error_norm = float(torch.linalg.vector_norm(diff))
    return {**identity, "max_abs_error": float(diff.abs().max()), "l2_abs_error": error_norm,
            "reference_l2_norm": ref_norm, "relative_l2_error": error_norm / ref_norm if ref_norm else None,
            "zero_reference_norm": ref_norm == 0,
            "bitwise_equal": all(reference[n].dtype == reconstructed[n].dtype and torch.equal(
                reference[n].detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                reconstructed[n].detach().cpu().contiguous().reshape(-1).view(torch.uint8)) for n in reference),
            "passed": bool(torch.allclose(got, ref, atol=ATOL, rtol=RTOL)), "atol": ATOL, "rtol": RTOL}


def run(cfg):
    fixed = dict(model="opt", dataset="sst2", partition="iid", method="hofl", seed=42,
                 n=30, k=6, E=5, q=2, batch_size=16, rounds=3, mu=.001, lr=1e-5,
                 weight_decay=.0005, eps=1e-8)
    for key, value in fixed.items():
        setattr(cfg, key, value)
    cfg.beta, cfg.sampler = .5, "dep_round_v1"
    cfg.resolve()
    out = Path(cfg.output)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    b = Backend(cfg)
    selected = [cid for cut in (3, 6, 9) for cid in [i for i, c in enumerate(b.cuts) if c == cut][:2]]
    streams = {cid: ClientStream(b.manifest["clients"][cid], cfg.batch_size) for cid in selected}
    state = {n: p.detach().clone() for n, p in b.params.items()}
    rows, wire_records = [], []
    for r in range(1, 4):
        current_delta = {n: torch.zeros_like(p) for n, p in state.items()}
        replay_delta = {n: torch.zeros_like(p) for n, p in state.items()}
        for cid in selected:
            directions = client_directions(cfg.seed, cfg.model, r, cid)
            restore(b.params, state)
            optimizer = b.optimizer()
            cut = b.cuts[cid]
            prefix, suffix = partition_parameters(b.params, cut)
            original_trace, packet = [], []
            for e in range(5):
                before = snapshot(prefix)
                _, scalars = b.local_step(cid, streams[cid].next(), optimizer, directions, e, state)
                after = snapshot(prefix)
                original_trace.append({"gradient": snapshot({n: p.grad for n, p in prefix.items()}),
                                       "parameter": after, "step_delta": subtract(after, before),
                                       "round_delta": subtract(after, snapshot({n: state[n] for n in prefix})),
                                       **optimizer_snapshot(prefix, optimizer)})
                packet.append({"step": e, "directions": [{"seed": directions.used_seeds[e, p], "scalar": scalar.item()} for p, scalar in enumerate(scalars)]})
            fo_delta = {n: p.detach() - state[n] for n, p in suffix.items()}
            replayed, replay_trace = replay_client(state, cut, packet, fo_delta, cfg)
            for e, (reference, reconstructed) in enumerate(zip(original_trace, replay_trace), 1):
                for obj in reference:
                    rows.append(compare(reference[obj], reconstructed[obj], round=r, client=cid, cut=cut, step=e, object=obj))
            final = snapshot(b.params)
            replayed_cpu = snapshot(replayed)
            rows.append(compare(final, replayed_cpu, round=r, client=cid, cut=cut, step=5, object="client_full_parameter"))
            rows.append(compare(subtract(final, snapshot(state)), subtract(replayed_cpu, snapshot(state)), round=r, client=cid, cut=cut, step=5, object="client_full_delta"))
            with torch.no_grad():
                for n, p in b.params.items():
                    current_delta[n].add_(p.detach() - state[n])
                    replay_delta[n].add_(replayed[n] - state[n])
            wire_records.append({"round": r, "client": cid, "cut": cut,
                                 "direction_master_seed": directions.master_seed, "steps": packet})
            del optimizer, original_trace, replay_trace, replayed, directions
        current_next = {n: p.clone() for n, p in state.items()}
        replay_next = {n: p.clone() for n, p in state.items()}
        with torch.no_grad():
            for n in state:
                current_next[n].add_(current_delta[n], alpha=1 / 6)
                replay_next[n].add_(replay_delta[n], alpha=1 / 6)
        rows.append(compare(current_next, replay_next, round=r, client="all", cut="mixed", step=5, object="aggregate_parameter"))
        rows.append(compare(subtract(current_next, state), subtract(replay_next, state), round=r, client="all", cut="mixed", step=5, object="aggregate_delta"))
        state = current_next
        restore(b.params, state)
        print(json.dumps({"round": r, "passed": all(row["passed"] for row in rows), "max_abs_error": max(row["max_abs_error"] for row in rows)}), flush=True)
    with (out / "replay_errors.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "direction_records.json", wire_records)
    summary = {"status": "PASSED" if all(row["passed"] for row in rows) else "FAILED",
               "all_bitwise_equal": all(row["bitwise_equal"] for row in rows), "comparisons": len(rows),
               "max_abs_error": max(row["max_abs_error"] for row in rows),
               "max_relative_l2_error": max(row["relative_l2_error"] or 0 for row in rows),
               "zero_reference_rows": sum(row["zero_reference_norm"] for row in rows),
               "atol": ATOL, "rtol": RTOL, "selected_clients": selected, "cuts": [b.cuts[c] for c in selected],
               "partition_id": b.partition_id, "initial_parameter_sha256": b.initial_hash,
               "reference_path": "formal_dense_mean_delta", "reconstruction_path": "per_client_sequential_adamw_from_seed_scalar_and_FO_delta",
               "round_inputs": "both_paths_restart_from_current_path_global_parameters",
               "device_backend": torch.device(cfg.device).type, "dtype": "float32",
               "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
               "client_training_executions": 18, "local_updates": 90, "server_forward_backward_calls": 0,
               "optimizer_state_reset": "each_client_each_round"}
    write_json(out / "summary.json", summary)
    write_json(out / "config.json", asdict(cfg))
    if summary["status"] != "PASSED":
        raise RuntimeError("Replay failed fixed tolerances; inspect per-step diagnostics")
    return summary


def main():
    run(parse_config())


if __name__ == "__main__":
    main()
