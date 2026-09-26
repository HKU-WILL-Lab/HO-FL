import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from datasets import load_dataset, load_from_disk

from .config import SPECS
from .io import write_json
from .partitioning import allocate, numpy_manifest, squad_manifest, torch_manifest
from .randomness import stable_seed
from .tokenization import sciq_tokens, squad_tokens

DATASETS = {
    "sst2": ("stanfordnlp/sst2", None, "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb", 67349, 872),
    "boolq": ("aps/super_glue", "boolq", "3de24cf8022e94f4ee4b9d55a6f539891524d646", 9427, 3270),
    "sciq": ("allenai/sciq", None, "2c94ad3e1aafab77146f384e23536f97a4849815", 11679, 1000),
    "ag_news": ("fancyzhx/ag_news", None, "eb185aade064a813bc0b7f42de02595523103ca4", 120000, 7600),
    "squad": ("rajpurkar/squad", None, "7b6d24c440a36b6815f21b70d25016731768db1f", 87599, 10570),
}


def load_tokens(cfg, tokenizer):
    if cfg.token_cache:
        result = json.loads(Path(cfg.token_cache).read_text())
    else:
        key = f"{cfg.dataset}_{cfg.model}_{cfg.seed if cfg.dataset == 'sciq' else 'shared'}"
        path = Path(cfg.cache_dir) / "tokens" / f"{key}.json"
        if path.exists():
            result = json.loads(path.read_text())
        else:
            identifier, subset, revision, _, _ = DATASETS[cfg.dataset]
            raw = load_from_disk(cfg.data_path) if cfg.data_path else load_dataset(identifier, subset, revision=revision, cache_dir=cfg.cache_dir)
            if cfg.dataset == "squad":
                result = squad_tokens(raw, tokenizer, {"revision": SPECS[cfg.model][1]}, {"revision": revision})
            else:
                result = {}
                eval_split = "test" if cfg.dataset in {"ag_news", "sciq"} else "validation"
                for split, target in [("train", "train"), (eval_split, "validation")]:
                    ds = raw[split]
                    if cfg.dataset == "sciq":
                        rng = np.random.default_rng(50000 + cfg.seed + (1000 if split == "test" else 0))
                        pairs = [sciq_tokens(row, tokenizer, rng) for row in ds]
                        ids, labels = map(list, zip(*pairs))
                    else:
                        if cfg.dataset == "sst2":
                            prompts = [s + " It was" for s in ds["sentence"]]
                        elif cfg.dataset == "boolq":
                            prompts = [f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:" for r in ds]
                        else:
                            prompts = ["News: " + r["text"] + "\nCategory (A World, B Sports, C Business, D Sci-Tech):" for r in ds]
                        ids = tokenizer(prompts, truncation=True, max_length=cfg.max_length, padding=False, add_special_tokens=True)["input_ids"]
                        labels = [int(x) for x in ds["label"]]
                    result[target] = {"input_ids": ids, "labels": labels}
            write_json(path, result)
    expected = DATASETS[cfg.dataset][-2:]
    assert tuple(len(result[s]["input_ids"]) for s in ("train", "validation")) == expected
    if cfg.dataset == "sciq" and "metadata" in result:
        assert result["metadata"]["option_seed"] == 50000 + cfg.seed
    return result


def generate_manifest(cfg, train):
    adapter = SimpleNamespace(seed=cfg.seed, model=SimpleNamespace(key=cfg.model, zo_layer_choices=SPECS[cfg.model][3]), data=SimpleNamespace(num_clients=cfg.n))
    if cfg.dataset == "squad":
        return squad_manifest(adapter, train)
    if cfg.dataset == "sst2" and cfg.model == "opt":
        return numpy_manifest(train["labels"], "opt", "iid", cfg.seed)
    if cfg.dataset in {"sst2", "boolq"}:
        return torch_manifest(adapter, train["labels"])
    labels = train["labels"]
    alpha = {"iid": None, "dirichlet_1": 1.0, "dirichlet_05": .5, "dirichlet_01": .1}[cfg.partition]
    clients, processing = allocate(labels, alpha, cfg.seed)
    assignment = torch.randperm(30, generator=torch.Generator().manual_seed(20000 + cfg.seed)).tolist()
    cuts = [0] * 30
    for j, cid in enumerate(assignment):
        cuts[cid] = SPECS[cfg.model][3][j // 10]
    for cid, client in enumerate(clients):
        client.update(client_id=cid, mixed_L_zo=cuts[cid], data_seed=stable_seed(cfg.seed, cfg.model, cid, "data"), unique_examples=len(set(client["indices"])), label_counts=np.bincount(np.asarray(labels)[client["indices"]], minlength=4).tolist())
    flat = [i for c in clients for i in c["indices"]]
    return {"dataset": cfg.dataset, "model_key": cfg.model, "partition": cfg.partition, "base_seed": cfg.seed,
            "partition_seed": 10000 + cfg.seed, "cut_assignment_seed": 20000 + cfg.seed,
            "protocol_id": "v4_torch_iid_v1" if alpha is None else "v4_dirichlet_depletion_first_v1",
            "client_processing_order": processing, "alpha": alpha, "examples_per_client": len(labels) // 30,
            "pooled_unique_examples": len(set(flat)), "repeated_fill_count": sum(c["repeated_fill_count"] for c in clients),
            "unused_indices": sorted(set(range(len(labels))) - set(flat)), "clients": clients}


def load_manifest(cfg, train):
    if cfg.manifest:
        path = Path(cfg.manifest)
    else:
        path = Path(cfg.cache_dir) / "manifests" / cfg.model / cfg.dataset / cfg.partition / f"seed_{cfg.seed}.json"
        if not path.exists():
            write_json(path, generate_manifest(cfg, train))
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    partition_id = hashlib.sha256(raw).hexdigest()
    record = json.loads(raw)
    assert record["base_seed"] == cfg.seed
    assert record.get("model_key", record.get("domain")) == cfg.model
    assert record.get("partition", "iid") == cfg.partition
    assert len(record["clients"]) == cfg.n
    assert [c["client_id"] for c in record["clients"]] == list(range(cfg.n))
    assert len({len(c["indices"]) for c in record["clients"]}) == 1
    for c in record["clients"]:
        assert len(c["indices"]) >= cfg.batch_size
        assert min(c["indices"]) >= 0 and max(c["indices"]) < len(train["input_ids"])
        assert c["mixed_L_zo"] in SPECS[cfg.model][3]
        if "label_counts" in c:
            assert np.bincount(np.asarray(train["labels"])[c["indices"]], minlength=len(c["label_counts"])).tolist() == c["label_counts"]
    assert all(sum(c["mixed_L_zo"] == cut for c in record["clients"]) == 10 for cut in SPECS[cfg.model][3])
    return record, partition_id, raw
