import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

SPECS = {
    "opt": ("facebook/opt-125m", "27dcfa74d334bc871f3234de431e71c6eeba5dd6", 12, [3, 6, 9]),
    "qwen": ("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323", 28, [8, 16, 24]),
    "smollm3": ("HuggingFaceTB/SmolLM3-3B", "a07cc9a04f16550a088caea529712d1d335b0ac1", 36, [9, 18, 27]),
}


@dataclass
class Config:
    model: str = "opt"
    dataset: str = "sst2"
    method: str = "hofl"
    beta: float = 0.75
    seed: int = 42
    partition: str = "iid"
    n: int = 30
    k: int = 6
    E: int = 5
    q: int = 2
    batch_size: int = 16
    rounds: int = 160
    mu: float = 0.001
    lr: float = 0.00001
    prox_lambda: float = 0.001
    weight_decay: float = 0.0005
    eps: float = 1e-8
    sampler: str = "dep_round_v1"
    manifest: str = ""
    model_path: str = ""
    data_path: str = ""
    token_cache: str = ""
    cache_dir: str = "cache"
    output: str = "outputs/run"
    device: str = "cuda"
    num_threads: int = 4
    eval_batch_size: int = 64
    evaluate_every: int = 0
    local_files_only: bool = False

    @property
    def betas(self):
        return (0.9, 0.999)

    amsgrad = False
    foreach = True
    fused = False

    @property
    def max_length(self):
        return {"sst2": 128, "boolq": 256, "sciq": 256, "ag_news": 128, "squad": 576}[self.dataset]

    @property
    def verbalizers(self):
        return {"sst2": [" bad", " good"], "boolq": [" no", " yes"], "sciq": [" A", " B", " C", " D"], "ag_news": [" A", " B", " C", " D"], "squad": []}[self.dataset]

    @property
    def method_label(self):
        return f"HO-FL (\u03b2={self.beta:g})" if self.method == "hofl" else {"fedavg": "FedAvg", "fedprox": "FedProx", "decomfl": "DeComFL-AdamW"}[self.method]

    def resolve(self):
        if self.model not in SPECS or self.dataset not in {"sst2", "boolq", "sciq", "ag_news", "squad"}:
            raise ValueError("Unsupported model or dataset")
        if self.method not in {"fedavg", "fedprox", "decomfl", "hofl"}:
            raise ValueError("Unsupported method")
        if self.partition not in {"iid", "dirichlet_1", "dirichlet_05", "dirichlet_01"} or (self.dataset != "ag_news" and self.partition != "iid"):
            raise ValueError("Only AG News supports non-IID partitions")
        if self.n != 30 or not 1 <= self.k <= self.n or min(self.E, self.q, self.batch_size, self.rounds) < 1:
            raise ValueError("Three resource tiers require N=30; counts must be positive")
        if not 0 <= self.beta <= 1 or self.mu <= 0 or self.lr <= 0:
            raise ValueError("Invalid beta, smoothing radius, or learning rate")
        if self.method != "hofl":
            self.beta = 0.0
        if self.sampler == "auto":
            self.sampler = "dep_round_v1"
        if self.sampler not in {"dep_round_v1", "historical_torch", "historical_numpy_uniform"}:
            raise ValueError("Unsupported sampler")
        if self.beta and self.sampler != "dep_round_v1":
            raise ValueError("Resource-preference sampling requires DepRound")
        if self.evaluate_every == 0:
            self.evaluate_every = max(1, 50 // self.E)
        return self


def parse_config(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    for field in fields(Config):
        name = "--" + field.name.replace("_", "-")
        if field.type is bool:
            parser.add_argument(name, action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(name, type=field.type, default=None)
    args = vars(parser.parse_args(argv))
    path = args.pop("config")
    values = json.loads(path.read_text()) if path else {}
    unknown = set(values) - set(asdict(Config()))
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    values.update({k: v for k, v in args.items() if v is not None})
    return Config(**values).resolve()
