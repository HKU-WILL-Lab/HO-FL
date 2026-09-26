import hashlib
import json
from pathlib import Path


def tensor_hash(params):
    digest = hashlib.sha256()
    for name, param in params.items():
        digest.update(name.encode())
        digest.update(param.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)

def append_jsonl(path, obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(obj, sort_keys=True, allow_nan=False) + '\n')
        f.flush()

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
