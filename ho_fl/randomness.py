import hashlib
import json
import torch

def stable_seed(*parts):
    payload = json.dumps(parts, ensure_ascii=True, separators=(',', ':')).encode('ascii')
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little')

def tensor_direction(name, parameter, direction_seed):
    generator = torch.Generator(device=parameter.device)
    generator.manual_seed(stable_seed(direction_seed, name))
    return torch.randn(parameter.shape, generator=generator, device=parameter.device, dtype=parameter.dtype)

class RoundDirections:

    def __init__(self, master_seed):
        self.master_seed = master_seed
        self.cache = {}
        self.used_seeds = {}

    def get(self, params, local_step, perturbation):
        seed = stable_seed(self.master_seed, local_step, perturbation)
        self.used_seeds[local_step, perturbation] = seed
        result = {}
        for name, param in params.items():
            key = (local_step, perturbation, name)
            if key not in self.cache:
                self.cache[key] = tensor_direction(name, param, seed)
            result[name] = self.cache[key]
        return result


def client_directions(seed, model, round_id, client_id):
    return RoundDirections(stable_seed(seed, model, round_id, client_id, "zo_master"))
