import numpy as np

def probabilities(dimensions, directions, k, beta, tie_seed):
    dims = np.asarray(dimensions, dtype=float)
    q = np.broadcast_to(np.asarray(directions, dtype=float), dims.shape)
    if dims.ndim != 1 or not len(dims) or (not np.isfinite(dims).all()) or np.any(dims < 0):
        raise ValueError('Dimensions must be a finite nonnegative vector')
    if not np.isfinite(q).all() or np.any(q <= 0):
        raise ValueError('Direction counts must be finite and positive')
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= len(dims):
        raise ValueError('K must be an integer in [1, N]')
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError('beta must lie in [0, 1]')
    scores = dims / q
    order = np.random.default_rng(tie_seed).permutation(len(dims))
    order = order[np.argsort(scores[order], kind='stable')]
    preferred = order[:k]
    p = np.full(len(dims), 1 / len(dims))
    if not np.all(scores == scores[0]):
        p *= 1 - beta
        p[preferred] += beta / k
    assert np.isclose(p.sum(), 1) and np.all(k * p <= 1 + 1e-12)
    return (p, preferred, scores)

def dep_round(x, rng, tolerance=1e-12):
    values = np.asarray(x, dtype=float).copy()
    if values.ndim != 1 or not len(values) or (not np.isfinite(values).all()):
        raise ValueError('Expected a nonempty finite vector')
    if np.any(values < -tolerance) or np.any(values > 1 + tolerance):
        raise ValueError('Inclusion probabilities must be in [0, 1]')
    k = int(round(values.sum()))
    if abs(values.sum() - k) > tolerance * len(values):
        raise ValueError('Inclusion probabilities must sum to an integer')
    values = np.clip(values, 0, 1)
    order = rng.permutation(len(values))
    active = [int(i) for i in order if tolerance < values[i] < 1 - tolerance]
    while len(active) >= 2:
        i, j = (active.pop(), active.pop())
        a = min(1 - values[i], values[j])
        b = min(values[i], 1 - values[j])
        if rng.random() < b / (a + b):
            values[i] += a
            values[j] -= a
        else:
            values[i] -= b
            values[j] += b
        for index in (i, j):
            if values[index] <= tolerance:
                values[index] = 0
            elif values[index] >= 1 - tolerance:
                values[index] = 1
            else:
                active.append(index)
    if active:
        i = active.pop()
        if min(abs(values[i]), abs(1 - values[i])) > tolerance * len(values):
            raise ArithmeticError('Nonintegral remainder after dependent rounding')
        values[i] = round(values[i])
    selected = order[values[order] > 0.5]
    if len(selected) != k or len(np.unique(selected)) != k:
        raise ArithmeticError('Dependent rounding cardinality failure')
    return selected.tolist()

class DepRoundSampler:

    def __init__(self, dimensions, directions, k, beta, seed):
        self.tie_seed = 40000 + seed
        self.round_seed = 30000 + seed
        self.p, self.preferred, self.scores = probabilities(dimensions, directions, k, beta, self.tie_seed)
        self.k = k
        self.rng = np.random.default_rng(self.round_seed)

    def sample(self):
        return dep_round(self.k * self.p, self.rng)

    def metadata(self):
        return {'sampler': 'dep_round_v1', 'tie_seed': self.tie_seed, 'round_seed': self.round_seed, 'fixed_preferred_clients': self.preferred.tolist(), 'scores_d_over_q': self.scores.tolist(), 'p': self.p.tolist(), 'target_inclusion_probabilities': (self.k * self.p).tolist()}
