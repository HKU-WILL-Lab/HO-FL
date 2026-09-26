import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .io import write_json
from .sampling import dep_round, probabilities

CUT_METHOD = {8: "ho_small", 16: "ho_medium", 24: "ho_large"}
SOURCE = "local_sampling_simulation_plus_measured_profile"

def simulate(config, profiles):
    cuts = np.asarray(config['actual_cuts'])
    dims = [config['bottom_dimensions'][str(c)] for c in cuts]
    sampler = config.get('sampler_metadata') or config['sampler']
    preferred = sampler['fixed_preferred_clients']
    assert len(cuts) == 30 and all((sum(cuts == c) == 10 for c in CUT_METHOD))
    lookup = {(int(r.input_length), r.method): r for r in profiles.itertuples()}
    rounds, means, selections = ([], [], [])
    for beta in [0.0, 0.25, 0.5, 0.75]:
        p = np.full(30, (1 - beta) / 30)
        p[preferred] += beta / 6
        check, check_preferred, _ = probabilities(dims, 2, 6, beta, sampler['tie_seed'])
        assert check_preferred.tolist() == preferred and np.allclose(check, p, atol=1e-14, rtol=0)
        inclusion = 6 * p
        rng = np.random.default_rng(42)
        chosen_rounds = [dep_round(inclusion, rng) for _ in range(160)]
        for round_id, selected in enumerate(chosen_rounds, 1):
            assert len(selected) == len(set(selected)) == 6
            selections.append({'beta': beta, 'round': round_id, 'selected_client_ids': selected, 'source': SOURCE, 'simulation_seed': 42})
        for length in [512, 1024]:
            local = []
            for round_id, selected in enumerate(chosen_rounds, 1):
                row = {'length': length, 'beta': beta, 'round': round_id, 'simulation_seed': 42, 'source': SOURCE}
                for cut in CUT_METHOD:
                    row[f'cut{cut}_count'] = int(sum(cuts[selected] == cut))
                for field in ['allocated', 'reserved']:
                    row[f'total_peak_{field}_bytes'] = sum((int(getattr(lookup[length, CUT_METHOD[int(cuts[cid])]], f'peak_{field}_bytes_max')) for cid in selected))
                local.append(row)
            frame = pd.DataFrame(local)
            rounds.extend(local)
            summary = {'length': length, 'beta': beta, 'simulation_seed': 42, 'rounds': 160, 'source': SOURCE, 'variability_scope': 'between_sampling_rounds_not_training_seeds'}
            for cut in CUT_METHOD:
                target = float(inclusion[cuts == cut].sum())
                assert np.isclose(target, 2 + 4 * beta if cut == 8 else 2 * (1 - beta))
                summary[f'cut{cut}_mean_count'] = float(frame[f'cut{cut}_count'].mean())
                summary[f'cut{cut}_expected_count'] = target
            for field in ['allocated', 'reserved']:
                values = frame[f'total_peak_{field}_bytes']
                expected = sum((inclusion[cid] * int(getattr(lookup[length, CUT_METHOD[int(cut)]], f'peak_{field}_bytes_max')) for cid, cut in enumerate(cuts)))
                mean, sd = (float(values.mean()), float(values.std(ddof=1)))
                summary[f'total_peak_{field}_bytes_mean'] = mean
                summary[f'total_peak_{field}_bytes_round_sample_sd'] = sd
                summary[f'total_peak_{field}_bytes_analytic_expectation'] = float(expected)
                summary[f'total_peak_{field}_bytes_mean_minus_expectation'] = mean - expected
                summary[f'{field}_within_six_round_standard_errors'] = bool(abs(mean - expected) <= 6 * sd / np.sqrt(160) + 1)
                assert summary[f'{field}_within_six_round_standard_errors']
            means.append(summary)
    return (pd.DataFrame(rounds), pd.DataFrame(means), selections)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/memory"))
    args = parser.parse_args()
    config = json.loads(args.source_config.read_text())
    profiles = pd.read_csv(args.profiles)
    profiles = profiles[profiles.input_length.isin([512, 1024]) & profiles.method.isin(["inference", "fedavg", "decomfl", *CUT_METHOD.values()])]
    assert len(profiles) == 12 and profiles.status.eq("COMPLETED").all()
    rounds, means, selections = simulate(config, profiles)
    args.output.mkdir(parents=True, exist_ok=True)
    rounds.to_csv(args.output / "round_totals.csv", index=False)
    means.to_csv(args.output / "round_means.csv", index=False)
    write_json(args.output / "selections.json", selections)
    write_json(args.output / "source_config.json", config)
    write_json(args.output / "protocol.json", {"source": SOURCE, "simulation_seed": 42,
               "N": 30, "K": 6, "rounds": 160, "betas": [0, .25, .5, .75],
               "variability_scope": "between_sampling_rounds_not_training_seeds",
               "analytic_expectation": "sum_i K*p_i*measured_peak(cut_i)",
               "new_gpu_measurements": 0, "training_ledger_used": False})
    print(json.dumps({"round_rows": len(rounds), "summary_rows": len(means)}))


if __name__ == "__main__":
    main()
