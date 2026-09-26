# HO-FL

## Install

Python 3.11 and a CUDA GPU supporting BF16 (PyTorch 2.6.0 / CUDA 12.4):

```bash
python3 -m pip install uv
uv venv --python 3.11
uv pip install --torch-backend cu124 .
source .venv/bin/activate
```

## Train

```bash
ho-fl-train --model opt --dataset sst2 --method hofl --beta 0.75 --seed 42 --output outputs/opt-sst2
ho-fl-train --model qwen --dataset ag_news --partition dirichlet_05 --method hofl --beta 0.5 --seed 43 --output outputs/qwen-agnews
ho-fl-train --help
```

| Option | Values / Default |
| --- | --- |
| `--model` | `opt`, `qwen`, `smollm3` |
| `--dataset` | `sst2`, `boolq`, `sciq`, `ag_news`, `squad` |
| `--method` | `fedavg`, `fedprox`, `decomfl`, `hofl` |
| `--beta` | 0 to 1; default 0.75 for `hofl` |
| `--seed` | Base random seed; default 42 |
| `--partition` | `iid`; AG News also accepts `dirichlet_1`, `dirichlet_05`, `dirichlet_01` |
| `--k`, `--E`, `--q`, `--batch-size`, `--rounds` | Defaults: 6, 5, 2, 16, 160 |
| `--lr`, `--mu`, `--prox-lambda` | Defaults: 1e-5, 0.001, 0.001 |
| `--output`, `--cache-dir` | Output and download/cache directories |
| `--model-path`, `--data-path` | Optional local model and Datasets `save_to_disk` directories |
| `--manifest`, `--token-cache` | Optional local partition and tokenized-data files |
| `--device`, `--local-files-only` | Device selection and offline model loading |

Without local paths, models and datasets download automatically and client partitions are generated from `--seed`. For E=1/5/10 with equal processed-sample budgets, set `--rounds 800/160/80`, respectively.

## Other Commands

```bash
ho-fl-profile --method ho_small --length 512 --output outputs/profile.json
ho-fl-profile --help
ho-fl-memory --profiles profiles.csv --source-config outputs/qwen-seed42/config.json --output outputs/memory
ho-fl-memory --help
ho-fl-replay --output outputs/replay
ho-fl-replay --help
```
