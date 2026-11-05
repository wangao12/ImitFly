# Source map

The release is organized from the local research project. Existing sources were
selected and refactored into reusable modules with command-line arguments.

| Research file | Release location |
|---|---|
| `flybody/LSTMbcA_meansnn_v51.py` | `imitfly/models/policy.py`, `scve.py`, `hmam.py`, `losses.py`, `cli/train.py` |
| `flybody/inference_render_meansnnv51.py` | `imitfly/cli/evaluate.py`, `preprocessing.py`, `env.py` |
| `tests/dataset_MeanStd.py` | `imitfly/cli/collect.py`, `data.py` |
| `tests/LSTMbcA_meanstd_walk.py` | `imitfly/models/walking.py`, walking data and training interfaces |
| `tests/inference_render_meanstd_walk.py` | `imitfly/cli/evaluate_walking.py` |
| `tests/dataset_walking.py` | Walking expert collection interface |
| `tests/bcA_meanstd.py` | `imitfly/models/feedforward.py` |
| `tests/LSTMbc_meanstd.py` | `imitfly/models/lstm.py` |
| `tests/LSTMbcA_meanstd.py` | `imitfly/models/lstm_attention.py` |
| `flybody/fly_envs.py`, `tasks/`, `fruitfly/`, `quaternions.py` | Selected environment sources and runtime assets in `flybody/` |
| Original `README.md` and `LICENSE` | `docs/FLYBODY_UPSTREAM_README.md`, `LICENSE`, `NOTICE` |

The SCVE/HMAM/LSTM ablation switches, NPZ episode format, shared command-line
interfaces, configuration presets, and checkpoint recovery utilities are organized
for this release. Existing model parameter names are retained for the full v51 policy.

Old experiments, later v52/v53/v54 variants, online-distilled checkpoints,
large rollout outputs, IDE files, local caches, manuscript drafts, and private
absolute paths are excluded from the release. Weights and training data are supplied
separately. Importing a model module does not launch training or simulation.
