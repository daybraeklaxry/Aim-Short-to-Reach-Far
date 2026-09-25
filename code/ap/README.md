# Anchored Planning

This directory contains retrieval, action selection, and simulator evaluation
for frozen LeWM models. The paper's recorded results are in `../../data/`.
The released LeWM baseline is `../lewm/baseline/official_baseline.py`.

For paper results, run `python tools/reproduce_table1.py` from the repository
root. See [REPRODUCING.md](../../REPRODUCING.md) for the full experiment map.

## Install the controller

From this directory, install the analysis dependencies and package:

```bash
python -m pip install -r requirements/analysis.txt
python -m pip install --no-deps -e .
python -m anchored_planning --help
```

## Action selection

All methods use the same retrieval rule and five-action blocks. Direct executes
the closest record's action block. AP-rank scores eight retrieved blocks with
the frozen predictor, using an observed successor as the target. Final-goal
ranking scores the same blocks against the goal image. AP-CEM searches for new
actions toward the observed target, and final-goal CEM uses the same search
toward the goal image.

CEM uses 300 candidates, 30 iterations, and 30 elites. It also compares the
initial and final means, giving 31 prediction batches and 9,002 predicted blocks
per decision. Proposals are clipped to the action bounds before scoring and
execution. CEM does not use the recorded action values to initialize its search.

## External assets

Obtain the released model state dictionaries and original HDF5 datasets
from the [LeWM checkpoints and data collection](https://huggingface.co/collections/quentinll/lewm).
Preserve the original dataset row order, episode lengths, and offsets:
the frozen protocols refer directly to those row indices.

| Task | Model repository | Original HDF5 filename | Rows / episodes | Reset-state column |
| --- | --- | --- | --- | --- |
| Cube | [quentinll/lewm-cube](https://huggingface.co/quentinll/lewm-cube) | `cube_single_expert.h5` | 2,010,000 / 10,000 | `observation` (28) |
| PushT | [quentinll/lewm-pusht](https://huggingface.co/quentinll/lewm-pusht) | `pusht_expert_train.h5` | 2,336,736 / 18,685 | `state` (7) |
| Reacher | [quentinll/lewm-reacher](https://huggingface.co/quentinll/lewm-reacher) | `reacher.h5` | 2,010,000 / 10,000 | `observation` (6) |
| TwoRoom | [quentinll/lewm-tworooms](https://huggingface.co/quentinll/lewm-tworooms) | `tworoom.h5` | 920,809 / 10,000 | `observation` (10) |

Each model repository supplies `weights.pt`. The recorded PushT artifact
was named `official_weights_compat.pt`. The loader accepts the old and new
Transformers ViT key layouts and performs a strict 303-key load. It does not
train or replace weights. `protocols/asset_inventory.json` preserves the
recorded filenames. Public repositories may change, and their download
revisions were not preserved in this package.

All datasets must expose `pixels` (uint8, `[rows,224,224,3]`), `action`,
`ep_len`, `ep_offset`, and the reset-state column above. Cube actions have
five coordinates. The other tasks have two. HDF5 compression requires
`hdf5plugin`. The included `protocols/scalers.json` contains the original
full-dataset float64 StandardScaler records, excluding non-finite action
rows. These are the fixed training/evaluation normalization artifacts.
they are not recomputed from a selected memory bank.

Edit `examples/assets.json` or copy it to your own JSON file. Each path is
resolved relative to that JSON file. Absolute paths are also accepted.
The example expects `assets/checkpoints/<task>/weights.pt`, the HDF5 files
under `assets/data/`, and caches under `assets/cache/`. No download is
triggered automatically.

## Native environment setup

The completed experiment used Linux, Python 3.11.14, PyTorch 2.7.1 with CUDA
12.6, torchvision 0.22.1, and `stable-worldmodel` 0.0.6. Install the common
native environment in a separate virtual environment:

```bash
python3.11 -m venv .venv-native
source .venv-native/bin/activate
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements/native.txt
python -m pip install --no-deps -e .
```

The original renderer used Mesa OSMesa/llvmpipe, not GPU rendering. On
Debian/Ubuntu the system libraries include `libosmesa6`, `libgl1`,
`libegl1`, and `libglib2.0-0`. Install these through your OS package manager.
The entry point sets `MUJOCO_GL=osmesa`, `PYOPENGL_PLATFORM=osmesa`,
`GALLIUM_DRIVER=llvmpipe`, `LP_NUM_THREADS=1`, and `SDL_VIDEODRIVER=dummy`
unless already configured. It records the effective renderer settings.

**Reacher needs its own environment:** use the same commands with a
`.venv-reacher` folder and `requirements/native-reacher.txt`. Its recorded
runtime uses MuJoCo 3.10.0 and DM-Control 1.0.43. Cube, PushT, and TwoRoom
use MuJoCo 3.12.0 and DM-Control 1.0.44. The evaluator checks this pair before
running. This difference is required for the released Reacher observations.

The native requirements pin the directly used packages observed in the
runtime. They are not a complete OS image or a tested clean-install lock
of every transitive dependency. See `NOTICE.md` for source provenance.

## Prepare retrieval caches and run new evaluations

After setting your asset paths:

```bash
python -m anchored_planning inspect-assets --assets examples/assets.json --task cube
python -m anchored_planning encode-cache --assets examples/assets.json --task cube --device cuda:0 --batch-size 128
python -m anchored_planning evaluate --assets examples/assets.json --task cube --study main --output outputs/new-main --limit 1
python -m anchored_planning evaluate --assets examples/assets.json --task cube --study local-target --output outputs/new-local --limit 1
```

`inspect-assets` reads dataset metadata and reports dependencies and cache
shape. It does not run a simulation. If you have the recorded `.pt` latent
cache, point the config to it. Otherwise `encode-cache` creates an `.npy`
cache in original row order, encoding in batches and storing float16
latents. The loader converts them to float32 for retrieval. CUDA retrieval
encoding uses the recorded bfloat16 autocast path. Native planning/target
encoding uses projected float32 features without autocast. Cache generation
over millions of images is a substantial external computation.

The example `--limit 1` is a small new run, not a result reproduction check.
Omit `--limit` for all 128 frozen queries of a task. Repeat for all four tasks
and each study, switching to the Reacher environment for Reacher. Each task
executes all three starts and all its study arms. For separate workers,
use `--shards 2 --shard-index 0` and `--shards 2 --shard-index 1` with distinct
GPU visibility. Workers can share the output folder because their outcome
files and query traces are disjoint. Do not change shard settings within
an existing output folder. Frozen query ordinals preserve the prescribed
optimizer seeds.

The runner writes `outcomes_<task>_shard<i>of<n>.jsonl`, per-query traces,
run metadata, and a shard summary. It reconstructs reset states and fixed
prefixes from dataset indices, checks the same starting image and prefix
outcome across paired arms, and checks actual predictor work. It treats a
terminal event without physical success as failure. At least 24 GB free
GPU memory per worker was used in the original launcher. This is a recorded
configuration, not a newly measured minimum.

Collect the completed task/shard outcome files for a study in one folder:

```bash
python -m anchored_planning analyze --study main --outcomes outputs/new-main --output outputs/new-main-analysis
python -m anchored_planning analyze --study local-target --outcomes outputs/new-local --output outputs/new-local-analysis
```

The analyzer rejects incomplete cohorts and duplicate outcomes. It summarizes
the new evaluation output. For the paper's recorded results, use the root
`tools/reproduce_table1.py` command.

## Source layout

- `anchored_planning/`: controller, retrieval, action handling, and evaluation.
- `protocols/`: query identities, memory membership, and action-normalizer settings.
- `examples/`: external-asset configuration.
- `requirements/`: analysis and task-specific simulator dependencies.
- `tests/`: action handling, retrieval, and optimizer checks.

The upstream MIT notice is preserved in `LICENSE`. See `NOTICE.md` for attribution.
