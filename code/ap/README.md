# Aim Short to Reach Far: Goal Distance Underestimates Your Frozen World Model

This directory preserves the original implementation and earlier result snapshots. Its local analysis commands reproduce those snapshots. For the current manuscript results, use `../../data/`; for the released LeWM baseline, use `../lewm/audit/official_baseline.py`. See the repository root README for all current entry points.

Code and compact results for planning toward retrieved observed targets with a
frozen LeWM world model. This release includes the five-arm target-interface
study and the separate paired anchor-versus-transport study.

The CPU analysis commands below reproduce the supplied numerical tables.
Fresh simulator runs require the external LeWM checkpoints and original
benchmark datasets. **Checkpoints, datasets, latent caches, and trajectory
images are intentionally not included.**

## Reproduce the reported tables

Run from this folder with Python 3.11 or 3.12:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements/analysis.txt
python -m pip install --no-deps -e .

python -m anchored_planning analyze --study main --output outputs/main --check-reference
python -m anchored_planning analyze --study local-target --output outputs/local_target --check-reference
python examples/reproduce_supporting_tables.py --output outputs/supporting
python -m unittest discover -s tests -v
```

`analyze` writes CSV tables and a JSON summary. `--check-reference` compares
every generated table field to the released tables and exits with an error
on a difference. Verified during packaging: **160 main-study rows and 46
local-target rows match**. The supporting script regenerates another 11 rows
for the earlier diagnostic, LeWM-controller, and search-budget studies.
It uses the original saved statistics and each study's separate protocol;
see `results/supporting/README.md` for the source mapping.
The analysis and action-domain tests run with NumPy only; the two native
kernel tests explicitly skip if PyTorch is unavailable.

The main input is `results/main/successes.json`: 512 query records, each with
a 3-condition by 5-arm binary success matrix, in the order defined by
`protocols/main.json`. It contains 7,680 outcomes. The local-target input is
`results/local_target/paired_queries.csv`: 512 query records and 3,072
outcomes. These are different cohorts and are analyzed separately.

The main analysis uses 10,000 episode-cluster bootstrap replicates, seed
20260913, paired within each task and independent across tasks. It averages
the two perturbation starts within each query, then weights the four tasks
equally. Local-target effects use exact fractions before display rounding.
`actual_work.csv` contains recorded work and timing aggregates; success-only
inputs cannot regenerate timings, so the analysis does not reconstruct them.

## What each method does

All methods retrieve support using standardized features
`[E(current), E(goal), E(goal) - E(current)]`, with equal feature-group
weights. Support is restricted to the frozen training episodes and complete
remaining-horizon continuations. The action block is five primitive steps;
retrieval returns eight candidate support rows.

| Study / arm | Action search | Scoring target |
| --- | --- | --- |
| Main: `direct` | Execute the first retrieved record action block | No dynamics ranking |
| Main: `ap_observed` | Rank the same eight legal record blocks | Observed successor of the first support row |
| Main: `ap_final` | Rank the same eight legal record blocks | Final goal observation |
| Main: `gaussian_observed` | Gaussian CEM from a fixed standardized-zero prior | Observed successor of the first support row |
| Main: `gaussian_final` | Same Gaussian CEM | Final goal observation |
| Local: `anchor` | Same Gaussian CEM; observation-only controller memory | `E(source + 5)` |
| Local: `transport` | Same Gaussian CEM; observation-only controller memory | `E(live) + E(source + 5) - E(source)` |

The Gaussian planner scores 300 samples for 30 iterations, keeps 30 elites,
then compares the prior and final mean with a strict improvement test. This
is 31 predictor batch calls and 9,002 candidate blocks per decision. It
projects candidates to the actual environment Box before scoring. The
recorded action values are used by main-study retrieval actions and by the
evaluator's fixed perturbation prefixes; Gaussian action initialization
does not use them. Physical reset states belong to the evaluator.

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
was named `official_weights_compat.pt`; the loader accepts the old and new
Transformers ViT key layouts and performs a strict 303-key load. It does not
train or replace weights. `protocols/asset_inventory.json` preserves the
recorded filenames; public repositories may change, and their download
revisions were not preserved in this package.

All datasets must expose `pixels` (uint8, `[rows,224,224,3]`), `action`,
`ep_len`, `ep_offset`, and the reset-state column above. Cube actions have
five coordinates; the other tasks have two. HDF5 compression requires
`hdf5plugin`. The included `protocols/scalers.json` contains the original
full-dataset float64 StandardScaler records, excluding non-finite action
rows. These are the fixed training/evaluation normalization artifacts;
they are not recomputed from a selected memory bank.

Edit `examples/assets.json` or copy it to your own JSON file. Each path is
resolved relative to that JSON file; absolute paths are also accepted.
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
`libegl1`, and `libglib2.0-0`; install these through your OS package manager.
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
encoding uses the recorded bfloat16 autocast path; native planning/target
encoding uses projected float32 features without autocast. Cache generation
over millions of images is a substantial external computation.

The example `--limit 1` is a small new run, not a result reproduction check.
Omit `--limit` for all 128 frozen queries of a task; repeat for all four tasks
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
GPU memory per worker was used in the original launcher; this is a recorded
configuration, not a newly measured minimum.

Collect the completed task/shard outcome files for a study in one folder:

```bash
python -m anchored_planning analyze --study main --outcomes outputs/new-main --output outputs/new-main-analysis
python -m anchored_planning analyze --study local-target --outcomes outputs/new-local --output outputs/new-local-analysis
```

The analyzer rejects incomplete cohorts and duplicate outcomes. Add
`--check-reference` only when you intend to compare the new result to the
released table values. GPU, rendering, or regenerated-cache differences
can change planning outcomes; the portable evaluator has not been rerun
over the complete cohort during packaging.

## Validation and layout

Packaging checks reproduced all seven success/effect CSV tables, checked
the complete cohort and held-out-episode assumptions, and tested action
projection. CPU native tests checked train-only retrieval and the bounded
Gaussian optimizer with its 9,002-block count. A separate read-only check
loaded all four existing external checkpoints strictly, encoded an actual
dataset image through both interfaces, and scored eight candidates through
the native predictor for each task. No GPU experiment or new simulator
rollout was performed during packaging. Clean installation and full native
cohort reproduction remain unverified.

- `anchored_planning/`: model, retrieval, scoring, optimizer, simulator,
  analysis, and command-line modules.
- `protocols/`: frozen cohorts, removed/training episodes, source indices,
  optimizer settings, and normalization records.
- `results/`: compact success records and released numerical tables.
- `examples/assets.json`: editable external-asset paths.
- `examples/reproduce_supporting_tables.py`: numerical reproduction of the earlier supporting-study snapshots.
- `requirements/`: CPU analysis and task-specific native requirements.
- `tests/`: aggregation, action-domain, retrieval, and optimizer checks.

The top-level `LICENSE` retains the upstream MIT notice. See `NOTICE.md`
for the included source extracts and release-specific changes.
