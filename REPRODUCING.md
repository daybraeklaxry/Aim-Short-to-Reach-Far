# Reproducing the paper

[Project page](https://daybraeklaxry.github.io/Aim-Short-to-Reach-Far/) | [Repository overview](README.md) | [Paper](https://arxiv.org/abs/2609.30036)

The repository includes the measurements behind the paper, experiment code,
and saved simulator states for the demonstrations. Use the recorded results
to recompute the tables, or set up the external models and simulators to run
new evaluations.

- [Recompute Table 1](#reproduce-table-1)
- [Find code and data](#code-and-data-layout)
- [Implementation settings and result-file index](IMPLEMENTATION.md)
- [Render the demonstrations](#replay-the-project-page-demonstrations)
- [Run simulator experiments](#running-simulator-experiments)

## Reproduce Table 1

From the repository root, with Python 3.11 or newer:

```bash
python tools/reproduce_table1.py
```

This command uses only the Python standard library. It reads **12,288 recorded
outcomes**: four tasks, eight controllers, 128 paired queries, and three starts.
It recomputes every success rate and verifies all 128 task/controller/start cells
against `data/main_results.csv`. It writes `outputs/table1/table1.md`,
`table1.csv`, and `table1_unrounded.csv`.

For standard starts, we use the recorded state. For perturbed starts, we average
the two assigned prefixes within each query. The mean row weights tasks equally.
We round after aggregation. This command reads the recorded measurements and
does not run new simulator episodes.

The LeWM rows use the released `WorldModelPolicy` / `CEMSolver`, planning and
executing up to 25 primitive actions. The other predictive controllers use
five-action blocks and the action mapping described in the paper. All rows use
the same main queries, goals, and task action allowances.

## Code and data layout

| Path | Contents |
| --- | --- |
| `data/` | Current paper measurements and per-episode outcomes. |
| `tools/reproduce_table1.py` | Standalone reconstruction of Table 1. |
| `docs/` | Static project page, paper figures, and sixteen paired demos: eight AP-rank and eight AP-CEM. |
| `data/project_demos/` | Demo query identities and the saved physical states for both controllers. |
| `tools/render_project_demos.py` | Render the paired demos from those saved states. |
| `tools/render_rank_pairs.py` | Render final-goal ranking and AP-rank comparisons. |
| `code/ap/` | Anchored Planning controller, asset configuration, and main query protocol. |
| `code/expanded/` | Learned-target training, target quality, budget/offset/memory interventions, and exact-endpoint diagnostics. |
| `code/original_sources/` | Frozen runtime modules imported by the experiment scripts. |
| `code/lewm/baseline/official_baseline.py` | Released LeWM baseline on the main queries. |
| `code/lewm/protocol_check/` | Released-evaluator comparison and five-action replanning variant. |
| `lewm_checks/` | Query identities, per-seed parity, single-factor checks, and recorded LeWM outcomes. |
| `reproduction_inputs/` | Target-model training records, target-quality samples, and runtime protocol metadata. |

`data/main_episodes.csv` and `data/intervention_episodes.csv` include the released
LeWM outcomes used by the paper. `data/first_block_episodes.csv` and
`data/diagnostic_episode_metrics.csv` give the per-query diagnostic records.

## Replay the project-page demonstrations

The page shows sixteen paired examples: two per task for AP-CEM and two per task
for AP-rank, covering Cube, PushT, Reacher, and TwoRoom. The first example from
each task is visible in the main grids. **More examples** reveals the second set.

AP-CEM is compared with final-goal CEM at standard starts. AP-rank is compared
with final-goal ranking under `prefix_a`, the first assigned five-action
perturbation. Each pair shares its initial state and final goal. Ranking uses
the same eight retrieved candidates, and only the scoring target changes.

For each method and task, we select the first two queries in query order where
AP succeeds, the final-goal baseline fails, and AP executes more than ten actions.
These are selected examples, not estimates of overall performance. The page
reports the full evaluation separately and labels every video's recorded outcome.

The videos render saved simulator states at 20 actions per second and hold the
terminal state after a controller stops. Rendering does not call a world model
or execute new simulator actions. The PushT green overlay is set to the query's
goal pose.

With the task simulator dependencies described below, FFmpeg on `PATH`, and
DejaVu Sans fonts installed in the Linux rendering environment:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa SDL_VIDEODRIVER=dummy \
python tools/render_project_demos.py \
  --trace-root data/project_demos/traces \
  --cases data/project_demos/cases.json \
  --output outputs/demos --task pusht
```

Use `cube`, `reacher`, or `tworoom` to render another task. The script checks paired
starts, goals, outcomes, and state counts before rendering. It writes the videos,
goal images, posters, and a small provenance JSON for each case. The saved data
include physical states only. No model weights or original image datasets are
needed for this replay.

To render the AP-rank comparisons, use the same environment:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa SDL_VIDEODRIVER=dummy \
python tools/render_rank_pairs.py \
  --trace-root data/project_demos/traces \
  --cases data/project_demos/rank_cases.json \
  --output outputs/rank-demos --task pusht
```

Use `--task cube`, `--task reacher`, or `--task tworoom` for the other tasks. The renderer uses the same saved-state
restoration as the AP-CEM demos, without running a controller or advancing physics.

## Running simulator experiments

Fresh runs require external LeWM weights, original HDF5 benchmark data, encoded
observation caches, and the task simulators. These large upstream assets are not
included. Start with [the asset and environment instructions](code/ap/README.md)
and `code/ap/examples/assets.json`. Upstream model/data links and the task-specific
reset-state fields are listed there.

The five-arm controller has a configurable evaluation entry point:

```bash
python -m pip install -r code/ap/requirements/analysis.txt
python -m pip install --no-deps -e code/ap
python -m anchored_planning --help
```

Native evaluation additionally requires the pinned task dependencies and CUDA
PyTorch described in `code/ap/README.md`. The expanded experiment sources are
saved experiment scripts, not a packaged one-command benchmark. Their entry
points are:

- `code/expanded/train_grid.py`: fit and select learned target models.
- `code/expanded/run_suite.py`: target, budget, goal-offset, and memory settings.
- `code/expanded/target_quality.py`: held-out target accuracy measurements.
- `code/expanded/run_exact_v2.py`: exact-endpoint and first-block interventions.
- `code/lewm/baseline/official_baseline.py`: the paper's released LeWM baseline.

Machine-specific source paths have been replaced with repository-relative
`runtime/ap/` and `external/` paths. Run these scripts from the repository root
after supplying the external assets and arranging the protocol locations in
`reproduction_inputs/`. The original directory hierarchy below these roots is
preserved so that the source and recorded configurations remain identifiable.
`code/expanded/run_long_v2.py` supplies runtime components. Run
`code/lewm/baseline/official_baseline.py` to evaluate the LeWM baseline.
