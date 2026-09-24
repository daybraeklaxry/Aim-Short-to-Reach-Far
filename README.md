# Aim Short to Reach Far: Goal Distance Underestimates Your Frozen World Model

**Xvyuan Liu, Jianjie Fang, Chen Gao, Yong Li — Tsinghua University**

![Anchored Planning overview](docs/assets/figure-1.svg)

Code and recorded results for **Anchored Planning**. The method retrieves an
observed target and uses a frozen world model to synthesize or rank actions toward
it. The paper compares final, learned, and observed targets while separating
target choice, search, and live-state prediction.

- [Paper](paper.pdf)
- [Project page](https://daybraeklaxry.github.io/Aim-Short-to-Reach-Far/)
- [Per-episode main results](data/main_episodes.csv)
- [Data definitions and units](data/README.md)
- [LeWM protocol checks](lewm_checks/README.md)

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

Standard uses the recorded start. Perturbed averages the two assigned prefixes
within each query; the mean row weights tasks equally. Display rounding happens
after aggregation. This command reproduces the table from saved measurements;
it does not run new simulator episodes.

The LeWM rows use the released `WorldModelPolicy` / `CEMSolver`, planning and
executing up to 25 primitive actions. The other predictive controllers use
five-action blocks and the action mapping described in the paper. All rows use
the same main queries, goals, and task action allowances.

## Code and data layout

| Path | Contents |
| --- | --- |
| `data/` | Current paper measurements and per-episode outcomes. |
| `tools/reproduce_table1.py` | Standalone reconstruction of Table 1. |
| `docs/` | Static project page, paper figures, and twelve paired demos: four AP-rank and eight AP-CEM. |
| `data/project_demos/` | Demo query identities and the saved physical states for both controllers. |
| `tools/render_project_demos.py` | Render the paired demos from those saved states. |
| `tools/render_rank_demos.py` | Render the AP-rank and Direct comparisons. |
| `code/ap/` | Anchored Planning controller, asset configuration, and main query protocol. |
| `code/expanded/` | Learned-target training, target quality, budget/offset/memory interventions, and exact-endpoint diagnostics. |
| `code/original_sources/` | Frozen runtime modules imported by the experiment scripts. |
| `code/lewm/audit/official_baseline.py` | Released LeWM baseline on the main queries. |
| `code/lewm/protocol_check/` | Released-evaluator comparison and five-action replanning variant. |
| `lewm_checks/` | Query identities, per-seed parity, single-factor checks, and recorded LeWM outcomes. |
| `reproduction_inputs/` | Target-model training records, target-quality samples, and runtime protocol metadata. |

`data/main_episodes.csv` and `data/intervention_episodes.csv` include the released
LeWM outcomes used by the paper. `data/first_block_episodes.csv` and
`data/diagnostic_episode_metrics.csv` give the per-query diagnostic records.
`data/legacy_tables.csv` and the local result snapshots under `code/ap/results/`
document separate earlier supporting studies; they are not inputs to Table 1.

## Replay the project-page demonstrations

The gallery has two views. AP-rank shows two perturbed-start examples each on
PushT and Cube, comparing predictive action selection with Direct from the same
start and goal. We take the first two queries in query order where AP-rank
succeeds, Direct fails, and AP-rank executes more than ten actions under
`prefix_a`, the first preassigned five-action perturbation.

AP-CEM shows two standard-start examples per task, comparing AP-CEM with
final-goal CEM from the same start and goal. For each task, we take the first two
queries in query order where AP-CEM succeeds, final-goal CEM fails, and AP-CEM
executes more than ten actions. Both views show selected success examples; the
full success rates are reported separately in Table 1 and on the project page.

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
include physical states only; no model weights or original image datasets are
needed for this replay.

To render the AP-rank comparisons, use the same environment:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa SDL_VIDEODRIVER=dummy \
python tools/render_rank_demos.py \
  --trace-root data/project_demos/traces \
  --cases data/project_demos/rank_cases.json \
  --output outputs/rank-demos --task pusht
```

Use `--task cube` for the Cube comparisons. The renderer uses the same saved-state
restoration as the AP-CEM demos, without running a controller or advancing physics.

## Running simulator experiments

Fresh runs require external LeWM weights, original HDF5 benchmark data, encoded
observation caches, and the task simulators. These large upstream assets are not
included. Start with [the asset and environment instructions](code/ap/README.md)
and `code/ap/examples/assets.json`. Upstream model/data links and the task-specific
reset-state fields are listed there.

The original five-arm controller has a configurable evaluation entry point:

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
- `code/lewm/audit/official_baseline.py`: the paper's released LeWM baseline.

Machine-specific source paths have been replaced with repository-relative
`runtime/ap/` and `external/` paths. Run these scripts from the repository root
after supplying the external assets and arranging the protocol locations in
`reproduction_inputs/`. The original directory hierarchy below these roots is
preserved so that the source and recorded configurations remain identifiable.
`code/expanded/run_long_v2.py` is an older dependency; use
`official_baseline.py` for the paper's LeWM baseline.

## Citation

```bibtex
@misc{liu2026aimshort,
  title  = {Aim Short to Reach Far: Goal Distance Underestimates Your Frozen World Model},
  author = {Xvyuan Liu and Jianjie Fang and Chen Gao and Yong Li},
  year   = {2026},
  url    = {https://github.com/daybraeklaxry/Aim-Short-to-Reach-Far}
}
```

## License

Original code in this repository is released under the [MIT License](LICENSE).
Parts derived from LeWM retain their original MIT license and notices in
[`code/ap/LICENSE`](code/ap/LICENSE) and [`code/ap/NOTICE.md`](code/ap/NOTICE.md).
External datasets, simulators, and checkpoints remain subject to their original
terms.

### Third-party attribution

The Anchored Planning package's license and third-party notices are in
[`code/ap/LICENSE`](code/ap/LICENSE) and [`code/ap/NOTICE.md`](code/ap/NOTICE.md).
Preserved upstream source files retain their attribution and license notices.
External datasets, simulators, and checkpoints remain subject to their original
terms. This repository does not redistribute those external assets.
