# Aim Short to Reach Far: Goal Distance Underestimates Your Frozen World Model

**Xvyuan Liu, Jianjie Fang, Chen Gao, Yong Li — Tsinghua University**

Code and recorded results for **Anchored Planning**. The method retrieves an
observed target and uses a frozen world model to synthesize or rank actions toward
it. The paper compares final, learned, and observed targets while separating
target choice, search, and live-state prediction.

- [Paper](paper.pdf)
- [Project page](https://aim-short-to-reach-far.mkhfsvnloyeafgk9875.chatgpt.site)
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
| `docs/` | Static project page, paper figures, and a recorded PushT replay. |
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

## Attribution and licenses

The Anchored Planning package's license and third-party notices are in
[`code/ap/LICENSE`](code/ap/LICENSE) and [`code/ap/NOTICE.md`](code/ap/NOTICE.md).
Preserved upstream source files retain their attribution and license notices.
External datasets, simulators, and checkpoints remain subject to their original
terms. This repository does not redistribute those external assets.
