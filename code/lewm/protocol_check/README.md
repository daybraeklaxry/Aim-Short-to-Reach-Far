# LeWM protocol-check sources

These programs collect the additional measurements in Appendix D.2. The [LeWM evaluator](https://github.com/lucas-maes/le-wm) snapshot is unchanged and includes its MIT license and public source
record. `../baseline/official_baseline.py` is the same planning implementation used
for the paper's main LeWM results.

## Required assets and environment

Use the external frozen checkpoints, HDF5 datasets, saved main queries and
simulator states described in the reproduction guide. Configure their paths in
the source runtime metadata. The runtime environment and Cube cache paths in
`run.sh` are explicit installation placeholders. The original Reacher compatibility
environment uses MuJoCo 3.10.0 and dm-control 1.0.43. See
`lewm_checks/runtime_versions.json` for task-specific package versions. The repository does not redistribute simulators or pretrained weights.

Only import locations and machine-specific paths were adapted for packaging.
No prediction, scaling, query-selection, action-selection or success logic was
changed. `fetch_runtime_wheels.py` documents the isolated additional dependencies.

## Evaluation order

1. `prepare_official_assets.py --task TASK` links the existing dataset and serializes
   the LeWM checkpoint object, checking equality of all state tensors.
2. `observe_official.py --task TASK --seed SEED --mode official` executes the
   unchanged LeWM `eval.py`. Repeat with `--mode wrapper` to run the main
   planning implementation on identical reference queries and the same CEM stream.
   Seeds are 42, 43, 44 and each run uses 50 queries. The observer records query
   identities, actions and outcomes. Completed output directories are not reused.
3. `summarize_checks.py` writes LeWM evaluation and parity CSV files. Complete
   all three task seeds before interpreting the task's follow-up experiments.
4. `run_gap_factor.py --task TASK --factor FACTOR --start 0 --stop 128` runs one
   changed main-H25 setting. Factors are `official_queries`, `goal_frame24`, and
   `dataset_initial_image`. Cube also has `budget50` and `cube_full_state`.
   `rescore_main.py` applies both physical predicates to saved main trajectories.
5. `run_replan5.py --task TASK --start 0 --stop 128` evaluates the separately named
   five-action replanning variant on all three main start conditions. The reported
   runs used disjoint query chunks. The first four standard-start profile episodes
   were reused once, not rerun or counted twice.
6. `summarize_followups.py` emits complete 128-query cells and individual outcomes.

Cube and Reacher seed 44 use Mesa software EGL. Both evaluators in each pair
use the same renderer. The per-run metadata records this setting alongside the
package versions. These programs evaluate LeWM and leave AP results unchanged.
