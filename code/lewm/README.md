# Released LeWM baseline

The primary runner is `baseline/official_baseline.py`. It uses the released
`WorldModelPolicy` and `CEMSolver`, planning and executing up to 25 primitive
actions. Main-query per-episode measurements are in `../../data/main_episodes.csv`.
all measured goal offsets are in `../../lewm_checks/released_planner_episodes.csv`.
The protocol-check sources and results are in `protocol_check/` and
`../../lewm_checks/`. See the root README for external asset requirements and
the repository-relative runtime paths.
