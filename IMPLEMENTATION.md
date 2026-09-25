# Implementation details

This document records the implementation settings and result-file locations for
the experiments reported in the paper. For commands, assets, and dependencies,
see [REPRODUCING.md](REPRODUCING.md).

## Query sampling and random seeds

The main queries use seed `26091391`. The two preassigned five-action perturbation
prefixes use seeds `26091392` and `26091393`. Query records specify the episode,
initial and goal indices, environment seed, and prefix. All controllers share
the same queries, goals, and prefix assignments.

The nested memory subsets use seed `26091600`. Each subset is selected once,
and its retrieval statistics are recomputed from its member episodes.

The LeWM evaluator comparison averages seeds `42`, `43`, and `44`, with
50 queries per seed. The LeWM configuration specifies seed `42`.

## Encodings and action normalization

Action scoring uses FP32, ImageNet channel normalization, 224-pixel images, and
192-dimensional projected latent endpoints. CUDA matrix-multiplication TF32
and cuDNN TF32 are disabled.

Live-query retrieval features are computed with CUDA bfloat16 autocast and
returned as float32 before comparison with the preserved frozen-cache features.
Current and target encodings for action scoring are computed separately in FP32.
The learned-target encodings and the anchoring/transport comparison also use
the frozen encoder in FP32.

LeWM's pretrained action `StandardScaler` keeps its original float64 mean,
scale, and affine transform. It is fitted to finite source-dataset action rows
after removing NaNs. Action clipping and execution follow the controller
settings described in Appendix A.

## Runtime and LeWM planner

The experiments use NVIDIA H20 GPUs. The main study uses Python `3.11.14`,
PyTorch `2.7.1`, CUDA `12.6`, and cuDNN `90501`.
Cube, PushT, and TwoRoom use MuJoCo `3.12.0`. Reacher uses MuJoCo `3.10.0`
with dm-control `1.0.43`.

The [LeWM planner](https://github.com/lucas-maes/le-wm) uses `WorldModelPolicy` and `CEMSolver` from
`stable-worldmodel` version `0.0.6`. On Cube, the LeWM evaluator restores
the full recorded `qpos` and `qvel`, while the main-study controllers start
from a state reconstructed from the public observation. See Appendix D.2
for the evaluation-protocol comparison.

## Result files

| Location | Contents |
| --- | --- |
| [`data/diagnostics.csv`](data/diagnostics.csv) | Endpoint-error and selection-regret summaries, including eligible-start and query counts. |
| [`data/first_block.csv`](data/first_block.csv) | First-block interventions and separate-prefix results. |
| [`data/target_quality.csv`](data/target_quality.csv) | Means and medians of successor errors and nearest-memory distances. |
| [`data/interventions.csv`](data/interventions.csv) | Results and prediction work across budget, goal-offset, and ablation settings. |
| [`lewm_checks/`](lewm_checks/) | Query identities, per-seed outcomes, implementation parity, and single-factor protocol comparisons. |
| [`lewm_checks/lewm_replan5.csv`](lewm_checks/lewm_replan5.csv) | Five-action LeWM replanning summaries. |
| [`lewm_checks/lewm_replan5_episodes.csv`](lewm_checks/lewm_replan5_episodes.csv) | Episode outcomes and prediction work for five-action LeWM replanning. |

The data directories also contain separate-start results and behavior-threshold
sweeps. Their layout is described in
[REPRODUCING.md](REPRODUCING.md#code-and-data-layout).
