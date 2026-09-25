# Recorded measurements

These files accompany the paper. Main-query measurements use the [LeWM planner](https://github.com/lucas-maes/le-wm). The tables in the PDF present standard starts and the query-wise perturbation average. These CSV files also give both individual prefixes, all measured settings, and sample counts.

The root command `python tools/reproduce_table1.py` recomputes Table 1 from
`main_episodes.csv`. The other per-episode files are `intervention_episodes.csv`,
`first_block_episodes.csv`, and `diagnostic_episode_metrics.csv`. Episode identity
is the task, query ordinal, and condition (`clean`, `prefix_a`, or `prefix_b`).
The main query protocol in `code/ap/protocols/main.json` supplies source indices.

## Files

| File | Contents |
|---|---|
| `main_results.csv` | All main controllers, tasks, and four start summaries, with success and mean episode prediction blocks. |
| `interventions.csv` | Every search budget, goal offset, target/memory/key ablation, and transported-target setting, including success, steps, and prediction blocks. `config_*` columns identify the setting. |
| `behavior.csv` | Physical stalling and detours at every measured threshold and start, including numerators and eligible denominators. Saved latent-detour measurements are retained here. |
| `first_block.csv` | All eight first-block selectors, with a common Direct continuation, at each task and start. |
| `diagnostics.csv` | Candidate endpoint error, action-selection regret, and eligible start/query counts, including separate prefixes. |
| `target_models.csv` | All sixteen trained target-model configurations, selected checkpoint steps, validation MSE, and training time. |
| `target_quality.csv` | Held-out successor-error and nearest-memory-distance means and medians for learned and observed targets. |
| `independent_replication.csv` | Final-goal versus observed-target comparisons on independent supporting queries, with CEM at standard starts and ranking at all starts. |

## Units and aggregation

- `success_percent` is measured on all assigned queries, including outcomes resolved before policy entry. Standard and each individual prefix use 128 queries per task. `Perturbed` averages the two outcomes within each query, then queries. It does not add a third independent test set.
- `mean_pred_blocks` counts actual predicted five-action blocks per episode, counted until each episode ends. It is not wall-clock time or total computation. `mean_steps` counts executed primitive actions when recorded.
- We compute stalling rates over failed rollouts with at least one policy decision and detour rates over successful rollouts that entered policy control. For perturbed starts, we pool eligible rollouts from both prefixes. Each metric has `_numerator`, `_denominator`, and `_percent` fields. An empty percentage means there were no eligible rollouts.
- `stall_W5`, `stall_W10`, and `stall_W20` use the corresponding trailing decision windows. `physical_detour_0.25`, `_0.5`, and `_1.0` use increases in physical error normalized by success thresholds. `latent_detour` uses the previously recorded relative latent-distance threshold. Direct has no logged latent-scoring measurement.
- Diagnostic distances and regrets use task-specific squared latent-distance units. Each endpoint error averages eight candidates. Perturbed diagnostics average eligible starts within each query, then weight queries equally. Eligible-start and eligible-query counts are separate columns.
- Target-model validation MSE averages latent coordinates. Target-quality errors sum squared latent-coordinate differences. Training worker minutes exclude feature preparation and reflect concurrent training processes.
- Numeric results preserve saved precision. Supporting-set success counts are uniquely recoverable from the published one-decimal rates with 128 episodes, as recorded in the `precision` column.

## Controller names and protocols

`cem_final`, `cem_learned`, and `cem_observed` use five-action CEM with final,
learned, and observed targets. `cem_transport` uses the transported recorded
displacement. `rank_final`, `rank_learned`, and `rank_observed` score the same
eight retrieved action blocks. `direct` executes the closest record's block.

The LeWM planner uses `WorldModelPolicy` and `CEMSolver`, planning five blocks
and executing up to five blocks with five primitive actions per block. Actions
are de-normalized by the pretrained action normalizer and sent to the simulator.
The other main-study controllers clip proposals to the action bounds before
scoring and execution. All controllers share queries, goals, action allowances,
checkpoints, and physical success criteria.

LeWM behavior measurements cover the original goal offsets. The independent
queries reported in Appendix F use de-normalized actions without additional
clipping and are recorded in `independent_replication.csv`.

## Reproduction sources

Use `code/ap/` for the controller and `code/expanded/` for learned targets and
interventions. The LeWM entry point is `code/lewm/baseline/official_baseline.py`.
Model checkpoints, simulator assets, and observation caches are external
dependencies. See [the reproduction guide](../REPRODUCING.md) for setup.
