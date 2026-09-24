# Recorded measurements

These files accompany the paper. Main-query measurements use the released LeWM planner. The tables in the PDF present standard starts and the query-wise perturbation average; these CSV files also give both individual prefixes, all measured settings, and sample counts.

The root command `python tools/reproduce_table1.py` recomputes Table 1 from
`main_episodes.csv`. The other per-episode files are `intervention_episodes.csv`,
`first_block_episodes.csv`, and `diagnostic_episode_metrics.csv`. Episode identity
is the task, query ordinal, and condition (`clean`, `prefix_a`, or `prefix_b`);
the main query protocol in `code/ap/protocols/main.json` supplies source indices.

## Files

| File | Contents |
|---|---|
| `main_results.csv` | All main controllers, tasks, and four start summaries; success and mean episode prediction blocks. |
| `interventions.csv` | Every search budget, goal offset, target/memory/key ablation, and transported-target setting, including success, steps, and prediction blocks. `config_*` columns identify the setting. |
| `behavior.csv` | Physical stalling and detours at every measured threshold and start, including numerators and eligible denominators. Saved latent-detour measurements are retained here. |
| `first_block.csv` | All eight first-block selectors, with a common Direct continuation, at each task and start. |
| `diagnostics.csv` | Candidate endpoint error, action-selection regret, and eligible start/query counts, including separate prefixes. |
| `target_models.csv` | All sixteen trained target-model configurations, selected checkpoint steps, validation MSE, and training time. |
| `target_quality.csv` | Held-out successor-error and nearest-memory-distance means and medians for learned and observed targets. |
| `independent_replication.csv` | Final-goal versus observed-target comparisons on independent supporting queries; CEM at standard starts and ranking at all starts. |
| `legacy_tables.csv` | Original cells of earlier additional-query and supporting-study tables removed from the PDF. The source table and row/column identify every cell, including headers. This is an archival record, separate from current main-query estimates. |

## Units and aggregation

- `success_percent` is measured on all assigned queries, including outcomes resolved before policy entry. Standard and each individual prefix use 128 queries per task. `Perturbed` averages the two outcomes within each query, then queries. It does not add a third independent test set.
- `mean_pred_blocks` counts actual predicted five-action blocks per episode, including early stopping. It is not wall-clock time or total computation. `mean_steps` counts executed primitive actions when recorded.
- Behavioral rates use eligible rollouts: stalling conditions on failures with decisions; detours condition on successes that entered control. Behavioral `Perturbed` pools eligible rollouts from the two prefixes. Each metric has `_numerator`, `_denominator`, and `_percent` fields; an empty percentage means no eligible rollouts, not zero percent.
- `stall_W5`, `stall_W10`, and `stall_W20` use the corresponding trailing decision windows. `physical_detour_0.25`, `_0.5`, and `_1.0` use increases in physical error normalized by success thresholds. `latent_detour` uses the previously recorded relative latent-distance threshold. Direct has no logged latent-scoring measurement.
- Diagnostic distances and regrets use task-specific squared latent-distance units. Each endpoint error averages eight candidates. Perturbed diagnostics average eligible starts within each query, then queries equally; eligible-start and eligible-query counts are separate columns.
- Target-model validation MSE averages latent coordinates. Target-quality errors sum squared latent-coordinate differences. Training worker minutes exclude feature preparation and reflect concurrent training processes.
- Numeric results preserve saved precision. Supporting-set success counts are uniquely recoverable from the published one-decimal rates with 128 episodes, as recorded in the `precision` column. `legacy_tables.csv` preserves original displayed cells without inventing extra precision.

## Controller names and protocols

`cem_final`, `cem_learned`, and `cem_observed` are five-action CEM with final, learned, and observed targets. `cem_transport` uses the transported recorded displacement. `rank_final`, `rank_learned`, and `rank_observed` score the same eight retrieved blocks; `direct` executes the leading record. Some source summaries use their full display names.

LeWM results in the main and intervention files use the released `WorldModelPolicy` / `CEMSolver` recipe: plan five blocks and execute up to five blocks, with five primitive actions per block. They retain the released inverse-scaler environment interface. Other main-study controls use raw-action Box projection. The same query states, goals, allowances, checkpoints, and physical success criteria are shared. Corrected LeWM physical-behavior measurements are available at the original offsets; retired-recipe behavior at other offsets is not presented as current data.

The independent supporting queries use de-normalized actions without an external Box projection. The earlier additional-query anchoring study, archived in `legacy_tables.csv`, used Box projection and a separate memory exclusion set. These are distinct studies, not extra rows of the main benchmark. All comparisons in the PDF appendix use current main-query measurements except its final independent-replication section.

## Reproduction sources

The accompanying repository contains the controller source, target-model/ablation experiment source, and corrected LeWM runner. Use the current CSV files for the paper's numerical results. The corrected LeWM entry point is `code/lewm/audit/official_baseline.py`; the older multiblock runner in the target-experiment source is not the paper's LeWM baseline. Model checkpoints, simulator assets, and observation caches are upstream dependencies and are not duplicated in this source archive.
