# LeWM protocol and implementation checks

These files accompany Appendix D.2. They evaluate the released LeWM baseline and its five-action replanning variant. They do not replace the main target-comparison results or rerun AP.

- `official_repro.csv` lists task, seed, success count, and the exact dataset episode and frame identities used by the released evaluator. Each row contains 50 queries. Seed 42 is specified in the released configuration. 43 and 44 are additional seeds. The configured 25-frame slice ends at source +24.
- `parity.csv` compares the released evaluator with the paper's planning implementation on the same query images and CEM random stream. Input, checkpoint, scaler, and action comparisons are also recorded. Outcome agreement does not imply bitwise-identical trajectories.
- `gap_attribution.csv` changes one setting at a time from the main H=25 evaluation. Query sampling changes the sampled sources. The other interventions retain them. Effects must not be added together as a decomposition of the total protocol difference.
- `success_rescoring.csv` applies both physical success predicates to the saved main trajectories, without new rollouts.
- `lewm_replan5.csv` reports complete task/start cells for the additional LeWM variant. It plans 25 actions and executes at most five before replanning. Its CEM parameters, main queries, goals, and action allowances are unchanged.
- `lewm_replan5_episodes.csv` contains the corresponding individual outcomes and prediction counts. `clean` denotes standard starts. `prefix_a` and `prefix_b` denote the two fixed perturbed starts. The paper averages the two perturbations with equal weight.
- `protocol_diff.md` explains the differences between the released and main evaluation protocols.
- `model_source_check.json` records equality of the inference code and checkpoint tensors, and identifies differences in training utilities. `runtime_versions.json` records package versions under each task's launcher.

Percentages use episode outcomes before display rounding. Prediction work counts predicted five-action blocks, counted until each episode ends. It is not wall-clock time. One decision of either LeWM configuration evaluates 45,000 blocks. Full source and goal images, pretrained weights, and simulator datasets remain external assets.

The comparison programs and released evaluator snapshot are provided under `code/lewm/protocol_check/`. The LeWM implementation is `code/lewm/baseline/official_baseline.py`.
