# LeWM released evaluation and the paper protocol

Status: all released-evaluator and paired-interface runs are complete (four tasks, three seeds, 50 queries per seed), and all 12 pairs pass the declared criterion. All paired initial images, checkpoint tensors and action-scaler statistics match. All 18 protocol-intervention/rescoring rows and all 1,536 five-action replanning episodes are complete. The variant has 128 queries for each task and each of the three fixed start conditions.

## Reference implementation

The reference is the [initial LeWM release](https://github.com/lucas-maes/le-wm/tree/83f97d72ad067855bc89a1b74b4aff11d4dfdf0c), using `stable-worldmodel` 0.0.6. Its unchanged `eval.py`, model definitions and evaluation YAML files are retained with the audit. This release uses `AutoCostModel` and `evaluate_from_dataset`; later repository revisions require different package APIs and were not mixed into this evaluation.

The released checkpoint tensors are the same tensors used by the main study. A serialized model object supplies the format expected by `AutoCostModel`; all 303 tensors were compared directly after serialization. No weights were trained or changed. The JEPA definition, predictor and attention methods, and image preprocessor match the released inference code. `model_source_check.json` records these comparisons and differences confined to training regularization, normalizer packaging, and checkpoint-saving utilities.

The released YAML specifies seed 42. Seeds 43 and 44 are additional fixed audit seeds, not three separately documented official defaults. All three are retained. The LeWM paper describes 10 CEM iterations outside PushT, whereas the released YAML specifies 30 for every task. The code-default evaluation uses 30 without tuning to the reported scores.

## Protocol comparison

| Component | Released evaluation | Main study and its H=25 comparison |
|---|---|---|
| Time units | Primitive environment actions; environment frame skip is one | Primitive environment actions |
| Model and planning blocks | Five actions per model block; five blocks per plan | The same released LeWM configuration |
| Execution | Execute 25 actions before replanning | Execute up to 25, with per-episode stopping on success, failure or allowance |
| Configured short-distance goal offset | 25 | 25 |
| Actual selected goal frame | Source +24: an exclusive-end, 25-frame slice is loaded and its last frame is used | Source +25, explicitly specified by the query manifest |
| Action allowance | 50 for every task | At H=25, Cube uses 25 and the other tasks use 50 |
| Queries | Valid rows sampled from the full supplied dataset, then sorted by global index | One start from each of 128 preassigned reserved episodes |
| Datasets | `pusht_expert_train`, `ogbench/cube_single_expert`, `dmc/reacher_random`, `tworoom` | The same source datasets; the main query split is training for PushT and validation for the other tasks |
| Goal image | The selected dataset frame | The selected dataset frame; the index rule above differs |
| First scoring image | The source dataset frame | A rendering of the restored live source state |
| Later scoring images | Live environment renderings | Live environment renderings |
| Cube initialization | Restore the recorded full `qpos` and `qvel` | Reconstruct from the public 28-dimensional observation, with separate initialization of unobserved joints and velocities |
| CEM | 300 candidates, 30 iterations, 30 elites, standardized initial deviation one; released warm start and history updates | The same released policy and solver |
| Action mapping | Inverse pretrained `StandardScaler`, without an additional raw-action Box projection | The same mapping for LeWM; the single-block target comparisons use their own shared Box projection |
| Images | RGB; `ToImage`, float32 scaling, ImageNet normalization and antialiased bilinear resize to 224 | The same transform; transformed images are moved to the GPU before expanding the candidate dimension |
| Optimizer randomness | A seeded CEM stream shared across vectorized queries | Fixed query-specific CEM seeds in the main study; the parity check shares the reference stream and ordering |
| Success aggregation | First success reached during 50 executed steps, aggregated by logical OR | Check the initial state and every primitive step; stop at the first success |
| Rendering | EGL | OSMesa in the main runner; the reference and its paired wrapper check both use EGL |

The reference's reported environment-seed field is null for these datasets. Its configured audit seed controls query selection and the CEM generator. Therefore the reported parity is the predeclared agreement in episode outcomes, not a claim of bitwise equality of every later physical state or rendered image.

The 128 Cube source reconstructions match recorded arm joint positions exactly and the object's pose to floating-point precision. In 122 queries, the remaining gripper coordinates or velocities differ from full recorded simulator state. All main controllers receive the same reconstruction. The separate `cube_full_state` intervention measures this difference rather than assuming its effect.

## Physical success predicates

| Task | Released physical success rule |
|---|---|
| Cube | The selected object's position is within 0.04 m of its target; neither orientation nor gripper position enters this test. |
| PushT | The joint Euclidean error of agent and block positions is below 20 pixels, and the shortest block-angle error is below pi/9. |
| Reacher | Each arm-joint position error is below 0.05 radians. |
| TwoRoom | The position error is below 16 pixels. |

`success_rescoring.csv` applies both implementations to the existing H=25 main trajectories. Across 512 episodes and 15,265 saved states, the predicates agree at every checked state and reproduce every recorded episode outcome. Success counts remain 41, 78, 88 and 99 in task order. Initial successes number 25, 0, 0 and 8. This check measures predicate equivalence on the saved main trajectories; the different initial-success aggregation and query protocols remain as specified above.

## Parity and attribution

The paired wrapper evaluation calls the main runner's actual `policy_for`, `raw_info` and `released_block` functions. The released evaluator supplies the source episodes and goal images, and the per-query policies consume the reference CEM random stream in the same order. Checkpoint tensors and action-scaler statistics are also compared directly. The predeclared gate is a success-count difference of at most three out of 50 and episode-outcome agreement of at least 90% for each seed. All 12 pairs pass: 11 have identical episode outcomes, and PushT seed 44 differs on one of 50 outcomes (43 successes for the released evaluator, 42 for the paper interface). Minimum per-seed outcome agreement is 98%. Published point estimates and remeasured rates are reported separately; this agreement concerns implementations on matched queries.

`official_repro.csv` records all query identities and reported scores; `parity.csv` records outcome agreement and action differences. No failed launch is counted as an episode outcome. NVIDIA EGL aborted during rendering for Cube and one Reacher seed, including a Cube random-policy diagnostic. A complete 50-query, 50-step Cube diagnostic passed with Mesa software EGL. Failed evaluations are retried under that EGL driver with the same seeds and scientific configuration; each reference/wrapper pair uses the same driver. Per-run metadata records the choice. The released Python source, checkpoint, query sampling, physics engine, and CEM settings are unchanged. See `runtime_check.md` for the runtime diagnosis.

All 18 rows in `gap_attribution.csv` are complete. Each changes one setting from the main H=25 protocol or rescoring rule. Query sampling has the largest measured effect on Cube and PushT: it changes Cube success from 41/128 to 92/128 and PushT from 78/128 to 106/128. Cube's larger action allowance alone gives 47/128, full-state restoration gives 42/128, and the dataset source image leaves its count unchanged. On Reacher, using the dataset source image changes success from 88/128 to 102/128; changing the goal frame to source +24 gives 98/128. These effects are not an additive decomposition of the full protocol gap. Budget changes are needed only for Cube because the other main H=25 allowances already equal 50. The goal remains a dataset image in both protocols.

## Five-action replanning variant

`LeWM (5-action replanning)` changes only the released policy's receding horizon from five model blocks to one. It still plans five blocks, uses the same CEM parameters, and operates on the main queries, starts, goals and allowances. Each decision evaluates 45,000 predicted five-action blocks. Episode work includes early stopping and is reported in `lewm_replan5.csv` and `lewm_replan5_episodes.csv`. Fixed profile episodes are reused in the final result set rather than sampled again. No AP, learned-target, ranking or Direct result is rerun by this audit.
