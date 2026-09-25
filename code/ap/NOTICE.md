# Source and result provenance

The upstream LeWM model source is available at
https://github.com/lucas-maes/le-wm. `jepa.py` and `module.py` preserve the
source copied into the experiment runtime. The included MIT license and
copyright notice belong to Lucas Maes (2026) and must remain with those
sources. Release integration code is distributed under the same MIT terms.
No additional author identity or unpublished repository URL is asserted.

The runtime's recorded normalization reference identifies upstream commit
`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`, `eval.py:66-79`. The runtime
checkout itself did not expose Git metadata during packaging, so that
reference is not a claim that all experiment-specific sources came from
that upstream commit. The model/data source is the public LeWM collection:
https://huggingface.co/collections/quentinll/lewm.

The external simulator and official CEM implementation are supplied by
`stable-worldmodel` 0.0.6 (https://github.com/galilai-group/stable-worldmodel).
They remain external dependencies with their own licenses. Checkpoints,
datasets, native binaries, and trajectory images are not redistributed.

Included experiment sources:

| Release source | Origin and packaging change |
| --- | --- |
| `jepa.py`, `module.py` | LeWM runtime source, copied unchanged |
| `model.py` | Targeted model/preprocessing/strict checkpoint-loader functions extracted from the experiment runtime. Local imports and an explicit inference entry point |
| `environments.py` | Recorded reset, task success, and state snapshot functions extracted from the experiment runtime |
| `native_adapter.py` | Original native float32 scoring and official CEM adapter |
| `observation_bank.py` | Original observation-only bank, feature normalization, exact top-k search, and train/episode filtering |
| `gaussian_cem.py` | Original five-step bounded Gaussian optimizer and seed function |
| `evaluation_helpers.py`, `lifecycle.py`, `action_domain.py` | Original evaluator-prefix, renderer-cleanup, and Box-projection helpers |
| `planners.py` | Main `TargetRank`/`TargetGaussian` and local `ObservationGaussian` assembled behind explicit arguments. Same numerical scoring and target expressions |
| `statistics.py` | Original main-study bootstrap and numerical table estimators. New I/O is in `analysis.py` |
| `analysis.py`, `runtime.py`, `__main__.py` | Release integration: relative asset paths, bounded HDF5/cache I/O, CLI, result export, and focused checks |

`protocols/main.json` and `protocols/local_target.json` retain the completed
experiments' exact selected query identities, training-memory masks,
removed episodes, numerical settings, and fixed prefix source rows. Local
machine paths and historical worker/approval records were omitted. The
model/dataset/cache basenames are retained in `asset_inventory.json`.

The current paper's per-episode measurements are supplied in the repository's
`data/` directory. The root reproduction guide describes their protocols and
aggregation.

The portable runner reconstructs the evaluator directly from the public
dataset row indices instead of requiring the original machine's private
fixture paths. It is an inspectable reproduction entry point, not a claim
of a newly verified complete experiment. The README lists the checks that
actually ran and the remaining reproduction limits.
