"""Fixed-budget extension on the original main-study queries and constructions."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
import common_rollout as runtime
import construction_fresh as construction
import observed_target
from memory_mask import apply as apply_memory_mask
from target_scorers import TargetGaussian

BASE = Path('runtime/ap/fresh_target_interface_study_v2')

def main(a):
    start = time.perf_counter()
    p = runtime.read(BASE / 'protocol.json')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, _, _, native_protocol, api, ew, context, native, _ = runtime.load_runtime(a.task, 'confirm')
    assert native_protocol['frozen_utc'] == p['native_frozen_utc']
    membership = apply_memory_mask(context, p, a.task)
    rows = p['studies'][a.task]['confirm']
    sources = {(s['identity']['record_id'], s['condition']): s for s in runtime.read(BASE / 'source_manifest.json')['rows'] if s['task'] == a.task}
    args = SimpleNamespace(task=a.task, stage='confirm', reference_root=BASE)
    counter = runtime.module('extension_counter', runtime.REFERENCE).WorkCounter(native.model)
    output = a.output / a.task
    output.mkdir(parents=True, exist_ok=True)
    record_file = output / f'outcomes_{a.mode}_shard{a.shard_index}of{a.shards}.jsonl'
    assert not record_file.exists(), record_file
    settings = [(30, kind) for kind in ('final', 'observed')] if a.mode == 'replay' else [(i, kind) for i in (1, 2, 5, 10) for kind in ('final', 'observed')]
    assignments = [(o, c) for o in range(128) if o % a.shards == a.shard_index for c in p['conditions']]
    if a.mode == 'replay':
        assignments = [(0, 'clean'), (0, 'prefix_a'), (1, 'clean')]
    metadata = dict(task=a.task, mode=a.mode, settings=settings, assignments=len(assignments), membership=membership, base_study=str(BASE), source_code=str(Path(__file__).resolve().parent), visible_gpu=os.environ['CUDA_VISIBLE_DEVICES'], scientific_changes='CEM iterations only; original target rules, queries, starts, memory, seed, and raw-action projection', started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    runtime.write(output / f'run_{a.mode}_shard{a.shard_index}of{a.shards}.json', metadata)
    completed = 0
    replay_decisions = 0
    try:
        with ew.h5py.File(ew.v6._data_path(context), 'r') as handle, torch.inference_mode(), record_file.open('x') as records:
            scorers = {kind: TargetGaussian(native, handle, kind) for kind in ('observed', 'final')}
            for ordinal, condition in assignments:
                row = rows[ordinal]
                source = sources[(row['record_id'], condition)]
                for iterations, kind in settings:
                    observed_target.ITERATIONS = iterations
                    scorer = scorers[kind]
                    scorer.iterations = iterations
                    scorer.set_goal_record(row['global_goal'])
                    arm = 'gaussian_' + kind
                    trace, arrays = runtime.rollout(args, p, api, ew, context, native, construction, scorer, counter, row, source, arm, ordinal, None)
                    if a.mode == 'replay':
                        old_stem = BASE / 'evaluation/traces' / a.task / f"confirm_{row['record_id']}_{condition}_{arm}"
                        old = runtime.read(old_stem.with_suffix('.json'))
                        for key in ['success', 'primitive_steps', 'decisions', 'predictor_calls', 'predictor_candidate_blocks', 'prelude']:
                            assert trace['compact'][key] == old['compact'][key], (a.task, row['record_id'], kind, key)
                        with np.load(old_stem.with_suffix('.npz')) as previous:
                            for key in ['actions', 'success', 'score_current', 'score_target', 'retrieved_sources']:
                                if key in previous:
                                    np.testing.assert_array_equal(arrays[key], previous[key], err_msg=f'{a.task}/{ordinal}/{condition}/{arm}/{key}')
                        replay_decisions += len(trace['decisions'])
                    trace['compact'].update(extension='unified_budget_20260923', iterations=iterations)
                    stem = output / 'traces' / f"{row['record_id']}_{condition}_{kind}_i{iterations}"
                    stem.parent.mkdir(parents=True, exist_ok=True)
                    runtime.write(stem.with_suffix('.json'), trace)
                    np.savez_compressed(stem.with_suffix('.npz'), **arrays)
                    records.write(json.dumps(trace['compact'], allow_nan=False) + '\n')
                    records.flush()
                    completed += 1
                    print(json.dumps(dict(task=a.task, mode=a.mode, completed=completed, expected=len(assignments)*len(settings), elapsed=round(time.perf_counter()-start, 1))), flush=True)
    finally:
        counter.handle.remove()
    assert completed == len(assignments) * len(settings)
    if a.mode == 'replay':
        assert replay_decisions > 0
    runtime.write(output / f'summary_{a.mode}_shard{a.shard_index}of{a.shards}.json', dict(completed=True, outcomes=completed, replay_exact=a.mode=='replay', replay_decisions=replay_decisions, elapsed_seconds=time.perf_counter()-start))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=('cube', 'pusht', 'reacher', 'tworoom'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('replay', 'budget'), required=True)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    main(p.parse_args())
