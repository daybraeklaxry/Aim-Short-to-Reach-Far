"""Recorded-action-prior CEM on the existing development/confirmation states."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

from observed_target import query_seed

BASE = Path('external/le-wm')
CONFIRMATION = BASE / 'research/ap_rank_confirmatory_20260912/confirmation'
REFERENCE = BASE / 'research/ap_rank_capability_20260912/released_cem_reference_20260913/frozen_code/released_cem_reference.py'
ORIGINAL_CEM = BASE / 'research/ap_rank_capability_20260912/target_only_cem_multitask_20260913/frozen_code/target_only_cem_multitask.py'
STUDY = 'record_neighborhood_reference_20260914'
TASKS = ('cube', 'pusht', 'reacher', 'tworoom')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def load_runtime(task, stage):
    main_protocol = read(CONFIRMATION / 'protocol.json')
    root = Path(main_protocol['development_evidence'][task]['development_protocol']).parent if stage == 'profile' else CONFIRMATION
    protocol = read(root / 'protocol.json')
    native_root = Path(protocol['native_root'])
    native_protocol = read(native_root / 'protocol.json')
    assert native_protocol['frozen_utc'] == protocol['native_frozen_utc']
    sys.path.insert(0, str(native_root / 'frozen_code'))
    runner = module('calibrated_native_runtime', native_root / 'frozen_code/run_eval.py')
    api, ew = runner.load_shared(native_root, native_protocol)
    construction = module('calibrated_construction', root / 'frozen_code/construction.py')
    if task == 'cube':
        lifecycle = module('calibrated_cube_lifecycle', root / 'frozen_code/lifecycle.py')
        lifecycle.install_cube_renderer_cleanup(ew)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    horizon = 150 if task == 'pusht' else protocol['studies'][task]['horizon']
    context, _ = ew.runtime._context(task, horizon, 32, 50, device)
    context['model'].eval().requires_grad_(False)
    from native_adapter import NativeAdapter
    native = NativeAdapter(ew, context, task, native_protocol, native_root)
    native.target_mode = 'recorded'
    return root, main_protocol, protocol, native_protocol, api, ew, context, native, construction


@torch.inference_mode()
def rollout(args, protocol, api, ew, context, native, construction, scorer, counter, row, source, arm, ordinal, original_cem):
    study = protocol['studies'][args.task]
    env, obs, info, true_goal, setup, cached = construction.construct(
        ew, api, context, native.scaler, args.task, row, args.stage, args.reference_root, protocol, source)
    scorer.set_action_bounds(env.action_space.low, env.action_space.high)
    seed = query_seed(args.task, ordinal)
    generator = torch.Generator(device=native.device).manual_seed(seed)
    neutral_verification = None
    if args.stage == 'profile' and source['condition'] == 'clean':
        neutral_verification = scorer.verify_source_consistency(
            cached['initial_sources'][0], cached['initial_raw'][0], cached['initial_normalized'][0], seed)
    started = time.perf_counter()
    success = bool(setup['prelude']['success'])
    stopped = bool(setup['prelude']['terminal_without_success'])
    actions, flags, terms, truncs, decisions = [], [], [], [], []
    trajectory = [setup['post_state']]
    arrays = {}
    work_calls = work_blocks = 0
    try:
        while not success and not stopped and len(actions) < study['budget']:
            step = len(actions)
            rgb = np.ascontiguousarray(cached['post_rgb'] if step == 0 else env.render())
            current = cached['current'].copy().reshape(1, -1) if step == 0 else \
                ew.runtime._encode(context, rgb[None]).astype(np.float32)
            goal = cached['goal'].reshape(1, -1)
            remaining = max(5, int(row['horizon']) - step)
            sources, _, retrieval_cost = context['bank'].nearest_topk(current, goal, remaining, 8)
            sources = np.asarray(sources[0], np.int64)
            bank = context['bank']
            assert sources.shape == (8,) and np.all(bank.train_mask[bank.episode_idx[sources]])
            assert np.array_equal(bank.episode_idx[sources], bank.episode_idx[sources + remaining])
            raw = ew.runtime._raw_actions(context, sources, 5).astype(np.float32)
            normalized = native.scaler.transform(raw.reshape(-1, native.action_dim)).astype(np.float32).reshape(8, 5, native.action_dim)
            if step == 0:
                np.testing.assert_array_equal(sources, cached['initial_sources'])
                np.testing.assert_array_equal(raw, cached['initial_raw'])
                np.testing.assert_array_equal(normalized, cached['initial_normalized'])
            counter.reset()
            tick = time.perf_counter()
            proposed, detail, scoring = scorer.choose(rgb, sources, raw, normalized, generator)
            torch.cuda.synchronize(native.device)
            assert detail['predictor_batch_calls'] == scorer.iterations + 1
            assert detail['predictor_candidate_transitions'] == 300 * scorer.iterations + 2
            assert counter.calls == detail['predictor_total_calls']
            assert counter.candidates == detail['predictor_total_candidate_blocks']
            work_calls += counter.calls
            work_blocks += counter.candidates
            decision = dict(t=step, remaining=remaining, retrieved_sources=sources.tolist(),
                retrieval_cost=np.asarray(retrieval_cost[0]).tolist(),
                predictor_calls=counter.calls, predictor_candidate_blocks=counter.candidates,
                planning_seconds=time.perf_counter()-tick, **detail)
            for key, value in dict(retrieved_sources=sources, retrieved_raw_actions=raw,
                                   retrieved_normalized_actions=normalized, **scoring).items():
                arrays.setdefault(key, []).append(np.asarray(value).copy())
            for action in proposed:
                obs, _, terminated, truncated, info = env.step(action.copy())
                success = bool(ew.runtime._success_now(args.task, obs, info, true_goal, False, False))
                stopped = bool((terminated or truncated) and not success)
                actions.append(action.copy())
                flags.append(success)
                terms.append(bool(terminated))
                truncs.append(bool(truncated))
                trajectory.append(ew.physical_state(args.task, env))
                if success or stopped or len(actions) >= study['budget']:
                    break
            decision['executed_actions'] = len(actions) - step
            decisions.append(decision)
        elapsed = time.perf_counter() - started
        action_array = np.asarray(actions, np.float32).reshape(-1, native.action_dim)
        arrays.update(goal_encoding=np.asarray(cached['goal'], np.float32), actions=action_array, success=np.asarray(flags, bool),
                      terminated=np.asarray(terms, bool), truncated=np.asarray(truncs, bool))
        arrays = {key: np.asarray(value) for key, value in arrays.items()}
        compact = dict(study=STUDY, task=args.task, stage=args.stage, identity=row,
            condition=source['condition'], arm=arm, completed=True, success=success,
            primitive_steps=len(actions), decisions=len(decisions), budget=study['budget'],
            horizon=row['horizon'], prelude=setup['prelude'], predictor_calls=work_calls,
            predictor_candidate_blocks=work_blocks,
            source_prediction_misses=sum(d['source_prediction_misses'] for d in decisions),
            elapsed_seconds=elapsed, optimizer_seed=seed, query_ordinal=ordinal)
        trace = dict(compact=compact, source=source, construction=setup, decisions=decisions,
                     trajectory=trajectory, final_state=ew.physical_state(args.task, env),
                     source_consistency_verification=neutral_verification,
                     true_goal=np.asarray(true_goal).tolist())
        return trace, arrays
    finally:
        env.close()

