"""Fresh Direct/raw AP rollout with matched valid policy actions.

Derived from frozen oral-revision run_calibrated.py formal direct/local rollout.
Only policy action projection/evidence are added; profile replay is excluded.
Scorer, retrieval, observed target, argmin/ties and termination remain original.
"""
import time
import numpy as np
import torch
from action_domain import counts, prepare_candidates

STUDY = 'fresh_projected_baseline_adapter'
BASELINE_SOURCE = 'external/le-wm/research/ap_rank_oral_revision_20260913/frozen_code/run_calibrated.py'


@torch.inference_mode()
def rollout(args, protocol, api, ew, context, native, construction, scorer, counter, row, source, arm):
    assert args.stage == 'confirm' and arm in ('direct', 'ap_observed', 'ap_final')
    study = protocol['studies'][args.task]
    env, obs, info, true_goal, setup, cached = construction.construct(
        ew, api, context, native.scaler, args.task, row, args.stage, args.reference_root, protocol, source)
    low, high = env.action_space.low.copy(), env.action_space.high.copy()
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
            original_normalized = normalized
            effective, normalized, projected = prepare_candidates(raw, low, high, native.scaler)
            counter.reset()
            tick = time.perf_counter()
            rank, detail, scoring = scorer.choose(arm, rgb, sources, normalized)
            torch.cuda.synchronize(native.device)
            assert counter.calls == (0 if arm == 'direct' else 1)
            assert counter.candidates == (0 if arm == 'direct' else 8)
            work_calls += counter.calls
            work_blocks += counter.candidates
            decision = dict(t=step, remaining=remaining, selected_rank=rank,
                selected_source=int(sources[rank]), candidate_sources=sources.tolist(),
                retrieval_cost=np.asarray(retrieval_cost[0]).tolist(),
                predictor_calls=counter.calls, predictor_candidate_blocks=counter.candidates,
                planning_seconds=time.perf_counter()-tick,
                action_projection=dict(candidates=projected, selected=counts(raw[rank], effective[rank])), **detail)
            for key, value in dict(candidate_sources=sources, candidate_raw_actions=raw,
                                   candidate_original_normalized_actions=original_normalized,
                                   candidate_effective_raw_actions=effective,
                                   candidate_normalized_actions=normalized, **scoring).items():
                arrays.setdefault(key, []).append(np.asarray(value).copy())
            for action in effective[rank]:
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
            n = decision['executed_actions']
            decision['action_projection']['executed'] = counts(raw[rank, :n], effective[rank, :n])
            decisions.append(decision)
        elapsed = time.perf_counter() - started
        action_array = np.asarray(actions, np.float32).reshape(-1, native.action_dim)
        arrays.update(actions=action_array, success=np.asarray(flags, bool),
                      terminated=np.asarray(terms, bool), truncated=np.asarray(truncs, bool))
        arrays = {key: np.asarray(value) for key, value in arrays.items()}
        compact = dict(study=STUDY, task=args.task, stage=args.stage, identity=row,
            condition=source['condition'], arm=arm, completed=True, success=success,
            primitive_steps=len(actions), decisions=len(decisions), budget=study['budget'],
            horizon=row['horizon'], prelude=setup['prelude'], predictor_calls=work_calls,
            predictor_candidate_blocks=work_blocks,
            source_prediction_misses=sum(d['source_prediction_misses'] for d in decisions),
            elapsed_seconds=elapsed, baseline_replay_match=None,
            shared_action_domain='new_shared_action_domain',
            action_projection_totals={scope: {unit: sum(d['action_projection'][scope][unit] for d in decisions)
                for unit in ('chunks','action_steps','coordinates')} for scope in ('candidates','selected','executed')})
        trace = dict(compact=compact, source=source, construction=setup, decisions=decisions,
                     action_domain=dict(name='new_shared_action_domain', bounds_source='env.action_space',
                         action_low=np.asarray(low).tolist(), action_high=np.asarray(high).tolist(),
                         candidate_raw_actions='original retrieved record; used for initial cache equality',
                         candidate_effective_raw_actions='P(record); normalized for scoring and executed without a second transform'),
                     trajectory=trajectory, final_state=ew.physical_state(args.task, env),
                     true_goal=np.asarray(true_goal).tolist())
        return trace, arrays
    finally:
        env.close()

