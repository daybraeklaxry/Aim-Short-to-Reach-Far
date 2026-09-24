"""First-state, same-eight-action effects; importing this module runs nothing.

The environment factory reconstructs a query and its fixed TRAIN prelude. It
returns the same six values as the confirmation construction function. No live
environment is reset or restored. Exact effects stop at physical success or an
environment stop; fixed-five-step prediction diagnostics require all eight
branches to complete five primitives. All six selectors remain defined using
the observed stopping endpoints, even when those diagnostics are unavailable.
"""

from copy import deepcopy


SELECTORS = (
    'direct', 'learned_local', 'learned_far', 'exact_local', 'exact_far',
    'recorded_delta_local',
)


def _numpy(np, value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _ranks(np, scores):
    """Ascending average ranks, with exact equal-score ties."""
    order = np.argsort(scores, kind='stable')
    ranks = np.empty(len(scores), dtype=np.float64)
    first = 0
    while first < len(order):
        last = first + 1
        while last < len(order) and scores[order[last]] == scores[order[first]]:
            last += 1
        ranks[order[first:last]] = (first + last - 1) / 2.0
        first = last
    return ranks


def _diagnostics(np, predicted_scores, exact_scores):
    """Same-target prediction fidelity; these are not task-success metrics."""
    predicted = np.asarray(predicted_scores, dtype=np.float64)
    exact = np.asarray(exact_scores, dtype=np.float64)
    predicted_rank, exact_rank = _ranks(np, predicted), _ranks(np, exact)
    p, e = predicted_rank - predicted_rank.mean(), exact_rank - exact_rank.mean()
    denom = float(np.linalg.norm(p) * np.linalg.norm(e))
    i, j = np.triu_indices(len(predicted), 1)
    predicted_margin, exact_margin = predicted[i] - predicted[j], exact[i] - exact[j]
    non_tied = exact_margin != 0
    signs_match = np.sign(predicted_margin) == np.sign(exact_margin)
    error = predicted - exact
    chosen, best = int(np.argmin(predicted)), int(np.argmin(exact))
    predicted_sorted, exact_sorted = np.sort(predicted), np.sort(exact)
    return dict(
        candidate_count=int(len(predicted)),
        predicted_selected_rank=chosen,
        exact_selected_rank=best,
        selected_exact_argmin=bool(exact[chosen] == exact[best]),
        spearman=None if denom == 0 else float(np.dot(p, e) / denom),
        true_non_tied_pair_count=int(non_tied.sum()),
        pairwise_sign_accuracy=None if not non_tied.any() else float(signs_match[non_tied].mean()),
        score_bias=float(error.mean()),
        score_rmse=float(np.sqrt(np.mean(error ** 2))),
        centered_score_rmse=float(np.sqrt(np.mean((error - error.mean()) ** 2))),
        pairwise_margin_rmse=float(np.sqrt(np.mean((predicted_margin - exact_margin) ** 2))),
        selected_exact_cost=float(exact[chosen]),
        selected_exact_regret=float(exact[chosen] - exact[best]),
        direct_exact_cost=float(exact[0]),
        selected_exact_gain_over_direct=float(exact[0] - exact[chosen]),
        predicted_gain_over_direct=float(predicted[0] - predicted[chosen]),
        gain_over_direct_error=float((predicted[0] - predicted[chosen]) - (exact[0] - exact[chosen])),
        predicted_best_runner_up_margin=float(predicted_sorted[1] - predicted_sorted[0]),
        exact_best_runner_up_margin=float(exact_sorted[1] - exact_sorted[0]),
    )


def summarize_effects(np, arrays):
    """Keep stopping-endpoint outcomes separate from fixed-horizon fidelity."""
    score_keys = dict(
        learned_local='learned_local_scores', learned_far='learned_far_scores',
        exact_local='exact_local_scores', exact_far='exact_far_scores',
        recorded_delta_local='recorded_delta_local_scores',
    )
    selected = dict(direct=0)
    selected.update({name: int(np.argmin(arrays[key])) for name, key in score_keys.items()})
    groups = {str(rank): [name for name in SELECTORS if selected[name] == rank]
              for rank in sorted(set(selected.values()))}
    for name, key in score_keys.items():
        arrays[f'{name}_ascending_ranks'] = _ranks(np, arrays[key])
    full = np.asarray(arrays['complete_five_steps'], dtype=bool)
    diagnostics = None
    if full.all():
        embedding_error = np.square(arrays['predicted_endpoints'] - arrays['exact_endpoints']).sum(-1)
        delta_error = np.square(arrays['recorded_delta_endpoints'] - arrays['exact_endpoints']).sum(-1)
        diagnostics = dict(
            learned_local=_diagnostics(np, arrays['learned_local_scores'], arrays['exact_local_scores']),
            learned_far=_diagnostics(np, arrays['learned_far_scores'], arrays['exact_far_scores']),
            recorded_delta_local=_diagnostics(np, arrays['recorded_delta_local_scores'], arrays['exact_local_scores']),
            learned_endpoint_squared_error=embedding_error.tolist(),
            recorded_delta_endpoint_squared_error=delta_error.tolist(),
            learned_endpoint_mean_squared_l2_error=float(embedding_error.mean()),
            recorded_delta_endpoint_mean_squared_l2_error=float(delta_error.mean()),
        )
    return dict(
        selected_ranks=selected,
        continuation_ranks=sorted(set(selected.values())),
        continuation_selector_groups=groups,
        complete_five_candidate_count=int(full.sum()),
        fixed_five_diagnostic_eligible=bool(full.all()),
        prediction_diagnostics=diagnostics,
        direct_transfer=dict(
            recorded_start_to_live_squared_l2=float(np.square(arrays['current_latent'] - arrays['recorded_start_latents'][0]).sum()),
            stopped_endpoint_local_squared_l2=float(arrays['exact_local_scores'][0]),
            stopped_endpoint_far_squared_l2=float(arrays['exact_far_scores'][0]),
            executed_primitives=int(arrays['executed_primitive_count'][0]),
            complete_five_steps=bool(full[0]),
        ),
        selected_stopped_endpoint_effects={name: dict(
            rank=rank, source=int(arrays['candidate_sources'][rank]),
            local_squared_l2=float(arrays['exact_local_scores'][rank]),
            far_squared_l2=float(arrays['exact_far_scores'][rank]),
            success_during_first_chunk=bool(arrays['success'][rank].any()),
            executed_primitives=int(arrays['executed_primitive_count'][rank]),
        ) for name, rank in selected.items()},
    )


def _assert_replay(ew, task, env, obs, info, goal, replay, replay_arrays,
                   expected, expected_arrays, extra_state_fn):
    np = ew.np
    assert ew.equal_state(ew.physical_state(task, env), expected['post_state']), 'exact branch physical-state mismatch'
    assert extra_state_fn(ew, task, env) == expected['post_extra'], 'exact branch clock/hidden-state mismatch'
    assert replay['prelude'] == expected['prelude'], 'exact branch prelude-outcome mismatch'
    np.testing.assert_array_equal(goal, expected['true_goal'], err_msg='exact branch goal mismatch')
    for key in ('post_rgb', 'goal_rgb', 'initial_sources', 'initial_raw', 'initial_normalized'):
        np.testing.assert_array_equal(replay_arrays[key], expected_arrays[key], err_msg=f'exact branch {key} mismatch')
    np.testing.assert_array_equal(np.ascontiguousarray(env.render()), expected_arrays['post_rgb'],
                                  err_msg='exact branch live RGB mismatch')
    assert not bool(ew.runtime._success_now(task, obs, info, goal, False, False)), 'exact branch is already successful'


def simulate_eight(ew, task, env_factory, construction, cached, extra_state_fn, encode_rgb):
    """Reconstruct every branch independently and stop on every primitive."""
    np = ew.np
    raw = np.asarray(cached['initial_raw'], dtype=np.float32)
    assert raw.shape[:2] == (8, 5)
    endpoints, branches = [], []
    counts = np.zeros(8, dtype=np.int64)
    success, terminated, truncated, executed = (np.zeros((8, 5), dtype=bool) for _ in range(4))
    for rank in range(8):
        env, obs, info, goal, replay, replay_arrays = env_factory()
        try:
            _assert_replay(ew, task, env, obs, info, goal, replay, replay_arrays,
                           construction, cached, extra_state_fn)
            for t, action in enumerate(raw[rank]):
                obs, _, term, trunc, info = env.step(action.copy())
                won = bool(ew.runtime._success_now(task, obs, info, goal, False, False))
                counts[rank] += 1
                executed[rank, t] = True
                success[rank, t], terminated[rank, t], truncated[rank, t] = won, bool(term), bool(trunc)
                if won or term or trunc:
                    break
            endpoints.append(_numpy(np, encode_rgb(np.ascontiguousarray(env.render()))).reshape(-1))
            won = bool(success[rank].any())
            terminal = bool(terminated[rank].any() or truncated[rank].any())
            branches.append(dict(
                rank=rank, reconstructed_start_verified=True,
                executed_primitive_count=int(counts[rank]),
                complete_five_steps=bool(counts[rank] == 5), success=won,
                terminal_without_success=bool(terminal and not won),
                stopped=bool(won or terminal),
                stop_reason='success' if won else 'terminated' if terminated[rank].any()
                            else 'truncated' if truncated[rank].any() else 'five_steps',
                endpoint_state=ew.physical_state(task, env),
                endpoint_extra=extra_state_fn(ew, task, env),
            ))
        finally:
            env.close()
    return branches, dict(
        exact_endpoints=np.asarray(endpoints, dtype=np.float32),
        executed_primitive_count=counts, complete_five_steps=counts == 5,
        executed=executed, success=success, terminated=terminated, truncated=truncated,
    )


def measure_first_state(ew, native, task, row, construction, cached, handle,
                        env_factory, extra_state_fn, encoding_cache=None):
    """Return JSON metadata and NPZ arrays for one preregistered query/condition.

    cached supplies post_rgb, goal_rgb, initial_sources/raw/normalized. The
    factory returns (env, obs, info, true_goal, construction, cached). The
    optional cache contains ONLY this task's native.encode_rgb embeddings by
    dataset index; it must not be shared across different tasks.
    The caller saves these returned records beneath the new study directory.
    """
    np, torch = ew.np, ew.torch
    result = dict(
        schema='same_live_eight_exact_effect_v1', task=task,
        identity={key: row[key] for key in ('record_id', 'episode', 'global_start', 'global_goal', 'environment_seed', 'horizon')},
        prelude=deepcopy(construction['prelude']), entered_policy=construction['prelude']['category'] == 'policy_entered',
        scoring_space='native.encode_rgb: original frozen LeWM projected encoder, FP32',
        effect_semantics='At most five retrieved raw primitives; stop at first physical success, terminated or truncated.',
        diagnostic_semantics='Prediction/score/ranking diagnostics only when all eight candidates execute five primitives.',
        continuation_semantics='Deduplicate selected first ranks; execute that first chunk, then common live direct to the original total policy budget.',
        tie_rule='lowest retrieval rank among exactly equal scores', additional_training=False,
    )
    if not result['entered_policy']:
        result.update(selected_ranks={name: None for name in SELECTORS}, continuation_ranks=[],
                      continuation_selector_groups={}, branches=[], fixed_five_diagnostic_eligible=False,
                      complete_five_candidate_count=0, prediction_diagnostics=None)
        return result, dict(candidate_sources=np.empty(0, dtype=np.int64))

    sources = np.asarray(cached['initial_sources'], dtype=np.int64)
    raw = np.asarray(cached['initial_raw'], dtype=np.float32)
    normalized = np.asarray(cached['initial_normalized'], dtype=np.float32)
    assert sources.shape == (8,) and raw.shape == normalized.shape == (8, 5, native.action_dim)
    bank = native.context['bank']
    assert native.task == task
    assert np.all(bank.train_mask[bank.episode_idx[sources]])
    np.testing.assert_array_equal(bank.episode_idx[sources], bank.episode_idx[sources + 5])
    np.testing.assert_array_equal(raw, ew.runtime._raw_actions(native.context, sources, 5).astype(np.float32))
    np.testing.assert_array_equal(normalized, native.scaler.transform(raw.reshape(-1, native.action_dim)).astype(np.float32).reshape(raw.shape))
    cache = {} if encoding_cache is None else encoding_cache
    recorded_encode_calls = 0

    def recorded(index):
        nonlocal recorded_encode_calls
        index = int(index)
        if index not in cache:
            cache[index] = _numpy(np, native.encode_rgb(np.ascontiguousarray(handle['pixels'][index]))).reshape(-1).copy()
            recorded_encode_calls += 1
        return cache[index]

    with torch.inference_mode():
        rgb = np.ascontiguousarray(cached['post_rgb'])
        current = native.encode_rgb(rgb)
        far = native.encode_rgb(np.ascontiguousarray(cached['goal_rgb']))
        starts = np.asarray([recorded(index) for index in sources], dtype=np.float32)
        ends = np.asarray([recorded(index + 5) for index in sources], dtype=np.float32)
        local = torch.as_tensor(ends[0:1], device=native.device)
        candidates = torch.as_tensor(normalized.reshape(1, 8, 1, -1), device=native.device)
        score_info = dict(
            pixels=native.pixels(rgb)[:, None].expand(-1, 8, -1, -1, -1, -1),
            current_latent=current[:, None, None, :].expand(-1, 8, -1, -1),
            subgoal_latent=local[:, None, :].expand(-1, 8, -1),
        )
        local_cost = native.latent_cost.get_cost(score_info, candidates)[0]
        predicted = score_info['predicted_emb'][0, :, -1, :]
        far_cost = ((predicted - far) ** 2).sum(-1)
        branches, arrays = simulate_eight(ew, task, env_factory, construction, cached, extra_state_fn, native.encode_rgb)
    arrays.update(
        candidate_sources=sources.copy(), candidate_raw_actions=raw.copy(), candidate_normalized_actions=normalized.copy(),
        current_latent=_numpy(np, current).reshape(-1), local_target=ends[0].copy(), far_target=_numpy(np, far).reshape(-1),
        recorded_start_latents=starts, recorded_end_latents=ends,
        predicted_endpoints=_numpy(np, predicted), learned_local_scores=_numpy(np, local_cost), learned_far_scores=_numpy(np, far_cost),
    )
    arrays['recorded_delta_endpoints'] = arrays['current_latent'][None] + (ends - starts)
    arrays['recorded_delta_local_scores'] = np.square(arrays['recorded_delta_endpoints'] - arrays['local_target']).sum(-1)
    arrays['exact_local_scores'] = np.square(arrays['exact_endpoints'] - arrays['local_target']).sum(-1)
    arrays['exact_far_scores'] = np.square(arrays['exact_endpoints'] - arrays['far_target']).sum(-1)
    np.testing.assert_allclose(arrays['learned_local_scores'], np.square(arrays['predicted_endpoints'] - arrays['local_target']).sum(-1), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(arrays['learned_far_scores'], np.square(arrays['predicted_endpoints'] - arrays['far_target']).sum(-1), rtol=2e-5, atol=2e-5)
    assert all(np.isfinite(value).all() for value in arrays.values()), 'nonfinite exact-effect measurement'
    result.update(
        branches=branches, local_target_recorded_index=int(sources[0]) + 5,
        source_train_verified=True, source_five_step_episode_continuity_verified=True,
        reconstructed_start_state=deepcopy(construction['post_state']),
        reconstructed_start_extra=deepcopy(construction['post_extra']),
        learned_f1_batch_calls=1, learned_f1_candidate_transitions=8,
        current_encoder_calls=1, far_goal_encoder_calls=1, recorded_encoder_calls=recorded_encode_calls,
        exact_endpoint_encoder_calls=8, exact_physical_primitives=int(arrays['executed_primitive_count'].sum()),
    )
    result.update(summarize_effects(np, arrays))
    return result, arrays


def validate_first_chunk_artifacts(ew, result, arrays, rank, actions, success,
                                   terminated, truncated, endpoint_state, endpoint_extra):
    """Validate a fresh or saved first chunk, including its saved hidden state."""
    np = ew.np
    expected = result['branches'][rank]
    count = expected['executed_primitive_count']
    np.testing.assert_array_equal(np.asarray(actions), arrays['candidate_raw_actions'][rank, :count])
    for key, actual in (('success', success), ('terminated', terminated), ('truncated', truncated)):
        np.testing.assert_array_equal(np.asarray(actual, dtype=bool), arrays[key][rank, :count])
    assert ew.equal_state(endpoint_state, expected['endpoint_state']), 'forced first chunk physical endpoint mismatch'
    assert endpoint_extra == expected['endpoint_extra'], 'forced first chunk hidden-state/clock mismatch'


def validate_continuation_first_chunk(ew, task, env, result, arrays, rank,
                                     actions, success, terminated, truncated, extra_state_fn):
    """Call immediately after the forced chunk, before any common continuation."""
    validate_first_chunk_artifacts(
        ew, result, arrays, rank, actions, success, terminated, truncated,
        ew.physical_state(task, env), extra_state_fn(ew, task, env),
    )


def validate_saved_first_chunk(ew, task, rollout_result, rollout_arrays, probe_result, probe_arrays):
    """Check a cached full rollout's first-chunk artifacts against its probe."""
    assert rollout_result['task'] == probe_result['task'] == task
    assert rollout_result['identity'] == probe_result['identity']
    if not rollout_result['decisions']:
        assert not probe_result['entered_policy']
        return
    first = rollout_result['decisions'][0]
    count, rank = int(first['executed_primitive_count']), int(first['selected_rank'])
    ew.np.testing.assert_array_equal(rollout_arrays['candidate_sources'][0], probe_arrays['candidate_sources'])
    ew.np.testing.assert_array_equal(rollout_arrays['candidate_raw_actions'][0], probe_arrays['candidate_raw_actions'])
    validate_first_chunk_artifacts(
        ew, probe_result, probe_arrays, rank, rollout_arrays['actions'][:count],
        rollout_arrays['success'][:count], rollout_arrays['terminated'][:count], rollout_arrays['truncated'][:count],
        rollout_result['physical_trajectory'][count], first['first_chunk_extra'],
    )
