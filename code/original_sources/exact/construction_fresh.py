"""Fresh shared construction with legal fixed TRAIN-derived perturbations.

Derived from the frozen confirmation construction.py. Original prefix data are
checked unchanged; only the effective actions are projected to the actual Box.
All five arms reconstruct the same projected prefix, without resampling.
"""
import json
from pathlib import Path
from action_domain import project, counts


def extra_state(ew, task, env):
    np, base = ew.np, env.unwrapped
    counters = []
    wrapper = env
    while True:
        counters.append(dict(type=type(wrapper).__name__, elapsed=wrapper.__dict__.get('_elapsed_steps')))
        if wrapper is base:
            break
        wrapper = wrapper.env
    extra = dict(wrapper_counters=counters)
    if task == 'cube':
        pose = base.__dict__.get('_target_effector_pose')
        extra.update(target_effector_pose_present='_target_effector_pose' in base.__dict__,
                     target_effector_pose=None if pose is None else pose.wxyz_xyz.tolist(),
                     reset_next_step=bool(base._reset_next_step))
    elif task == 'pusht':
        extra.update(block_velocity=list(base.block.velocity), block_angular_velocity=float(base.block.angular_velocity),
                     agent_velocity=list(base.agent.velocity))
    elif task == 'reacher':
        dm = base.env._env
        extra.update(physics_time=float(dm.physics.time()), dm_step_count=int(dm._step_count),
                     dm_reset_next_step=bool(dm._reset_next_step))
    else:
        extra.update(agent_position=base.agent_position.tolist(), target_position=base.target_position.tolist(),
                     wall_axis=int(base.wall_axis), wall_thickness=int(base.wall_thickness),
                     door_positions=base.door_positions.tolist(), door_sizes=base.door_sizes.tolist())
    return extra


def physical_values(ew, task, env):
    np, base = ew.np, env.unwrapped
    if task == 'cube':
        return dict(tcp_xyz=base.data.site_xpos[base._pinch_site_id].tolist(),
                    cube_xyz=base.data.joint('object_joint_0').qpos[:3].tolist(),
                    gripper_opening=float(np.clip(base.data.qpos[base._gripper_opening_joint_id] / 0.8, 0, 1)))
    if task == 'pusht':
        return dict(agent_xy=list(base.agent.position), block_xy=list(base.block.position), block_angle=float(base.block.angle))
    if task == 'reacher':
        return dict(qpos=base.env.physics.data.qpos.copy().tolist())
    return dict(agent_xy=base.agent_position.tolist())


def displacement(np, task, before, after):
    norm = lambda key: float(np.linalg.norm(np.asarray(after[key]) - np.asarray(before[key])))
    wrapped = lambda key: (np.asarray(after[key]) - np.asarray(before[key]) + np.pi) % (2 * np.pi) - np.pi
    if task == 'cube':
        return dict(tcp_xyz_metres=norm('tcp_xyz'), cube_xyz_metres=norm('cube_xyz'),
                    gripper_opening_change=float(after['gripper_opening'] - before['gripper_opening']))
    if task == 'pusht':
        return dict(agent_xy_pixels=norm('agent_xy'), block_xy_pixels=norm('block_xy'),
                    block_angle_radians=float(wrapped('block_angle')))
    if task == 'reacher':
        return dict(qpos_radians=float(np.linalg.norm(wrapped('qpos'))), qpos_wrapped_components=wrapped('qpos').tolist())
    return dict(agent_xy_pixels=norm('agent_xy'))


def apply_prelude(ew, task, env, obs, info, true_goal, raw):
    """Never reinterpret termination as success, step after a stop, or edit clocks."""
    initial = bool(ew.runtime._success_now(task, obs, info, true_goal, False, False))
    success, stopped, actions, flags, terms, truncs = initial, False, [], [], [], []
    for action in raw:
        if success or stopped:
            break
        obs, _, terminated, truncated, info = env.step(action.copy())
        success = bool(ew.runtime._success_now(task, obs, info, true_goal, False, False))
        stopped = bool((terminated or truncated) and not success)
        actions.append(action.copy())
        flags.append(success)
        terms.append(bool(terminated))
        truncs.append(bool(truncated))
    category = 'initial_success' if initial else 'prelude_success' if success else 'prelude_terminal_failure' if stopped else 'policy_entered'
    return obs, info, dict(initial_success=initial, success=success, terminal_without_success=stopped,
        category=category, primitive_steps=len(actions)), dict(prelude_actions=actions, prelude_success=flags,
        prelude_terminated=terms, prelude_truncated=truncs)


def construct(ew, api, context, scaler, task, row, stage, root, protocol, source):
    np = ew.np
    metadata, cached = api.shared_cache(ew, context, task, row, stage, root, allow_create=True)
    env, obs, info, true_goal = ew.runtime._make_live(context, row)
    try:
        assert ew.equal_state(ew.physical_state(task, env), metadata['initial'])
        np.testing.assert_array_equal(env.action_space.low, metadata['action_low'])
        np.testing.assert_array_equal(env.action_space.high, metadata['action_high'])
        assert bool(ew.runtime._success_now(task, obs, info, true_goal, False, False)) == metadata['initial_success']
        if task == 'cube':
            assert not env.unwrapped._visualize_info
        initial_extra = extra_state(ew, task, env)
        before = physical_values(ew, task, env)
        raw = np.asarray(source['raw_actions'], np.float32).reshape(-1, env.action_space.shape[0])
        if source['condition'] == 'clean':
            assert len(raw) == 0
        else:
            np.testing.assert_array_equal(raw, ew.runtime._raw_actions(context, np.asarray([source['source']]), 5)[0])
        effective, prefix_projection = project(raw, env.action_space.low, env.action_space.high)
        np.testing.assert_array_equal(effective, np.clip(raw, env.action_space.low, env.action_space.high).astype(np.float32))
        obs, info, prelude, records = apply_prelude(ew, task, env, obs, info, true_goal, effective)
        after = physical_values(ew, task, env)
        state, extra = ew.physical_state(task, env), extra_state(ew, task, env)
        rgb = np.ascontiguousarray(env.render())
        result = dict(identity=source['identity'], source=source, prelude=prelude,
            initial_extra=initial_extra, post_state=state, post_extra=extra, true_goal=np.asarray(true_goal).tolist(),
            physical_before=before, physical_after=after, displacement=displacement(np, task, before, after),
            frozen_utc=protocol['frozen_utc'],
            shared_action_domain='new_shared_action_domain',
            action_low=np.asarray(env.action_space.low).tolist(), action_high=np.asarray(env.action_space.high).tolist(),
            prefix_projection=dict(proposed=prefix_projection,
                executed=counts(raw[:prelude['primitive_steps']], effective[:prelude['primitive_steps']])),
            prefix_semantics='P of the same preassigned TRAIN five-action chunk; no resampling; outside policy budget; clocks continue')
        arrays = {key: np.asarray(values, np.float32 if key == 'prelude_actions' else bool) for key, values in records.items()}
        arrays['prelude_actions'] = arrays['prelude_actions'].reshape(-1, raw.shape[1])
        arrays.update(original_prefix_raw_actions=raw.copy(), effective_prefix_raw_actions=effective.copy())
        arrays.update(post_rgb=rgb, original_initial_rgb=cached['initial_rgb'], goal_rgb=cached['goal_rgb'], goal=cached['goal'])
        stem = root / 'construction' / task / f"{stage}_{source['condition']}_{row['record_id']}"
        canonical = api.load_pair(stem, np)
        if canonical is None:
            if prelude['category'] == 'policy_entered':
                current = ew.runtime._encode(context, rgb[None]).astype(np.float32)
                found, _, costs = context['bank'].nearest_topk(current, cached['goal'].reshape(1, -1), row['horizon'], 8)
                sources = np.asarray(found[0], np.int64)
                candidates = ew.runtime._raw_actions(context, sources, 5).astype(np.float32)
                normalized = scaler.transform(candidates.reshape(-1, raw.shape[1])).astype(np.float32).reshape(candidates.shape)
                arrays.update(current=current, initial_sources=sources, initial_raw=candidates, initial_normalized=normalized,
                              initial_retrieval_cost=np.asarray(costs[0]))
                old_current = cached['current'][0]
                original, _, original_cost = context['bank'].nearest_topk(cached['current'], cached['goal'].reshape(1, -1), row['horizon'], 8)
                old_source = int(original[0, 0])
                bank_latents = context['bank'].latents
                result['mismatch'] = dict(perturbed_raw_latent_gap_squared=float(np.square(current[0] - bank_latents[sources[0]]).sum()),
                    perturbed_top1_pair_retrieval_cost=float(np.asarray(costs[0])[0]),
                    original_raw_latent_gap_squared=float(np.square(old_current - bank_latents[old_source]).sum()),
                    original_top1_pair_retrieval_cost=float(np.asarray(original_cost[0])[0]),
                    perturbed_top1_source=int(sources[0]), original_top1_source=old_source)
            else:
                result['mismatch'] = None
            api.save_arrays(stem.with_suffix('.npz'), np, **arrays)
            stem.with_suffix('.json').parent.mkdir(parents=True, exist_ok=True)
            with stem.with_suffix('.json').open('x') as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
            return env, obs, info, true_goal, result, arrays
        expected, expected_arrays = canonical
        for key in result:
            assert result[key] == expected[key], f'construction replay mismatch: {task}/{row["record_id"]}/{key}'
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], expected_arrays[key], err_msg=f'construction replay: {key}')
        return env, obs, info, true_goal, expected, expected_arrays
    except BaseException:
        env.close()
        raise


def prepare(ew, api, context, scaler, task, row, stage, root, protocol, source):
    stem = root / 'construction' / task / f"{stage}_{source['condition']}_{row['record_id']}"
    if api.load_pair(stem, ew.np) is None:
        env, *_ = construct(ew, api, context, scaler, task, row, stage, root, protocol, source)
        env.close()
