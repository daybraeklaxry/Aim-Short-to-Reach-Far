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
