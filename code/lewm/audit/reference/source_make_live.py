def _make_live(context: dict[str, Any], row: dict[str, Any]) -> tuple[Any, Any, dict[str, Any], np.ndarray]:
    if context["kind"] != "pure_pusht":
        return _exp4_make_live(context, row)
    arrays = context["arrays"]
    start = np.asarray(arrays["states"][int(row["global_start"])], dtype=np.float64)
    goal = np.asarray(arrays["states"][int(row["global_goal"])], dtype=np.float64)
    env = PushT(resolution=224, render_mode="rgb_array", relative=True)
    env.reset(seed=int(row["environment_seed"]))
    env._set_state(start)
    env._set_goal_state(goal)
    return env, env._get_obs(), env._get_info(), goal
