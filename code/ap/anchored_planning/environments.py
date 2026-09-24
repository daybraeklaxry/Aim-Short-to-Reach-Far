"""Benchmark reset and success functions extracted without numeric changes."""
from __future__ import annotations
from typing import Any
import math
import numpy as np
import gymnasium as gym
import mujoco
import stable_worldmodel.envs
import stable_worldmodel.utils
from stable_worldmodel.envs.pusht.env import PushT
def _cube_state_to_sim(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct CubeEnv's simulator state from its public 28-D observation."""
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    if state.shape != (28,):
        raise ValueError(f"Cube observation must have shape (28,), got {state.shape}")
    qpos = np.zeros(21, dtype=np.float64)
    qvel = np.zeros(20, dtype=np.float64)
    qpos[:6] = state[:6]
    qvel[:6] = state[6:12]
    opening = float(np.clip(state[17] / 3.0, 0.0, 1.0))
    driver = 0.8 * opening
    # The two driver joints are tied by an equality constraint.  The passive
    # four-bar joints are initialized at their neutral pose; the observation
    # and the rendered scene still agree on all task-relevant quantities.
    qpos[6] = driver
    qpos[10] = driver
    qpos[14:17] = state[19:22] / 10.0 + np.asarray([0.425, 0.0, 0.0])
    qpos[17:21] = state[22:26]
    return qpos, qvel

def _make_env(task: str) -> gym.Env:
    if task == "tworoom":
        return gym.make("swm/TwoRoom-v1", render_mode="rgb_array", disable_env_checker=True)
    if task == "reacher":
        return gym.make(
            "swm/ReacherDMControl-v0",
            task="qpos_match",
            disable_env_checker=True,
        )
    if task == "cube":
        return gym.make(
            "swm/OGBCube-v0",
            env_type="single",
            ob_type="states",
            height=224,
            width=224,
            mode="data_collection",
            # Match official Cube evaluation: hide the task target marker.
            visualize_info=False,
            terminate_at_goal=True,
            disable_env_checker=True,
        )
    raise ValueError(f"unsupported task {task}")

def _reset_env(
    env: gym.Env,
    task: str,
    row: dict[str, Any],
    start_state: np.ndarray,
    goal_state: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if task == "tworoom":
        env.reset(
            seed=int(row["environment_seed"]),
            options={"state": start_state[:2]},
        )
        env.unwrapped._set_state(start_state[:2])
        env.unwrapped._set_goal_state(goal_state[:2])
        obs = env.unwrapped._get_obs()
        info = env.unwrapped._get_info()
        info["distance_to_target"] = float(
            np.linalg.norm(start_state[:2] - goal_state[:2])
        )
        return np.asarray(obs, dtype=np.float32), info
    if task == "cube":
        # CubeEnv discards the reset seed when it samples its default start/goal
        # variations.  Both values are replaced below by the registered state,
        # so disable that redundant sampling to keep rendered observations exact.
        env.reset(seed=int(row["environment_seed"]), options={"variation": []})
        qpos, qvel = _cube_state_to_sim(start_state)
        env.unwrapped.set_state(qpos, qvel)
        goal_pos = goal_state[19:22] / 10.0 + np.asarray([0.425, 0.0, 0.0])
        env.unwrapped._target_block = 0
        env.unwrapped.set_target_pos(0, goal_pos, goal_state[22:26])
        env.unwrapped.pre_step()
        env.unwrapped.post_step()
        obs = env.unwrapped.compute_observation()
        return np.asarray(obs, dtype=np.float32), env.unwrapped.get_step_info()
    env.reset(
        seed=int(row["environment_seed"]),
        options={"target_qpos": goal_state[:2].astype(np.float64)},
    )
    # The official Reacher evaluation is qpos matching: the future trajectory
    # state defines the goal, while the rendered target remains hidden.
    env.unwrapped.set_state(
        start_state[:2].astype(np.float64), start_state[4:6].astype(np.float64)
    )
    # The released Reacher rows contain the standard target marker only
    # indirectly, as ``observation[2:4] == target_pos - finger_pos``.  The
    # manifest environment seed is a replay seed, not the data-generation
    # seed, so reset() alone samples a different hidden marker.  Restore the
    # recorded marker from the state row so the complete DM-Control
    # observation, including the non-rendered to_target diagnostic, matches
    # the released trajectory.
    physics = env.unwrapped.env.physics
    finger_pos = np.asarray(env.unwrapped.info["finger_pos"], dtype=np.float64)
    target_pos = finger_pos + start_state[2:4].astype(np.float64)
    physics.named.model.geom_pos["target", "x"] = target_pos[0]
    physics.named.model.geom_pos["target", "y"] = target_pos[1]
    physics.forward()
    obs = env.unwrapped._obs_to_array(
        env.unwrapped.env.task.get_observation(env.unwrapped.env.physics)
    )
    return np.asarray(obs, dtype=np.float32), env.unwrapped.info

def _success(
    task: str,
    info: dict[str, Any],
    observation: np.ndarray,
    threshold: float,
    goal_state: np.ndarray,
) -> bool:
    if task == "tworoom":
        return float(info.get("distance_to_target", np.inf)) < 16.0
    if task == "cube":
        return bool(info.get("success", False))
    qpos = np.asarray(info["qpos"], dtype=np.float32)
    return bool(np.all(np.abs(qpos - np.asarray(goal_state[:2])) < threshold))
def official_success(goal: np.ndarray, state: np.ndarray) -> bool:
    angle = abs(float(goal[4]) - float(state[4])) % (2 * math.pi)
    angle = min(angle, 2 * math.pi - angle)
    return bool(np.linalg.norm(goal[:4] - state[:4]) < 20 and angle < math.pi / 9)
def _make_live(
    context: dict[str, Any], row: dict[str, Any]
) -> tuple[Any, Any, dict[str, Any], np.ndarray]:
    task = context["task"]
    arrays = context["arrays"]
    start = arrays["states"][int(row["global_start"])].astype(np.float64)
    goal = arrays["states"][int(row["global_goal"])].astype(np.float64)
    if task == "pusht":
        env = PushT(resolution=224, render_mode="rgb_array", relative=True)
        env.reset(seed=int(row["environment_seed"]))
        env._set_state(start)
        env._set_goal_state(goal)
        return env, env._get_obs(), env._get_info(), goal
    env = _make_env(task)
    observation, info = _reset_env(env, task, row, start, goal)
    return env, observation, info, goal

def _elapsed(env: Any) -> int | None:
    return int(env._elapsed_steps) if hasattr(env, "_elapsed_steps") else None

def _snapshot(task: str, env: Any) -> dict[str, Any]:
    if task == "pusht":
        return {"state": np.asarray(env._get_obs()).copy(), "elapsed": _elapsed(env)}
    if task == "cube":
        return {
            "qpos": env.unwrapped.data.qpos.copy(),
            "qvel": env.unwrapped.data.qvel.copy(),
            "elapsed": _elapsed(env),
        }
    if task == "reacher":
        return {
            "physics": env.unwrapped.env.physics.get_state().copy(),
            "elapsed": _elapsed(env),
        }
    return {"state": np.asarray(env.unwrapped._get_obs()).copy(), "elapsed": _elapsed(env)}

def _success_now(task: str, observation: Any, info: dict[str, Any], goal: np.ndarray, terminated: bool = False, truncated: bool = False) -> bool:
    if task == "pusht":
        state = np.asarray(observation["state"] if isinstance(observation, dict) else observation)
        return bool(terminated or truncated or official_success(goal, state))
    return bool(terminated or truncated or _success(task, info, observation, 0.05, goal))
