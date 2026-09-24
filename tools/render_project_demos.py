"""Render paired saved simulator states. No policy, model, or env.step calls."""
import argparse
import json
from pathlib import Path
import subprocess

import gymnasium as gym
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import stable_worldmodel.envs
from stable_worldmodel.envs.pusht.env import PushT


def make_env(task, trace):
    seed = trace['compact']['identity']['environment_seed']
    goal = np.asarray(trace['true_goal'], dtype=np.float64)
    if task == 'pusht':
        env = PushT(resolution=320, render_mode='rgb_array', relative=True)
        env.reset(seed=seed)
        env._set_goal_state(goal)
        # Show the evaluated goal, not PushT's unrelated default green marker.
        env.goal_pose = goal[2:5].copy()
    elif task == 'cube':
        env = gym.make('swm/OGBCube-v0', env_type='single', ob_type='states',
                       height=320, width=320, mode='data_collection',
                       visualize_info=False, terminate_at_goal=True, disable_env_checker=True)
        env.reset(seed=seed, options={'variation': []})
        env.unwrapped._target_block = 0
        env.unwrapped.set_target_pos(0, goal[19:22] / 10 + np.array([.425, 0, 0]), goal[22:26])
    elif task == 'reacher':
        env = gym.make('swm/ReacherDMControl-v0', task='qpos_match', disable_env_checker=True)
        env.reset(seed=seed, options={'target_qpos': goal[:2]})
    else:
        env = gym.make('swm/TwoRoom-v1', render_mode='rgb_array', disable_env_checker=True)
        env.reset(seed=seed, options={'state': np.asarray(trace['trajectory'][0]['state'][:2])})
        env.unwrapped._set_goal_state(goal[:2])
    return env


def restore(env, task, snapshot):
    if task == 'cube':
        e = env.unwrapped
        s = np.asarray(snapshot['integration'], dtype=np.float64)
        mujoco.mj_setState(e.model, e.data, s, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(e.model, e.data)
        assert np.allclose(e.data.qpos, s[1:1 + e.model.nq], rtol=0, atol=1e-10)
    elif task == 'pusht':
        # PushT._set_state steps physics. Assign the logged pose directly instead.
        s = np.asarray(snapshot['state'])
        env.agent.position = tuple(s[:2])
        env.block.angle = float(s[4])
        env.block.position = tuple(s[2:4])
        env.agent.velocity = tuple(s[-2:])
        env.space.reindex_shapes_for_body(env.agent)
        env.space.reindex_shapes_for_body(env.block)
        assert np.allclose(env._get_obs(), s, rtol=0, atol=1e-8)
    elif task == 'reacher':
        p = env.unwrapped.env.physics
        s = np.asarray(snapshot['physics'])
        p.set_state(s)
        p.forward()
        assert np.allclose(p.get_state(), s, rtol=0, atol=1e-10)
    else:
        s = np.asarray(snapshot['state'])
        env.unwrapped._set_state(s[:2])
        assert np.allclose(np.asarray(env.unwrapped._get_obs())[:2], s[:2], rtol=0, atol=1e-5)


def image(env):
    a = np.asarray(env.render())
    if a.ndim == 3 and a.shape[0] == 3:
        a = a.transpose(1, 2, 0)
    if a.dtype != np.uint8:
        a = np.clip(a * (255 if a.max() <= 1.01 else 1), 0, 255).astype(np.uint8)
    return Image.fromarray(a[..., :3]).convert('RGB').resize((360, 360), Image.Resampling.LANCZOS)


def goal_image(env, task, trace):
    g = np.asarray(trace['true_goal'], dtype=np.float64)
    if task == 'cube':
        qpos = np.zeros(21)
        qvel = np.zeros(20)
        qpos[:6], qvel[:6] = g[:6], g[6:12]
        qpos[6] = qpos[10] = .8 * np.clip(g[17] / 3, 0, 1)
        qpos[14:17] = g[19:22] / 10 + np.array([.425, 0, 0])
        qpos[17:21] = g[22:26]
        env.unwrapped.set_state(qpos, qvel)
    elif task == 'reacher':
        env.unwrapped.env.physics.set_state(np.r_[g[:2], g[4:6]])
        env.unwrapped.env.physics.forward()
    else:
        restore(env, task, {'state': g.tolist()})
    return image(env)


def render(case, root, out, preview):
    task, query = case['task'], case['query_id']
    traces = [json.loads((root / task / f'confirm_{query}_clean_{arm}.json').read_text())
              for arm in ['gaussian_final', 'gaussian_observed']]
    assert traces[0]['trajectory'][0] == traces[1]['trajectory'][0]
    assert traces[0]['true_goal'] == traces[1]['true_goal']
    assert [t['compact']['success'] for t in traces] == [False, True]
    for t, count in zip(traces, [case['final_steps'], case['ap_steps']]):
        assert len(t['trajectory']) == count + 1 == t['compact']['primitive_steps'] + 1
    envs = [make_env(task, t) for t in traces]
    name = f'{task}-{case["query_ordinal"]:03d}'
    goal_image(envs[1], task, traces[1]).save(out / f'{name}-goal.jpg', quality=90)
    fonts = '/usr/share/fonts/truetype/dejavu/DejaVuSans'
    label = ImageFont.truetype(fonts + '-Bold.ttf', 20)
    small = ImageFont.truetype(fonts + '.ttf', 17)
    steps = max(case['final_steps'], case['ap_steps'])
    colors = ['#666666', '#24579b']
    cached_indices = [-1, -1]
    cached_images = [None, None]
    def frame(t):
        canvas = Image.new('RGB', (744, 426), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, (e, trace) in enumerate(zip(envs, traces)):
            index = min(t, len(trace['trajectory']) - 1)
            if cached_indices[i] != index:
                restore(e, task, trace['trajectory'][index])
                cached_images[i] = image(e)
                cached_indices[i] = index
            pic = cached_images[i]
            x = i * 384
            canvas.paste(pic, (x, 38))
            draw.text((x + 180, 17), ['Final-goal CEM', 'AP-CEM'][i], fill=colors[i], font=label, anchor='mm')
            status = f'Action {index}'
            if i == 1 and t >= case['ap_steps']:
                status = f'Reached at action {case["ap_steps"]}'
            elif i == 0 and t >= case['final_steps']:
                status = f'Not reached in {case["final_steps"]} actions'
            draw.text((x + 180, 413), status, fill=colors[i], font=small, anchor='mm')
        return canvas

    frame(0).save(out / f'{name}-start.jpg', quality=88)
    frame(min(steps, max(case['ap_steps'], steps // 2))).save(out / f'{name}-poster.jpg', quality=88)
    frame(steps).save(out / f'{name}-end.jpg', quality=88)
    if not preview:
        cmd = ['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '744x426', '-r', '20', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '23',
               '-preset', 'fast', '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(out / f'{name}.mp4')]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for t in [0] * 10 + list(range(steps + 1)) + [steps] * 30:
            proc.stdin.write(frame(t).tobytes())
        proc.stdin.close()
        assert proc.wait() == 0
    for e in envs:
        e.close()
    meta = dict(case, name=name, fps=20, actions_per_second=20, start='standard',
                final_success=False, ap_success=True, model_calls=0, simulated_actions=0,
                rendering='Simulator rendering of saved physical states; terminal states held.',
                goal_overlay='PushT green overlay shows the evaluated goal pose.' if task == 'pusht' else None,
                selection='First two query-ordered AP-CEM successes paired with final-goal failures per task, requiring more than ten executed AP-CEM actions.')
    (out / f'{name}.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps({'case': name, 'preview': preview, 'steps': steps}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--trace-root', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--preview-only', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    for c in json.loads(a.cases.read_text()):
        if c['task'] == a.task:
            render(c, a.trace_root, a.output, a.preview_only)
