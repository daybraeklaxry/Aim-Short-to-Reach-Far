"""Render paired AP-rank demonstrations from existing physical-state traces."""
import argparse
import json
from pathlib import Path
import subprocess
from render_project_demos import make_env, restore, image, goal_image


def render(case, root, out):
    task, query, condition = case['task'], case['query_id'], case['condition']
    arms = ['ap_final', 'ap_observed']
    traces = [json.loads((root / task / f'confirm_{query}_{condition}_{arm}.json').read_text()) for arm in arms]
    assert traces[0]['trajectory'][0] == traces[1]['trajectory'][0]
    assert traces[0]['true_goal'] == traces[1]['true_goal']
    for trace, side in zip(traces, ['baseline', 'ap']):
        assert len(trace['trajectory']) == case[f'{side}_steps'] + 1
        assert trace['compact']['success'] == case[f'{side}_success']
        assert trace['compact']['primitive_steps'] == case[f'{side}_steps']
    name = f'rank-final-{task}-{case["query_ordinal"]:03d}'
    assets = out / 'comparisons'
    demos = out / 'demos'
    assets.mkdir(parents=True, exist_ok=True)
    demos.mkdir(parents=True, exist_ok=True)
    longest = max(case['baseline_steps'], case['ap_steps'])
    for trace, arm, side in zip(traces, arms, ['baseline', 'ap']):
        env = make_env(task, trace)
        if side == 'ap':
            goal_image(env, task, trace).save(demos / f'{name}-goal.jpg', quality=90)
        frames = trace['trajectory']
        cmd = ['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '360x360', '-r', '20', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '23',
               '-preset', 'fast', '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(assets / f'{name}-{side}.mp4')]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        previous = -1
        for step in list(range(longest + 1)) + [longest] * 30:
            index = min(step, len(frames) - 1)
            if index != previous:
                restore(env, task, frames[index])
                pic = image(env)
                previous = index
                if step == 0:
                    pic.save(assets / f'{name}-{side}.jpg', quality=90)
                if index == len(frames) - 1:
                    pic.save(out / f'{name}-{side}-end.jpg', quality=90)
            proc.stdin.write(pic.tobytes())
        proc.stdin.close()
        assert proc.wait() == 0
        env.close()
        compact = {k: trace['compact'][k] for k in ['success', 'primitive_steps', 'budget', 'horizon'] if k in trace['compact']}
        compact['identity'] = {k: v for k, v in trace['compact']['identity'].items()
                               if k in ['record_id', 'environment_seed', 'task', 'horizon', 'episode', 'global_start', 'global_goal']}
        public = dict(compact=compact, trajectory=trace['trajectory'], true_goal=trace['true_goal'])
        dest = out / 'traces' / task
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f'confirm_{query}_{condition}_{arm}.json').write_text(json.dumps(public) + '\n')
    meta = dict(case, name=name, fps=20, actions_per_second=20,
                rendering='Replay of recorded physical states, holding each final state after stopping.',
                selection='First two query-ordered AP-rank successes longer than ten actions paired with final-goal ranking failures under the first assigned perturbation.',
                model_calls_for_rendering=0, simulated_actions_for_rendering=0)
    (demos / f'{name}.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps({'case':name, 'baseline_success':case['baseline_success'], 'ap_success':case['ap_success']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--trace-root', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', required=True)
    args = p.parse_args()
    for case in json.loads(args.cases.read_text()):
        if case['task'] == args.task:
            render(case, args.trace_root, args.output)
