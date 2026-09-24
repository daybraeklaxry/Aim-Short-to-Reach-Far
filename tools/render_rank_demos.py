"""Render AP-rank and Direct from paired saved physical states."""
import argparse
import json
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw, ImageFont
from render_project_demos import make_env, restore, image, goal_image


def render(case, root, out, trace_export):
    task, query, condition = case['task'], case['query_id'], case['condition']
    arms = ['direct', 'ap_observed']
    traces = [json.loads((root / task / f'confirm_{query}_{condition}_{arm}.json').read_text()) for arm in arms]
    assert traces[0]['trajectory'][0] == traces[1]['trajectory'][0]
    assert traces[0]['true_goal'] == traces[1]['true_goal']
    assert [t['compact']['success'] for t in traces] == [False, True]
    for trace, count in zip(traces, [case['direct_steps'], case['rank_steps']]):
        assert len(trace['trajectory']) == count + 1 == trace['compact']['primitive_steps'] + 1
    envs = [make_env(task, t) for t in traces]
    name = f'rank-{task}-{case["query_ordinal"]:03d}'
    goal_image(envs[1], task, traces[1]).save(out / f'{name}-goal.jpg', quality=90)
    font = '/usr/share/fonts/truetype/dejavu/DejaVuSans'
    title = ImageFont.truetype(font + '-Bold.ttf', 20)
    status_font = ImageFont.truetype(font + '.ttf', 17)
    colors = ['#666666', '#24579b']
    indices, pics = [-1, -1], [None, None]
    steps = max(case['direct_steps'], case['rank_steps'])

    def frame(t):
        canvas = Image.new('RGB', (744, 426), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, (env, trace) in enumerate(zip(envs, traces)):
            index = min(t, len(trace['trajectory']) - 1)
            if indices[i] != index:
                restore(env, task, trace['trajectory'][index])
                pics[i] = image(env)
                indices[i] = index
            x = i * 384
            canvas.paste(pics[i], (x, 38))
            draw.text((x + 180, 17), ['Direct', 'AP-rank'][i], fill=colors[i], font=title, anchor='mm')
            status = f'Action {index}'
            if i == 1 and t >= case['rank_steps']:
                status = f'Reached at action {case["rank_steps"]}'
            elif i == 0 and t >= case['direct_steps']:
                status = f'Not reached in {case["direct_steps"]} actions'
            draw.text((x + 180, 413), status, fill=colors[i], font=status_font, anchor='mm')
        return canvas

    frame(steps).save(out / f'{name}-poster.jpg', quality=88)
    frame(0).save(out / f'{name}-start.jpg', quality=88)
    frame(steps).save(out / f'{name}-end.jpg', quality=88)
    cmd = ['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', '744x426', '-r', '20', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '23',
           '-preset', 'fast', '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / f'{name}.mp4')]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for t in [0] * 10 + list(range(steps + 1)) + [steps] * 30:
        proc.stdin.write(frame(t).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    for env in envs:
        env.close()
    meta = dict(case, name=name, fps=20, actions_per_second=20,
                rendering='Saved physical states; terminal states held after stopping.',
                selection='First two query-ordered Direct failures paired with AP-rank successes of more than ten actions, under the first assigned perturbation, on PushT and Cube.',
                model_calls_for_rendering=0, simulated_actions_for_rendering=0,
                goal_overlay='PushT green overlay shows the evaluated goal pose.' if task == 'pusht' else None)
    (out / f'{name}.json').write_text(json.dumps(meta, indent=2) + '\n')
    if trace_export:
        dest = trace_export / task
        dest.mkdir(parents=True, exist_ok=True)
        for arm, t in zip(arms, traces):
            identity = {k:v for k,v in t['compact']['identity'].items()
                        if k in ('record_id','environment_seed','task','horizon','episode','global_start','global_goal')}
            compact = {k:t['compact'][k] for k in ('success','primitive_steps','budget','horizon') if k in t['compact']}
            compact['identity'] = identity
            public = dict(compact=compact, trajectory=t['trajectory'], true_goal=t['true_goal'])
            (dest / f'confirm_{query}_{condition}_{arm}.json').write_text(json.dumps(public) + '\n')
    print(json.dumps({'case':name,'steps':steps}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--trace-root', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', choices=['pusht','cube'], required=True)
    p.add_argument('--export-traces', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for case in json.loads(args.cases.read_text()):
        if case['task'] == args.task:
            render(case, args.trace_root, args.output, args.export_traces)
