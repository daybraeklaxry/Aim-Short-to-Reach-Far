"""Progress and a runtime-based projection from the existing queue records."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import statistics
import subprocess

ROLLOUTS = {'main', 'long', 'transport', 'budget', 'ablation', 'horizon'}


def family(job):
    arm = job['config'].get('arm', 'lewm' if job['kind'] == 'long' else '')
    return 'lewm' if arm == 'lewm' else ('rank' if arm.startswith('rank') else 'cem')


def write_progress(root, jobs, state):
    now = datetime.now(timezone.utc)
    samples = defaultdict(list)
    for job in jobs.values():
        entry = state['jobs'].get(job['id'], {})
        if job['kind'] not in ROLLOUTS or entry.get('status') != 'complete':
            continue
        cfg = job['config']
        n = (cfg['stop'] - cfg['start']) * len(cfg['conditions'])
        seconds = (datetime.fromisoformat(entry['completed_utc'])
                   - datetime.fromisoformat(entry['started_utc'])).total_seconds()
        key = (job['task'], family(job), cfg.get('iterations', 30))
        samples[key].append(seconds / n)
    workers = sum(v['status'] == 'running' for v in state['jobs'].values())
    remaining = 0.0
    pending = []
    for job in jobs.values():
        entry = state['jobs'].get(job['id'], {})
        if job['kind'] not in ROLLOUTS or entry.get('status') not in ('running', 'queued'):
            continue
        pending.append(job)
        cfg = job['config']
        key = (job['task'], family(job), cfg.get('iterations', 30))
        n = (cfg['stop'] - cfg['start']) * len(cfg['conditions'])
        expected = statistics.median(samples[key]) * n
        elapsed = ((now - datetime.fromisoformat(entry['started_utc'])).total_seconds()
                   if entry['status'] == 'running' else 0)
        remaining += max(0.0, expected - elapsed)
    counts = defaultdict(Counter)
    for job in jobs.values():
        counts[job['kind']][state['jobs'].get(job['id'], {}).get('status', 'queued')] += 1
    lines = ['# Experiment progress', '', 'Updated UTC: ' + now.isoformat(), '',
             '| Study | Complete | Running | Queued | Failed |',
             '|---|---:|---:|---:|---:|']
    for kind, c in counts.items():
        lines.append('| ' + kind + ' | ' + ' | '.join(str(c[k]) for k in
                     ('complete', 'running', 'queued', 'failed')) + ' |')
    if pending and workers:
        lines += ['', f'Provisional queue runtime: about {remaining / 3600 / workers:.1f} hours '
                  f'at the current {workers} workers ({remaining / 3600:.1f} worker-hours).',
                  'This projects the median runtime per assigned start for each task, controller '
                  'family, and CEM budget. Shorter-goal runtime profiles are still accumulating. '
                  'It excludes final analysis and manuscript preparation.']
    resources = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.total,memory.used,utilization.gpu', '--format=csv'],
        capture_output=True, text=True, check=True).stdout.strip()
    lines += ['', '```text', resources, '```', '',
              'All controller implementations are available. Queued jobs wait for worker slots '
              'and sufficient free GPU memory; completed outcomes are reused.']
    (root / 'PROGRESS.md').write_text('\n'.join(lines) + '\n')
