"""Collect only registered shards and publish rates only after complete paired cells."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse, json

ROOT = Path(__file__).resolve().parent
BASE = Path('runtime/ap/fresh_target_interface_study_v2')
TASKS = ['cube', 'pusht', 'reacher', 'tworoom']
STARTS = {'Standard': ['clean'], 'Perturbed 1': ['prefix_a'],
          'Perturbed 2': ['prefix_b'], 'Perturbed': ['prefix_a', 'prefix_b']}

def collect(progress=False):
    protocol = json.loads((BASE / 'protocol.json').read_text())
    jobs = json.loads((ROOT / 'official_launch.json').read_text())
    expected = set()
    original = {t: int(protocol['studies'][t]['horizon']) for t in TASKS}
    for task in TASKS:
        for h in sorted({25, 50, 100, original[task]}):
            for mode in (['released_raw', 'released_box'] if h == original[task] else ['released_raw']):
                expected.update((task, h, q, cond, mode) for q in range(128) for cond in STARTS['Standard'] + STARTS['Perturbed'])
    rows, seen, statuses = [], set(), []
    for job in jobs:
        task, start, stop = job['task'], job['start'], job['stop']
        folder = ROOT / 'official_results' / task
        path = folder / f'outcomes_q{start:03}-{stop:03}.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for r in records:
            key = (task, r['H'], r['query_ordinal'], r['condition'], r['mode'])
            assert r['task'] == task and start <= r['query_ordinal'] < stop
            assert key in expected and key not in seen, key
            assert r['record_id'] == protocol['studies'][task]['confirm'][r['query_ordinal']]['record_id']
            assert r['B'] == r['H'] * (1 if task == 'cube' else 2)
            assert 0 <= r['primitive_steps'] <= r['B']
            assert r['prediction_blocks'] == r['decisions'] * 45000
            assert r['predictor_calls'] == r['decisions'] * 150
            seen.add(key)
            rows.append(r)
        proc = Path(f'/proc/{job["pid"]}')
        alive = proc.exists() and (proc / 'stat').read_text().split()[2] != 'Z'
        done = folder / f'complete_q{start:03}-{stop:03}.json'
        tail = ''
        if not alive and not done.exists():
            log = Path(job['log'])
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 1600))
                tail = stream.read().decode(errors='replace')
        statuses.append({'task': task, 'range': [start, stop], 'gpu': job['gpu'],
                         'outcomes': len(records), 'alive': alive, 'complete': done.exists(), 'error_tail': tail})
    complete = seen == expected
    by_task = {t: sum(r['task'] == t for r in rows) for t in TASKS}
    status = {'complete': complete, 'episodes': len(rows), 'expected': len(expected),
              'by_task': by_task, 'active_workers': sum(s['alive'] for s in statuses),
              'finished_workers': sum(s['complete'] for s in statuses),
              'failed': [s for s in statuses if not s['alive'] and not s['complete']]}
    if progress or not complete:
        print(json.dumps(status))
        return
    cells = []
    for task in TASKS:
        for h in sorted({25, 50, 100, original[task]}):
            for mode in (['released_raw', 'released_box'] if h == original[task] else ['released_raw']):
                for start, conditions in STARTS.items():
                    selected = [r for r in rows if r['task'] == task and r['H'] == h and r['mode'] == mode and r['condition'] in conditions]
                    counts = Counter(r['query_ordinal'] for r in selected)
                    assert len(counts) == 128 and set(counts.values()) == {len(conditions)}
                    n = len(selected)
                    cells.append({'task': task, 'H': h, 'B': selected[0]['B'], 'mode': mode,
                                  'start': start, 'complete': True, 'query_count': 128,
                                  'completed_outcomes': n, 'expected_outcomes': n,
                                  'success_percent': 100 * sum(r['success'] for r in selected) / n,
                                  'mean_pred_blocks': sum(r['prediction_blocks'] for r in selected) / n,
                                  'mean_steps': sum(r['primitive_steps'] for r in selected) / n,
                                  'mean_decisions': sum(r['decisions'] for r in selected) / n})
    payload = {'collected_utc': datetime.now(timezone.utc).isoformat(), **status,
               'primary': 'released_raw', 'original_offsets': original,
               'aggregation': 'Each query has equal weight; Perturbed averages its two prefixes; task means weight tasks equally.',
               'manifest': jobs, 'rows': cells}
    (ROOT / 'official_summary.json').write_text(json.dumps(payload, indent=2))
    with (ROOT / 'official_outcomes.jsonl').open('w') as stream:
        for r in sorted(rows, key=lambda r: (r['task'], r['H'], r['query_ordinal'], r['condition'], r['mode'])):
            stream.write(json.dumps(r) + '\n')
    print(json.dumps({**status, 'cells': len(cells), 'summary': str(ROOT / 'official_summary.json')}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--progress', action='store_true')
    collect(parser.parse_args().progress)
