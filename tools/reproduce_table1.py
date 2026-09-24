"""Recompute Table 1 from the released per-episode outcomes (standard library only)."""
from pathlib import Path
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
import argparse
import csv

ROOT = Path(__file__).resolve().parents[1]
TASKS = ['cube', 'pusht', 'reacher', 'tworoom']
NAMES = ['Cube', 'PushT', 'Reacher', 'TwoRoom']
ARMS = ['LeWM planner', 'CEM (final goal)', 'CEM (learned target)', 'AP-CEM',
        'Direct', 'Rank (final goal)', 'Rank (learned target)', 'AP-rank']
STARTS = {'Standard': ['clean'], 'Perturbed 1': ['prefix_a'],
          'Perturbed 2': ['prefix_b'], 'Perturbed': ['prefix_a', 'prefix_b']}

def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))

def formatted(value):
    return str(Decimal(str(value)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'table1')
    args = parser.parse_args()
    episodes = read_csv(ROOT / 'data' / 'main_episodes.csv')
    groups = defaultdict(list)
    identities = set()
    for row in episodes:
        identity = (row['task'], row['controller'], row['condition'], row['query_ordinal'])
        assert identity not in identities, identity
        identities.add(identity)
        groups[identity[:3]].append(row)
    assert len(episodes) == 12288
    summary = []
    rates = {}
    for task in TASKS:
        for arm in ARMS:
            for start, conditions in STARTS.items():
                for condition in conditions:
                    selected = groups[(task, arm, condition)]
                    assert len(selected) == 128
                    assert {int(r['query_ordinal']) for r in selected} == set(range(128))
                rows = [r for c in conditions for r in groups[(task, arm, c)]]
                # Equal, paired prefixes: pooling equals averaging each query's
                # two outcomes, then averaging across the 128 queries.
                rate = sum(int(r['success']) for r in rows) * 100 / len(rows)
                rates[(task, arm, start)] = rate
                summary.append(dict(task=task, arm=arm, start=start,
                                    outcomes=len(rows), success_percent=rate))
    reference = read_csv(ROOT / 'data' / 'main_results.csv')
    assert len(reference) == len(summary)
    for row in reference:
        key = (row['task'], row['arm'], row['start'])
        assert abs(rates[key] - float(row['success_percent'])) < 1e-9, key
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'table1_unrounded.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    heading = ['Task', 'Start', 'LeWM', 'CEM final', 'CEM learned', 'AP-CEM',
               'Direct', 'Rank final', 'Rank learned', 'AP-rank']
    lines = ['| ' + ' | '.join(heading) + ' |', '| ' + ' | '.join(['---'] * len(heading)) + ' |']
    printed = []
    for task, name in list(zip(TASKS, NAMES)) + [('mean', 'Mean')]:
        for start in ['Standard', 'Perturbed']:
            values = [(sum(rates[(t, arm, start)] for t in TASKS) / 4
                       if task == 'mean' else rates[(task, arm, start)]) for arm in ARMS]
            cells = [formatted(value) for value in values]
            printed.append(dict(zip(heading, [name, start] + cells)))
            lines.append('| ' + ' | '.join([name, start] + cells) + ' |')
    (args.output / 'table1.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with (args.output / 'table1.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=heading)
        writer.writeheader(); writer.writerows(printed)
    print(f'Recomputed {len(episodes):,} outcomes; all {len(summary)} task/controller/start cells match data/main_results.csv.')
    print(f'Table 1: {args.output / "table1.md"}')

if __name__ == '__main__':
    main()
