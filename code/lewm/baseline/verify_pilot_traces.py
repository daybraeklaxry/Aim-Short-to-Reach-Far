from pathlib import Path
import json
import numpy as np

root = Path(__file__).resolve().parent
rows = []
for q in range(8):
    old = root / 'traces' / f'pusht_H25_q{q:03}_released_execute_25'
    new = root / 'official_results/pusht/H25/released_raw' / f'q{q:03}_clean'
    a, b = np.load(old.with_suffix('.npz')), np.load(new.with_suffix('.npz'))
    same = np.array_equal(a['actions'], b['actions'])
    goal_same = np.array_equal(a['goal'].reshape(-1), b['scoring_goal_encoding'].reshape(-1))
    prior = json.loads(old.with_suffix('.json').read_text())['result']
    current = json.loads(new.with_suffix('.json').read_text())['compact']
    row = {'query': q, 'actions_equal': same, 'goal_equal': goal_same,
           'steps': current['primitive_steps'], 'success_equal': prior['success'] == current['success']}
    assert same and goal_same and row['success_equal'], row
    rows.append(row)
(root / 'pilot_trace_equivalence.json').write_text(json.dumps(rows, indent=2))
print(json.dumps({'queries': len(rows), 'all_executed_actions_equal': True,
                  'all_goal_encodings_equal': True, 'all_success_outcomes_equal': True}))
