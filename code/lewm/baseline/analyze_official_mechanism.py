"""Apply the already published behavior definitions to the corrected LeWM traces."""
from pathlib import Path
import ast, json
import numpy as np
import tokenizers
import mujoco
from experiments.delta_jepa import sota_generic_retrieval as generic

ROOT = Path(__file__).resolve().parent
OLD = Path('runtime/ap/evidence_strengthening_20260923/results_v2')
summary = json.loads((ROOT / 'official_summary.json').read_text())
assert summary['complete']
env = generic._make_env('cube')
env.reset(seed=42, options={'variation': []})
model, data = env.unwrapped.model, env.unwrapped.data
kind = mujoco.mjtState.mjSTATE_INTEGRATION
v = np.empty(mujoco.mj_stateSize(model, kind))
mujoco.mj_getState(model, data, v, kind)
np.testing.assert_array_equal(v[1:1 + model.nq], data.qpos)
object_adr = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'object_joint_0')])
env.close()
tree = ast.parse((OLD / 'code/analyze_physical.py').read_text())
helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {'coords', 'angle', 'physical_error', 'stalling'}]
assert len(helpers) == 4
namespace = {'np': np, 'object_adr': object_adr}
exec(compile(ast.Module(body=helpers, type_ignores=[]), 'unchanged_physical_metrics', 'exec'), namespace)
coords, physical_error, stalling = (namespace[k] for k in ['coords', 'physical_error', 'stalling'])
outcomes = [json.loads(line) for line in (ROOT / 'official_outcomes.jsonl').read_text().splitlines()]
outcomes = [r for r in outcomes if r['mode'] == 'released_raw' and r['H'] == summary['original_offsets'][r['task']]]
assert len(outcomes) == 1536
rows = []
for r in outcomes:
    task = r['task']
    stem = ROOT / 'official_results' / task / f"H{r['H']}" / 'released_raw' / f"q{r['query_ordinal']:03}_{r['condition']}"
    trace = json.loads(stem.with_suffix('.json').read_text())
    assert trace['compact'] == r
    x = coords(task, trace['trajectory'])
    error = physical_error(task, x, trace['true_goal'])
    times = [d['step'] for d in trace['decisions']]
    assert len(x) == r['primitive_steps'] + 1 and len(times) == r['decisions'] and np.isfinite(error).all()
    entered = r['prelude']['category'] == 'policy_entered'
    if entered and r['success']:
        assert error[-1] <= 1 + 1e-5, (task, r['query_ordinal'], error[-1])
    row = {'task': task, 'query_ordinal': r['query_ordinal'], 'condition': r['condition']}
    for w in [5, 10, 20]:
        begin = times[-min(w, len(times))] if times else 0
        row[f'stall_W{w}'] = stalling(task, x[begin:]) if not r['success'] and times else None
    rise = float((error - np.minimum.accumulate(error)).max())
    for delta in [.25, .5, 1.0]:
        row[f'physical_detour_{delta}'] = rise >= delta if entered and r['success'] else None
    latent = None
    with np.load(stem.with_suffix('.npz')) as z:
        if times:
            distance = np.square(z['score_current'].astype(float) - z['scoring_goal_encoding'].reshape(-1).astype(float)).sum(-1)
            assert len(distance) == len(times)
            if distance[0] > 1e-12:
                latent = distance / distance[0]
    row['latent_detour'] = bool((latent - np.minimum.accumulate(latent)).max() >= .1) if entered and r['success'] and latent is not None else None
    rows.append(row)
cells = []
for task in summary['original_offsets']:
    for label, conditions in [('Standard', ['clean']), ('Perturbed 1', ['prefix_a']), ('Perturbed 2', ['prefix_b']), ('Perturbed', ['prefix_a', 'prefix_b'])]:
        selected = [r for r in rows if r['task'] == task and r['condition'] in conditions]
        assert len(selected) == 128 * len(conditions)
        cell = {'task': task, 'start': label, 'complete': True}
        for metric in ['stall_W5', 'stall_W10', 'stall_W20', 'physical_detour_0.25', 'physical_detour_0.5', 'physical_detour_1.0', 'latent_detour']:
            eligible = [r[metric] for r in selected if r[metric] is not None]
            n = sum(bool(v) for v in eligible)
            cell[metric] = {'numerator': n, 'denominator': len(eligible), 'percent': 100 * n / len(eligible) if eligible else None}
        cells.append(cell)
result = {'complete': True, 'episodes': len(rows), 'rows': cells,
          'physical_metric_functions': 'Unchanged functions from analyze_physical.py',
          'cube_object_qpos_index': object_adr, 'integration_layout_checked': True}
(ROOT / 'official_mechanism.json').write_text(json.dumps(result, indent=2))
print(json.dumps({'complete': True, 'episodes': len(rows), 'cells': len(cells)}))
