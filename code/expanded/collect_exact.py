"""Collect MAIN simulator diagnostics and eight first-block interventions."""
from pathlib import Path
from collections import defaultdict
import json,csv,datetime,os
import numpy as np
ROOT=Path('runtime/ap/evidence_strengthening_20260923/results_v2')
TASKS=['cube','pusht','reacher','tworoom']
SELECTORS={'direct':'Direct','learned_far':'Predictor-Final','learned_local':'Predictor-Observed','predictor_mlp':'Predictor-Learned','exact_far':'Simulator-Final','exact_local':'Simulator-Observed','exact_mlp':'Simulator-Learned','recorded_delta_local':'Displacement-Observed'}
GROUPS=[('Standard',['clean']),('Perturbed',['prefix_a','prefix_b']),('Perturbed 1',['prefix_a']),('Perturbed 2',['prefix_b'])]
def write(path,value):
 path.parent.mkdir(exist_ok=True,parents=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)
def paired_mean(rows,conditions,fn):
 # Fixed eligibility and fixed query weights: undefined starts do not become zero errors.
 values=[]
 for q in range(128):
  v=[fn(rows[(q,c)]) for c in conditions if (q,c) in rows]
  if v:values.append(float(np.mean(v)))
 return float(np.mean(values)) if values else None
all_rows={};flat=[]
for f in sorted((ROOT/'jobs').glob('exact-*/outcomes.jsonl')):
 for line in f.open():
  if not line.endswith('\n'):break
  r=json.loads(line);key=(r['task'],r['query_ordinal'],r['condition']);assert key not in all_rows,key
  assert r['completed'] and set(r['continuations'])==set(SELECTORS)
  if r['entered_policy']:assert r['main_prediction_replay_exact'] and r['direct_continuation_replay_exact']
  all_rows[key]=r
  d=r.get('prediction_diagnostics')
  if d:
   assert r['fixed_five_diagnostic_eligible'] and r['complete_five_candidate_count']==8
   item=dict(task=r['task'],query_ordinal=r['query_ordinal'],condition=r['condition'],predictor_error=d['learned_endpoint_mean_squared_l2_error'],displacement_error=d['recorded_delta_endpoint_mean_squared_l2_error'],predictor_regret=d['learned_local']['selected_exact_regret'],displacement_regret=d['recorded_delta_local']['selected_exact_regret'],direct_regret=d['learned_local']['direct_exact_cost']-d['learned_local']['selected_exact_cost']+d['learned_local']['selected_exact_regret'])
   assert min(item[k] for k in ['predictor_regret','displacement_regret','direct_regret'])>=-1e-5
   flat.append(item)
diag=[];first=[]
for task in TASKS:
 for group,conditions in GROUPS:
  rows={(q,c):r for (t,q,c),r in all_rows.items() if t==task and c in conditions}
  expected={(q,c) for q in range(128) for c in conditions};full=set(rows)==expected
  eligible={(r['query_ordinal'],r['condition']):r for r in flat if r['task']==task and r['condition'] in conditions}
  item=dict(task=task,start=group,complete=full,assigned=128*len(conditions),completed=len(rows),eligible_starts=len(eligible),eligible_queries=len({q for q,c in eligible}),eight_candidates_per_start=True)
  for metric in ['predictor_error','displacement_error','predictor_regret','displacement_regret','direct_regret']:
   item[metric]=paired_mean(eligible,conditions,lambda r:r[metric]) if full else None
  diag.append(item)
  for selector,label in SELECTORS.items():
   first.append(dict(task=task,start=group,selector=label,complete=full,assigned=128*len(conditions),completed=len(rows),success_percent=100*paired_mean(rows,conditions,lambda r:r['continuations'][selector]['success']) if full else None))
write(ROOT/'collected/diagnostic.json',dict(complete=all(r['complete'] for r in diag),rows=diag,aggregation='Within query average eligible starts, then equal weight eligible queries; endpoint errors first average eight candidates. All eight candidates must finish five primitive actions. Actual eligible start and query counts are reported.',source_outcomes=len(all_rows)))
write(ROOT/'collected/first_block.json',dict(complete=all(r['complete'] for r in first),rows=first,aggregation='All 128 queries; average paired perturbation starts within query then query means. Each selector changes only the first block; subsequent controller is identical Direct.',source_outcomes=len(all_rows)))
with (ROOT/'collected/diagnostic_episode_metrics.csv').open('w',newline='') as f:
 if flat:
  w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
if all(r['complete'] for r in first):
 target=ROOT/'collected/first_block_episodes.csv';temporary=target.with_suffix('.csv.tmp')
 with temporary.open('w',newline='') as stream:
  writer=csv.writer(stream)
  writer.writerow(['task','query_ordinal','condition','selector','entered_policy','success','primitive_steps','first_candidate_rank'])
  for (task,q,condition),row in sorted(all_rows.items()):
   for selector,label in SELECTORS.items():
    outcome=row['continuations'][selector]
    writer.writerow([task,q,condition,label,int(row['entered_policy']),int(outcome['success']),outcome['primitive_steps'],outcome['first_rank']])
 os.replace(temporary,target)
print(json.dumps(dict(outcomes=len(all_rows),fixed_five_starts=len(flat),complete_diagnostic_cells=sum(r['complete'] for r in diag),complete_first_block_cells=sum(r['complete'] for r in first))))
