"""Summarize complete, declared factors and replanning cells; never fill pending cells."""
from pathlib import Path
import csv,json,statistics

R=Path(__file__).parent;OUT=R/'lewm_checks';OUT.mkdir(exist_ok=True)
OLD=R.parent/'baseline/official_results'
tasks=['cube','pusht','reacher','tworoom']

def write(name,rows):
    if rows:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
            w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)

baseline={}
for task in tasks:
    baseline[task]=[json.loads((OLD/task/'H25/released_raw'/f'q{i:03}_clean.json').read_text())['compact'] for i in range(128)]
factors=[]
for folder in sorted((R/'gap_results').iterdir()):
    if not folder.is_dir():continue
    for task in tasks:
        rows=[]
        for f in folder.glob(f'{task}_*.jsonl'):rows.extend(json.loads(s) for s in f.read_text().splitlines())
        if len(rows)!=128:continue
        assert sorted(d['query_ordinal'] for d in rows)==list(range(128))
        base=baseline[task];x=sum(d['success'] for d in base);y=sum(d['success'] for d in rows)
        factors.append(dict(task=task,factor=folder.name,episodes=128,main_H25_successes=x,factor_successes=y,main_success_percent=x/128*100,factor_success_percent=y/128*100,gain_percentage_points=(y-x)/128*100,same_query_sources=folder.name!='official_queries',action_budget=rows[0]['B'],actual_goal_frame_offset=rows[0]['actual_goal_frame_offset'],mean_prediction_blocks=statistics.mean(d['prediction_blocks'] for d in rows),interpretation='one factor changed; effects are not an additive decomposition'))
rescored=list(csv.DictReader((OUT/'success_rescoring.csv').open()))
for task in tasks:
    rows=[d for d in rescored if d['task']==task];assert len(rows)==128
    assert all(d['paper_matches_record']=='True' for d in rows)
    x=sum(d['success'] for d in baseline[task]);y=sum(d['released_predicate_success']=='True' for d in rows)
    factors.append(dict(task=task,factor='success_predicate_rescore',episodes=128,main_H25_successes=x,factor_successes=y,main_success_percent=x/128*100,factor_success_percent=y/128*100,gain_percentage_points=(y-x)/128*100,same_query_sources=True,action_budget=baseline[task][0]['B'],actual_goal_frame_offset=25,mean_prediction_blocks=statistics.mean(d['prediction_blocks'] for d in baseline[task]),interpretation='saved main trajectories rescored; no new rollout or prediction work'))
write('gap_attribution.csv',factors)

records={}
if (R/'replan5_results').exists():
    for f in sorted((R/'replan5_results').glob('*.jsonl')):
        for line in f.read_text().splitlines():
            d=json.loads(line);key=(d['task'],d['query_ordinal'],d['condition']);assert key not in records,key;records[key]=d
cells=[]
for task in tasks:
    for condition in ['clean','prefix_a','prefix_b']:
        rows=[v for k,v in records.items() if k[0]==task and k[2]==condition]
        if len(rows)!=128:continue
        assert sorted(d['query_ordinal'] for d in rows)==list(range(128))
        cells.append(dict(task=task,condition=condition,episodes=128,successes=sum(d['success'] for d in rows),success_percent=statistics.mean(d['success'] for d in rows)*100,mean_prediction_blocks=statistics.mean(d['prediction_blocks'] for d in rows),mean_primitive_steps=statistics.mean(d['primitive_steps'] for d in rows),mean_decisions=statistics.mean(d['decisions'] for d in rows),planned_actions=25,maximum_executed_actions=5))
write('lewm_replan5.csv',cells)
write('lewm_replan5_episodes.csv',[records[k] for k in sorted(records)])
status={'complete_factors':len(factors),'expected_factors':18,'replan5_completed_episodes':len(records),'replan5_expected_episodes':1536,'replan5_complete_cells':len(cells),'replan5_expected_cells':12}
(OUT/'followup_status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status))
