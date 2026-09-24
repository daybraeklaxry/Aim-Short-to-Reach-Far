"""Incremental offline mechanism extraction with explicit FP32 scoring-goal coordinates."""
from pathlib import Path
from collections import defaultdict
import ast,csv,datetime,json,os
import numpy as np
ROOT=Path('runtime/ap/evidence_strengthening_20260923/results_v2');BASE=ROOT.parent.parent/'fresh_target_interface_study_v2'
OUT=ROOT/'mechanism_all';OUT.mkdir(exist_ok=True)
P=json.loads((BASE/'protocol.json').read_text());manifest=json.loads((ROOT/'queue-manifest.json').read_text());jobdefs={j['id']:j for j in manifest['jobs']}
# Reuse tested geometry functions without importing/rerunning the original full analysis.
module=ast.parse((ROOT/'code/analyze_physical.py').read_text(encoding='utf-8-sig'))
funcs=[n for n in module.body if isinstance(n,ast.FunctionDef)]
namespace=dict(np=np,object_adr=14)
exec(compile(ast.Module(body=funcs,type_ignores=[]),'mechanism_functions','exec'),namespace)
coords,physical_error,stalling=[namespace[k] for k in ['coords','physical_error','stalling']]
spec=json.loads((ROOT/'mechanism/specification.json').read_text());spec.update(goal_coordinates='Scoring singleton FP32 target, never the cached retrieval target. Original-H fallback uses the original CEM(final) scoring target for the identical query/start. Shorter-H fallback requires a matching final-goal CEM trace.',current_coordinates='Per-decision scoring latent retained by the actual controller.',state_mapping='Cube integration qpos offset 1, object joint qpos offset 14, checked against MuJoCo in original analysis.')
(OUT/'specification.json').write_text(json.dumps(spec,indent=2))
def lines(file):
 if file.exists():
  for line in file.open():
   if line.endswith('\n'):yield json.loads(line)
def key(r):return (r['task'],int(r['horizon']),int(r.get('query_ordinal',r['identity']['record_id'].split('-')[-1])),r['condition'])
def group(kind,task,cfg):return json.dumps(dict(kind=kind,task=task,config={k:v for k,v in cfg.items() if k not in ['start','stop','conditions']}),sort_keys=True)

expected_groups={group(j['kind'],j['task'],j['config']) for j in manifest['jobs'] if j['kind'] in ['main','budget','transport','ablation','horizon']}
for task,study in P['studies'].items():
 H=study['horizon'];expected_groups.add(group('main',task,dict(arm='lewm_25',H=H)))
 for iterations in [1,2,5,10]:
  for arm in ['cem_final','cem_observed']:
   expected_groups.add(group('budget',task,dict(arm=arm,iterations=iterations,H=H)))
entries=[]
for file in (ROOT/'lewm').glob('*/*/outcomes.jsonl'):
 for r in lines(file):entries.append((group('main',r['task'],dict(arm='lewm_25',H=r['horizon'])),r,file.parent/'traces'/f"{r['identity']['record_id']}_{r['condition']}"))
for job in manifest['jobs']:
 if job['kind'] not in ['main','budget','transport','ablation','horizon']:continue
 file=ROOT/'jobs'/job['id']/'episodes.jsonl'
 for r in lines(file):entries.append((group(job['kind'],job['task'],job['config']),r,file.parent/'traces'/f"{r['query_ordinal']:03}_{r['condition']}"))
for file in (ROOT.parent/'budget').glob('*/outcomes_budget*.jsonl'):
 for r in lines(file):
  arm='cem_'+r['arm'].split('_',1)[1];cfg=dict(arm=arm,iterations=r['iterations'],H=r['horizon'])
  entries.append((group('budget',r['task'],cfg),r,file.parent/'traces'/f"{r['identity']['record_id']}_{r['condition']}_{arm.split('_',1)[1]}_i{r['iterations']}"))
goals={};sidecars={}
for receipt in (ROOT/'goal_encodings').glob('*/summary.json'):
 d=json.loads(receipt.read_text())
 if d.get('complete'):
  with np.load(receipt.parent/'scoring_goals.npz') as z:
   sidecars.update({stem:z[key].copy().astype(float) for stem,key in d['trace_mapping'].items()})
# A completed future final-goal trace supplies any pre-update shorter-horizon goal.
for g,r,stem in entries:
 cfg=json.loads(g)['config']
 if cfg['arm']=='cem_final':
  with np.load(stem.with_suffix('.npz')) as z:
   if 'scoring_goal_encoding' in z:goals[key(r)]=z['scoring_goal_encoding'].reshape(-1).astype(float)
   elif 'score_target' in z and len(z['score_target']):goals[key(r)]=z['score_target'][0].astype(float)
rows=[];seen=set();missing=[];curves={}
for g,r,stem in entries:
 unique=(g,r['query_ordinal'] if 'query_ordinal' in r else int(r['identity']['record_id'].split('-')[-1]),r['condition']);assert unique not in seen,unique;seen.add(unique)
 task=r['task'];trace=json.loads(stem.with_suffix('.json').read_text());x=coords(task,trace['trajectory']);error=physical_error(task,x,trace['true_goal']);ts=[d['t'] for d in trace['decisions']]
 assert len(x)==r['primitive_steps']+1 and len(ts)==r['decisions'] and np.isfinite(error).all()
 entered=r['prelude']['category']=='policy_entered'
 if entered and r['success']:assert error[-1]<=1+1e-5,(unique,error[-1])
 row=dict(group=g,task=task,query_id=r['identity']['record_id'],query_ordinal=unique[1],condition=r['condition'],success=bool(r['success']),entered_policy=entered,decisions=r['decisions'],primitive_steps=r['primitive_steps'],initial_error=float(error[0]),terminal_error=float(error[-1]),max_physical_rise=float((error-np.minimum.accumulate(error)).max()),trace=str(stem))
 for w in [5,10,20]:
  start=ts[-min(w,len(ts))] if ts else 0
  row[f'stall_W{w}']=stalling(task,x[start:]) if not r['success'] and ts else None
  row[f'full_window_W{w}']=len(ts)>=w
 for delta in [.25,.5,1.0]:row[f'physical_detour_{delta}']=row['max_physical_rise']>=delta if entered and r['success'] else None
 goal=None;latent=None
 with np.load(stem.with_suffix('.npz')) as z:
  if 'scoring_goal_encoding' in z:goal=z['scoring_goal_encoding'].reshape(-1).astype(float)
  elif int(r['horizon'])==P['studies'][task]['horizon']:
   path=BASE/'evaluation/traces'/task/f"confirm_{r['identity']['record_id']}_{r['condition']}_gaussian_final.npz"
   with np.load(path) as old:
    if 'score_target' in old and len(old['score_target']):goal=old['score_target'][0].astype(float)
  elif key(r) in goals:goal=goals[key(r)]
  elif str(stem) in sidecars:goal=sidecars[str(stem)]
  if 'score_current' in z and len(ts):
   if goal is None:missing.append(dict(trace=str(stem),reason='matching FP32 scoring-goal encoding not available yet'))
   else:
    distance=np.square(z['score_current'].reshape(-1,192).astype(float)-goal).sum(-1);assert len(distance)==len(ts)
    if distance[0]>1e-12:latent=distance/distance[0]
 row['max_latent_rise']=float((latent-np.minimum.accumulate(latent)).max()) if latent is not None else None
 row['latent_detour']=row['max_latent_rise']>=.1 if entered and r['success'] and latent is not None else None
 rows.append(row)
 if task=='pusht' and r['condition']=='clean':curves[str(unique)]=dict(group=g,query_id=r['identity']['record_id'],success=r['success'],t=ts,error=error[np.asarray(ts,int)].tolist(),terminal_t=r['primitive_steps'],terminal_error=float(error[-1]))
summary=[]
by_group=defaultdict(list)
for row in rows:by_group[row['group']].append(row)
assert set(by_group)<=expected_groups,sorted(set(by_group)-expected_groups)
for g in sorted(expected_groups):
 for label,conditions in [('Standard',['clean']),('Perturbed',['prefix_a','prefix_b']),('Perturbed 1',['prefix_a']),('Perturbed 2',['prefix_b'])]:
  selected=[r for r in by_group[g] if r['condition'] in conditions];full={(r['query_ordinal'],r['condition']) for r in selected}=={(q,c) for q in range(128) for c in conditions}
  cell=dict(group=json.loads(g),start=label,complete=full,outcomes=len(selected),expected=128*len(conditions))
  for metric in ['stall_W5','stall_W10','stall_W20','physical_detour_0.25','physical_detour_0.5','physical_detour_1.0','latent_detour']:
   eligible=[r for r in selected if r[metric] is not None];count=sum(bool(r[metric]) for r in eligible)
   cell[metric]=dict(numerator=count,denominator=len(eligible),percent=100*count/len(eligible) if full and eligible else None)
  summary.append(cell)
def save(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,path)
with (OUT/'episodes.csv').open('w',newline='') as f:
 if rows:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
save(OUT/'summary.json',dict(collected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),episodes=len(rows),expected_groups=len(expected_groups),expected_cells=4*len(expected_groups),groups_without_outcomes=sum(not by_group[g] for g in expected_groups),complete=all(r['complete'] for r in summary) and not missing,rows=summary,missing_goal_encodings=missing,original_main_analysis=str(ROOT/'mechanism/summary.json')))
save(OUT/'curves.json',curves)
print(json.dumps(dict(episodes=len(rows),cells=len(summary),complete_cells=sum(r['complete'] for r in summary),groups_without_outcomes=sum(not by_group[g] for g in expected_groups),missing_goal_encodings=len(missing))))
