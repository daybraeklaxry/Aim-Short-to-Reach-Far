"""Collect complete cells for every registered intervention and explicit MAIN aliases."""
from pathlib import Path
from collections import defaultdict
import csv,datetime,json,os
ROOT=Path('runtime/ap/evidence_strengthening_20260923/results_v2');BASE=ROOT.parent.parent/'fresh_target_interface_study_v2'
TASKS=('cube','pusht','reacher','tworoom');CONDITIONS=('clean','prefix_a','prefix_b')
p=json.loads((BASE/'protocol.json').read_text());manifest=json.loads((ROOT/'queue-manifest.json').read_text())
def loadlines(path):
 if not path.exists():return
 for line in path.open():
  if line.endswith('\n'):yield json.loads(line)
def write(path,x):
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2));os.replace(temp,path)
def signature(kind,task,cfg):
 return json.dumps(dict(kind=kind,task=task,config={k:v for k,v in cfg.items() if k not in ['start','stop','conditions']}),sort_keys=True)
cells=defaultdict(dict);sources=defaultdict(set);definitions={};alias_receipts=[]
def add(sig,row,path):
 key=(row.get('query_ordinal',int(row['identity']['record_id'].split('-')[-1])),row['condition'])
 assert key not in cells[sig],(sig,key,str(path))
 assert row['completed']
 cells[sig][key]=row;sources[sig].add(str(path))
main={}
for f in (BASE/'evaluation').glob('outcomes_*_shard*.jsonl'):
 for r in loadlines(f):main[(r['task'],r['arm'],int(r['identity']['record_id'].split('-')[-1]),r['condition'])]=r
assert len(main)==7680
for j in manifest['jobs']:
 if j['kind'] not in ['transport','budget','ablation','horizon','main']:continue
 sig=signature(j['kind'],j['task'],j['config']);definitions[sig]=json.loads(sig)
 f=ROOT/'jobs'/j['id']/'episodes.jsonl'
 for r in loadlines(f):add(sig,r,f)
# Adopt the live original budget jobs, keyed by actual iteration and target.
for task in TASKS:
 for f in (ROOT.parent/'budget'/task).glob('outcomes_budget*.jsonl'):
  for r in loadlines(f):
   cfg=dict(arm={'gaussian_final':'cem_final','gaussian_observed':'cem_observed'}[r['arm']],iterations=r['iterations'],H=p['studies'][task]['horizon'])
   sig=signature('budget',task,cfg);definitions[sig]=json.loads(sig);add(sig,r,f)
# Every default point is a reference to the original episode object, not independently recomputed.
for task in TASKS:
 H=p['studies'][task]['horizon']
 for arm,oldarm in [('cem_final','gaussian_final'),('cem_observed','gaussian_observed'),('rank_final','ap_final'),('rank_observed','ap_observed')]:
  targets=[('horizon',dict(arm=arm,iterations=30,H=H,B=p['studies'][task]['budget']))]
  if arm.startswith('cem_'):targets.append(('budget',dict(arm=arm,iterations=30,H=H)))
  if arm in ['cem_observed','rank_observed']:targets.append(('ablation',dict(arm=arm,iterations=30,H=H,default=True)))
  for kind,cfg in targets:
   sig=signature(kind,task,cfg);definitions[sig]=json.loads(sig)
   for q in range(128):
    for condition in CONDITIONS:add(sig,main[(task,oldarm,q,condition)],BASE/'evaluation')
   alias_receipts.append(dict(destination=json.loads(sig),source_arm=oldarm,episodes=384,same_episode_records=True))
 # Transport's anchored control is exactly the corresponding original AP-CEM episode.
 sig=signature('transport',task,dict(arm='cem_observed',iterations=30,H=H));definitions[sig]=json.loads(sig)
 for q in range(128):
  for condition in CONDITIONS:add(sig,main[(task,'gaussian_observed',q,condition)],BASE/'evaluation')
 alias_receipts.append(dict(destination=json.loads(sig),source_arm='gaussian_observed',episodes=384,same_episode_records=True))
 # Learned 30-iteration and H-default aliases come from the same new MAIN episodes.
 source=signature('main',task,dict(arm='cem_learned',iterations=30,H=H))
 for arm in ['cem_learned','rank_learned']:
  source=signature('main',task,dict(arm=arm,iterations=30,H=H))
  for kind,cfg in [('horizon',dict(arm=arm,iterations=30,H=H,B=p['studies'][task]['budget']))]+([('budget',dict(arm=arm,iterations=30,H=H))] if arm=='cem_learned' else []):
   sig=signature(kind,task,cfg);definitions[sig]=json.loads(sig)
   for _,r in list(cells[source].items()):add(sig,r,'MAIN learned same-record alias')
 # LeWM original-H point uses its matching MAIN planner rows.
 sig=signature('horizon',task,dict(arm='lewm',iterations=30,H=H,B=p['studies'][task]['budget']));definitions[sig]=json.loads(sig)
 for f in (ROOT/'lewm'/task).glob('*/outcomes.jsonl'):
  for r in loadlines(f):add(sig,r,f)
results=[]
for sig,spec in sorted(definitions.items()):
 for label,conditions in [('Standard',['clean']),('Perturbed',['prefix_a','prefix_b']),('Perturbed 1',['prefix_a']),('Perturbed 2',['prefix_b'])]:
  expected={(q,c) for q in range(128) for c in conditions};available={k:r for k,r in cells[sig].items() if k[1] in conditions};assert set(available)<=expected
  full=set(available)==expected
  row=dict(**spec,start=label,complete=full,completed=len(available),expected=len(expected),success_percent=None,mean_pred_blocks=None,mean_steps=None,source_files=sorted(sources[sig]))
  if full:
   # Explicit per-query averaging, so paired starts remain together.
   averages=lambda key:sum(sum(float(available[(q,c)][key]) for c in conditions)/len(conditions) for q in range(128))/128
   row.update(success_percent=100*averages('success'),mean_pred_blocks=averages('predictor_candidate_blocks'),mean_steps=averages('primitive_steps'))
  results.append(row)
write(ROOT/'collected/interventions.json',dict(collected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=results,main_aliases=alias_receipts,complete=all(r['complete'] for r in results),complete_cells=sum(r['complete'] for r in results),total_cells=len(results)))
write(ROOT/'collected/reuse-equalities.json',dict(verified=True,checks=alias_receipts,criterion='Source episode dictionaries are reused directly; no reconstructed rounded percentages or independent default reruns.'))
if all(r['complete'] for r in results):
 target=ROOT/'collected/intervention_episodes.csv';temporary=target.with_suffix('.csv.tmp')
 with temporary.open('w',newline='') as stream:
  writer=csv.writer(stream)
  writer.writerow(['study','task','controller','configuration','query_ordinal','condition','success','primitive_steps','prediction_blocks'])
  for sig,spec in sorted(definitions.items()):
   cfg=json.dumps(spec['config'],sort_keys=True,separators=(',',':'))
   for (q,condition),row in sorted(cells[sig].items()):
    writer.writerow([spec['kind'],spec['task'],spec['config']['arm'],cfg,q,condition,int(row['success']),row['primitive_steps'],row['predictor_candidate_blocks']])
 os.replace(temporary,target)
print(json.dumps(dict(cells=len(results),complete=sum(r['complete'] for r in results),verified_aliases=len(alias_receipts))))
