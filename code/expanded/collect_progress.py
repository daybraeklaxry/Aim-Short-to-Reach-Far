"""Read-only collection from canonical episode files; partial cells stay marked incomplete."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,datetime,json,os
ROOT=Path('runtime/ap/evidence_strengthening_20260923/results_v2')
BASE=ROOT.parent.parent/'fresh_target_interface_study_v2'
ARMS={'gaussian_final':'CEM (final goal)','gaussian_observed':'AP-CEM','direct':'Direct','ap_final':'Rank (final goal)','ap_observed':'AP-rank','cem_learned':'CEM (learned target)','rank_learned':'Rank (learned target)','lewm_25':'LeWM planner'}
def save(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,path)
def rows(path):
 with path.open() as stream:
  for n,line in enumerate(stream,1):
   if not line.endswith('\n'):
    # A writer may currently be flushing this final row. Keep the file incomplete until the next snapshot.
    break
   yield json.loads(line)
manifest=json.loads((ROOT/'queue-manifest.json').read_text());jobs={j['id']:j for j in manifest['jobs']}
seen={};origins={};duplicate=[]
def add(r,source,arm=None):
 task=r['task'];key=(task,r.get('query_ordinal',r['identity'].get('cohort_index',int(r['identity']['record_id'].split('-')[-1]))),r['condition'],arm or ARMS[r['arm']])
 if key in seen:
  duplicate.append(dict(key=key,first=origins[key],second=str(source)));return
 seen[key]=r;origins[key]=str(source)
for file in sorted((BASE/'evaluation').glob('outcomes_*_shard*.jsonl')):
 for r in rows(file):assert r['completed'];add(r,file)
assert len(seen)==7680
for file in sorted((ROOT/'lewm').glob('*/*/outcomes.jsonl')):
 for r in rows(file):assert r['completed'];add(r,file)
for job in jobs.values():
 if job['kind']!='main':continue
 file=ROOT/'jobs'/job['id']/'episodes.jsonl'
 if file.exists():
  for r in rows(file):assert r['completed'];add(r,file)
assert not duplicate,duplicate[:2]
cells=[]
for task in ['cube','pusht','reacher','tworoom']:
 for arm in ARMS.values():
  for label,conditions in [('Standard',['clean']),('Perturbed',['prefix_a','prefix_b']),('Perturbed 1',['prefix_a']),('Perturbed 2',['prefix_b'])]:
   selected=[(key,r) for key,r in seen.items() if key[0]==task and key[3]==arm and key[2] in conditions]
   full=len(selected)==128*len(conditions)
   if full:
    assert {(k[1],k[2]) for k,r in selected}=={(q,c) for q in range(128) for c in conditions}
   cells.append(dict(task=task,arm=arm,start=label,complete=full,completed_outcomes=len(selected),expected_outcomes=128*len(conditions),success_percent=100*sum(r['success'] for k,r in selected)/len(selected) if full else None,mean_pred_blocks=sum(r['predictor_candidate_blocks'] for k,r in selected)/len(selected) if full else None))
main=dict(collected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),complete=all(c['complete'] for c in cells),duplicate_keys=duplicate,rows=cells,observed_main_outcomes=len(seen),expected_main_outcomes=4*128*3*8,partial_success_rates_withheld=True,aggregation='Within each query average two perturbation starts, then query means within task; four-task means equally weighted.')
save(ROOT/'collected/main.json',main)
with (ROOT/'collected/main.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(cells[0]));w.writeheader();w.writerows(cells)
if main['complete']:
 target=ROOT/'collected/main_episodes.csv';temporary=target.with_suffix('.csv.tmp')
 with temporary.open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['task','controller','query_ordinal','condition','success','primitive_steps','prediction_blocks','source_file'])
  for key,r in sorted(seen.items()):
   task,q,condition,arm=key
   w.writerow([task,arm,q,condition,int(r['success']),r['primitive_steps'],r['predictor_candidate_blocks'],origins[key]])
 os.replace(temporary,target)
state=json.loads((ROOT/'queue-state.json').read_text());status=dict(Counter(s['status'] for s in state['jobs'].values()))
failures=[dict(job=k,error=s.get('error_tail'),attempts=s.get('attempts')) for k,s in state['jobs'].items() if s['status']=='failed']
progress={'main_cells_complete':sum(c['complete'] for c in cells),'main_cells_total':len(cells),'main_outcomes':len(seen),'queue':status,'failures':failures}
save(ROOT/'collected/progress.json',progress)
from progress_report import write_progress
write_progress(ROOT,jobs,state)
from summarize_evidence import render_report
render_report()
print(json.dumps(progress))
