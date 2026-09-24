"""Freeze the full requested A/B suite once, before new control results are read."""
from pathlib import Path
import datetime,json
ROOT=Path('runtime/ap/evidence_strengthening_20260923/results_v2')
BASE=ROOT.parent.parent/'fresh_target_interface_study_v2'
p=json.loads((BASE/'protocol.json').read_text())
jobs=[];reused=[]
def add(kind,task,priority,config=None,dependency=None):
 c=config or {};jobid=kind+'-'+task+'-'+str(len(jobs)).zfill(4)
 jobs.append(dict(id=jobid,kind=kind,task=task,priority=priority,config=c,dependency=dependency))
def eval_jobs(kind,task,priority,config=None,dependency=None):
 for start in range(0,128,16):add(kind,task,priority,dict(config or {},start=start,stop=min(start+16,128),conditions=p['conditions']),dependency)
for task in p['studies']:
 H=p['studies'][task]['horizon']
 for width in [512,1024]:
  for residual in [0,1]:add('train',task,1,dict(width=width,residual=residual),'fp32_cache')
 for arm in ['cem_learned','rank_learned']:eval_jobs('main',task,1,dict(arm=arm,iterations=30,H=H),'selected_target')
 # q0 preflight for Cube is already live; its actual results are retained and reused.
 for start in range(0,128,16):
  begin=1 if task=='cube' and start==0 else start
  add('long',task,1,dict(start=begin,stop=min(start+16,128),conditions=p['conditions'],H=H),'long_profile')
 eval_jobs('exact',task,1,dict(selectors=['Direct','Predictor-Final','Predictor-Observed','Predictor-Learned','Simulator-Final','Simulator-Observed','Simulator-Learned','Displacement-Observed']),'selected_target')
 eval_jobs('transport',task,2,dict(arm='cem_transport',iterations=30,H=H))
 for I in [1,2,5,10]:eval_jobs('budget',task,2,dict(arm='cem_learned',iterations=I,H=H),'selected_target')
 # Final/observed 1,2,5,10 are adopted from the four existing live jobs.
 for key,vals,arms in [('target_L',[1,3,10],['rank_observed','cem_observed']),('memory_fraction',[.1,.25,.5],['rank_observed','cem_observed']),('key',['no_far','no_delta'],['rank_observed']),('span',['fixed'],['rank_observed'])]:
  for value in vals:
   for arm in arms:eval_jobs('ablation',task,2,dict(arm=arm,iterations=30,H=H,**{key:value}))
 for h in ([25,50] if task=='tworoom' else [25,50,100]):
  for arm in ['rank_final','rank_observed','rank_learned','cem_final','cem_observed','cem_learned','lewm']:
   eval_jobs('horizon',task,3,dict(arm=arm,iterations=30,H=h,B=int(p['studies'][task]['budget']*h/H)),'selected_target' if 'learned' in arm else None)
 reused.append(dict(task=task,source=str(BASE/'evaluation'),main_arms=5,main_episode_outcomes=1920,default_L=5,default_memory_fraction=1.0,default_H=H,default_iterations=30,budget_running=str(ROOT.parent/'budget'/task)))
manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),deadline_beijing='2026-09-26T12:00:00+08:00',query_count_per_task=128,conditions=p['conditions'],query_seed=26091391,prefix_seeds=[26091392,26091393],memory_subset_seed=26091600,protocol=str(BASE/'protocol.json'),no_new_query_sampling=True,no_uncertainty_plots=True,no_significance_tests=True,tiers='A and B only; no DINO-WM, SAGE protocol reproduction, or GCBC',jobs=jobs,reused=reused,additional_tasks=['memory_validation_target_quality_and_nearest_memory_distance','all_logged_controllers_physical_and_latent_mechanism_W5_10_20_delta025_05_10','selection_of_learned_target_only_after_all_four_memory_validation_configs_complete','tables_v2/main.tex','figures_v2/teaser.pdf','figures_v2/mechanism.pdf','tables_v2/mechanism.tex','figures_v2/compute.pdf','figures_v2/horizon.pdf','tables_v2/ablation.tex','tables_v2/diagnostic.tex','tables_v2/first_block.tex','tables_v2/transport.tex','tables_v2/appendix_full.tex','REPORT.md','revised_nine_page_main_PDF_and_source'])
path=ROOT/'queue-manifest.json'
assert not path.exists(),path
path.write_text(json.dumps(manifest,indent=2))
count={k:sum(j['kind']==k for j in jobs) for k in sorted({j['kind'] for j in jobs})}
print(json.dumps(dict(jobs=len(jobs),by_kind=count,new_episode_assignments=sum((j['config']['stop']-j['config']['start'])*3 for j in jobs if 'start' in j['config']),output=str(path))))
