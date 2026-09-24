"""Single coordinator for the fixed results_v2 queue. No other user's process is signalled."""
from pathlib import Path
import datetime,json,os,subprocess,time,traceback
ROOT=Path('runtime/ap/evidence_strengthening_20260923/results_v2')
CODE=ROOT/'code';STATE=ROOT/'queue-state.json';PY='runtime/environment/venv/bin/python'

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2));os.replace(tmp,p)
def alive(pid,marker):
 p=Path('/proc')/str(pid)
 try:return p.joinpath('stat').read_text().split()[2]!='Z' and marker in p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
 except FileNotFoundError:return False

def cmd(job):
 c=job['config'];task=job['task'];kind=job['kind']
 if kind=='train':
  cache=(ROOT.parent/'learned_target/cube' if task=='cube' else ROOT/'features'/task)/'native_fp32_memory.json'
  if not cache.exists():return None,'awaiting FP32 memory cache',None
  out=ROOT/'training'/task/f'w{c["width"]}_{"residual" if c["residual"] else "absolute"}'
  return ['train_grid.py','--task',task,'--mode','train','--width',str(c['width']),'--residual',str(c['residual'])],None,out/'summary.json'
 if kind=='long':
  return ['run_long_v2.py','--task',task,'--mode','evaluate','--start',str(c['start']),'--stop',str(c['stop']),'--output',str(ROOT/'lewm')],None,ROOT/'lewm'/task/f'q{c["start"]:03}-{c["stop"]:03}'/'summary.json'
 runner=CODE/'run_suite.py'
 if kind in ['main','budget','transport','ablation','horizon'] and runner.exists() and (CODE/'suite_ready.json').exists():
  if job.get('dependency')=='selected_target' and not (ROOT/'training'/task/'selected.json').exists():return None,'awaiting validation-only target selection',None
  return ['run_suite.py','--task',task,'--job-id',job['id']],None,ROOT/'jobs'/job['id']/'summary.json'
 if kind=='exact' and (CODE/'run_exact_v2.py').exists():
  if not (ROOT/'training'/task/'selected.json').exists():return None,'awaiting validation-only target selection',None
  return ['run_exact_v2.py','--task',task,'--job-id',job['id']],None,ROOT/'jobs'/job['id']/'summary.json'
 return None,'implementation/preflight pending',None

def select_targets():
 for task in ['cube','pusht','reacher','tworoom']:
  out=ROOT/'training'/task
  if (out/'selected.json').exists():continue
  summaries=[out/f'w{w}_{v}'/'summary.json' for w in [512,1024] for v in ['absolute','residual']]
  if not all(f.exists() for f in summaries):continue
  rows=[read(f) for f in summaries]
  if not all(r.get('complete') for r in rows):continue
  best=min(rows,key=lambda r:r['validation_mse'])
  write(out/'selected.json',dict(selected_utc=stamp(),criterion='lowest raw-latent validation MSE on the same 4096 held-out-memory transitions',candidates=rows,selected=best,control_results_used=False))

manifest=read(ROOT/'queue-manifest.json');jobs=manifest['jobs']
state=read(STATE) if STATE.exists() else dict(started_utc=stamp(),jobs={})
for j in jobs:state['jobs'].setdefault(j['id'],dict(status='queued',attempts=0))
(ROOT/'logs').mkdir(exist_ok=True);children={};last_report=0
print(json.dumps(dict(event='scheduler_started',pid=os.getpid(),registered=len(jobs),utc=stamp())),flush=True)
while True:
 try:
  select_targets()
  for job in jobs:
   s=state['jobs'][job['id']]
   if s['status']!='running':continue
   args,reason,summary=cmd(job)
   if alive(s['pid'],str(CODE)) or alive(s['pid'],str(ROOT/'frozen_workers')):continue
   if summary and summary.exists() and (read(summary).get('complete') or read(summary).get('completed')):
    s.update(status='complete',completed_utc=stamp(),summary=str(summary))
   else:
    log=Path(s['log']);tail=log.read_bytes()[-5000:].decode(errors='replace')
    oom='out of memory' in tail.lower();s.update(status='queued' if oom and s['attempts']<3 else 'failed',terminal_utc=stamp(),error_tail=tail[-2400:],oom=oom)
    if oom:state.setdefault('gpu_worker_caps',{})[str(s['gpu'])]=1
   if job['id'] in children:children.pop(job['id']).poll()
  output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  resources={int(x.split(',')[0]):[int(v) for v in x.split(',')[1:]] for x in output.splitlines()}
  counts={g:0 for g in resources}
  # Adopt the live jobs from the earlier phase; never restart on an observation timeout.
  for f in list(ROOT.parent.glob('*launch.json'))+list(ROOT.glob('*-launch.json')):
   d=read(f);workers=d.get('workers',[d])
   for w in workers:
    if 'pid' in w and 'gpu' in w and alive(w['pid'],str(ROOT.parent)):counts[w['gpu']]+=1
  for s in state['jobs'].values():
   if s['status']=='running':counts[s['gpu']]+=1
  for job in sorted(jobs,key=lambda j:(j['priority'],0 if j['kind']=='train' else 1,j['config'].get('start',0),j['id'])):
   s=state['jobs'][job['id']]
   if s['status']!='queued':continue
   args,reason,summary=cmd(job)
   if args is None:s['waiting_for']=reason;continue
   if summary.exists() and (read(summary).get('complete') or read(summary).get('completed')):
    s.update(status='complete',summary=str(summary),adopted=True);continue
   # Initial conservative single-job limits; completed worker receipts supply later measured peaks.
   demand=12000 if job['kind']=='train' else 9000 if job['kind']=='long' else 75000
   available=[g for g,(free,util) in resources.items() if demand<.85*free and counts[g]<min(2 if util>90 else 4,state.get('gpu_worker_caps',{}).get(str(g),4))]
   if not available:s['waiting_for']='GPU memory or active-worker limit';continue
   gpu=max(available,key=lambda g:(resources[g][0]-demand,-counts[g]));s['attempts']+=1
   log=ROOT/'logs'/f'{job["id"]}-attempt{s["attempts"]}.log'
   command=['bash',str(CODE/'run.sh'),*args]
   with log.open('xb') as handle:
    proc=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
   children[job['id']]=proc;s.update(status='running',pid=proc.pid,gpu=gpu,log=str(log),command=command,started_utc=stamp(),free_before_mib=resources[gpu][0]);s.pop('waiting_for',None)
   counts[gpu]+=1;resources[gpu][0]-=demand
   print(json.dumps(dict(event='launched',job=job['id'],task=job['task'],gpu=gpu,pid=proc.pid,utc=stamp())),flush=True)
  state['updated_utc']=stamp();write(STATE,state)
  if time.time()-last_report>=1800:
   statuses={v:sum(s['status']==v for s in state['jobs'].values()) for v in ['queued','running','complete','failed']}
   report='# Experiment progress\n\nUpdated UTC: '+stamp()+'\n\n'+json.dumps(statuses,indent=2)+'\n\nGPU free MiB / utilization: '+json.dumps(resources)+'\n\nEach pending job retains its explicit waiting reason in queue-state.json. Query results and running jobs are never discarded on a polling timeout. Total ETA is not yet reliable while remaining controller implementations and runtime profiles are being completed.\n'
   (ROOT/'PROGRESS.md').write_text(report);last_report=time.time()
  time.sleep(30)
 except Exception:
  (ROOT/'scheduler-last-error.txt').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True);time.sleep(30)
