"""Offline MAIN mechanism analysis using existing physical and latent traces."""
from pathlib import Path
import csv,json,time
import numpy as np

BASE=Path('runtime/ap/fresh_target_interface_study_v2')
ROOT=BASE.parent/'evidence_strengthening_20260923/results_v2'
OUT=ROOT/'mechanism';OUT.mkdir(parents=True,exist_ok=True)
P=json.loads((BASE/'protocol.json').read_text())
SPEC=dict(windows=[5,10,20],deltas=[.25,.5,1.0],primary_window=10,primary_delta=.5,latent_rise_threshold=.1,physical_thresholds=dict(cube=.04,pusht_position=20,pusht_angle=float(np.pi/9),reacher=.05,tworoom=16),success_boundaries='Use the original logged success flags. Cube includes equality at its threshold; do not reinterpret every task with a universal strict inequality.',stall='Failed episodes: maximum physical displacement from the state at the start of the last W decisions through termination is below one quarter of each success threshold. Fewer than W decisions use the available window and are counted separately.',physical_detour='Successful episodes: physical normalized error rises at least delta above its previous running minimum before reaching success.',latent_detour='Successful episodes: a logged decision squared goal distance rises by at least 10% of the initial decision distance above its previous running minimum. Only logged pre-action states are used.',interpretation='Descriptive behavior; stalling does not prove a local/global optimum or uniquely identify the cause of control failure.',aggregation='For Perturbed, average the two start-specific indicator/numerator/denominator values within each query before task means; report ratios of these means. Also report starts separately.',selection='Illustrations: first three query_ids in sorted order satisfying the specified success/failure condition; no visual selection.')
(OUT/'specification.json').write_text(json.dumps(SPEC,indent=2))
# The integration vector stores time followed by qpos. Confirm the object slice against MuJoCo on an actual environment before decoding saved vectors.
import tokenizers
from experiments.delta_jepa import sota_generic_retrieval as generic
import mujoco
cube_env=generic._make_env('cube');cube_env.reset(seed=42,options={'variation':[]})
model=cube_env.unwrapped.model;data=cube_env.unwrapped.data
kind=mujoco.mjtState.mjSTATE_INTEGRATION
v=np.empty(mujoco.mj_stateSize(model,kind));mujoco.mj_getState(model,data,v,kind)
np.testing.assert_array_equal(v[1:1+model.nq],data.qpos)
object_adr=int(model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'object_joint_0')])
print(json.dumps(dict(cube_qpos_offset=1,cube_object_qpos_index=object_adr,integration_layout_checked=True)),flush=True)
cube_env.close()

def coords(task,trajectory):
 if task=='cube':return np.asarray([s['integration'][1+object_adr:1+object_adr+3] for s in trajectory],float)
 if task=='reacher':return np.asarray([s['physics'][:2] for s in trajectory],float)
 if task=='pusht':return np.asarray([s['state'][:5] for s in trajectory],float)
 return np.asarray([s['state'][:2] for s in trajectory],float)

def angle(a):return (a+np.pi)%(2*np.pi)-np.pi

def physical_error(task,x,g):
 g=np.asarray(g,float)
 if task=='cube':return np.linalg.norm(x-(g[19:22]/10+np.asarray([.425,0,0])),axis=-1)/.04
 if task=='pusht':return np.maximum(np.linalg.norm(x[:,:4]-g[:4],axis=-1)/20,np.abs(angle(x[:,4]-g[4]))/(np.pi/9))
 if task=='reacher':return np.max(np.abs(x-g[:2]),axis=-1)/.05
 return np.linalg.norm(x-g[:2],axis=-1)/16

def stalling(task,x):
 d=x-x[0]
 if task=='cube':return bool(np.linalg.norm(d,axis=-1).max()<.01)
 if task=='pusht':return bool(np.linalg.norm(d[:,2:4],axis=-1).max()<5 and np.abs(angle(d[:,4])).max()<np.pi/36)
 if task=='reacher':return bool(np.abs(d).max()<.0125)
 return bool(np.linalg.norm(d,axis=-1).max()<4)

rows=[];curves={};started=time.perf_counter()
for task in P['studies']:
 for query in P['studies'][task]['confirm']:
  qid=query['record_id']
  for condition in P['conditions']:
   prefix=BASE/'evaluation/traces'/task/f'confirm_{qid}_{condition}'
   goal=None
   with np.load(Path(str(prefix)+'_gaussian_final.npz')) as z:
    if 'score_target' in z:goal=z['score_target'][0].astype(float)
   for arm in ['gaussian_final','gaussian_observed','ap_final','ap_observed','direct']:
    stem=Path(str(prefix)+'_'+arm);trace=json.loads(stem.with_suffix('.json').read_text());c=trace['compact']
    x=coords(task,trace['trajectory']);err=physical_error(task,x,trace['true_goal']);assert len(x)==c['primitive_steps']+1 and np.isfinite(err).all()
    # Compare only policy-entered terminal states; initially solved/prelude outcomes retain their source labels.
    entered=c['prelude']['category']=='policy_entered'
    if entered and c['success']:
     assert err[-1]<=1+1e-5,(task,qid,condition,arm,float(err[-1]))
    decision_steps=[d['t'] for d in trace['decisions']]
    base=dict(task=task,query_id=qid,condition=condition,arm=arm,success=bool(c['success']),entered_policy=entered,decisions=c['decisions'],primitive_steps=c['primitive_steps'],initial_error=float(err[0]),terminal_error=float(err[-1]),max_physical_rise=float((err-np.minimum.accumulate(err)).max()))
    for w in SPEC['windows']:
     begin=decision_steps[-min(w,len(decision_steps))] if decision_steps else 0
     base[f'stall_W{w}']=stalling(task,x[begin:]) if not c['success'] and decision_steps else None
     base[f'full_window_W{w}']=len(decision_steps)>=w
    for delta in SPEC['deltas']:base[f'physical_detour_{delta}']=base['max_physical_rise']>=delta if c['success'] and entered else None
    latent=None
    with np.load(stem.with_suffix('.npz')) as z:
     if 'score_current' in z and goal is not None:
      distance=np.square(z['score_current'].astype(float)-goal).sum(-1)
      assert len(distance)==len(decision_steps)
      if len(distance) and distance[0]>1e-12:latent=distance/distance[0]
    base['max_latent_rise']=float((latent-np.minimum.accumulate(latent)).max()) if latent is not None else None
    base['latent_detour']=base['max_latent_rise']>=.1 if c['success'] and entered and latent is not None else None
    rows.append(base)
    if task=='pusht' and condition=='clean' and arm in ['gaussian_final','gaussian_observed']:
     curves[f'{qid}/{arm}']=dict(query_id=qid,arm=arm,success=bool(c['success']),t=decision_steps,error=err[np.asarray(decision_steps,dtype=int)].tolist(),terminal_t=c['primitive_steps'],terminal_error=float(err[-1]))
 print(json.dumps(dict(task=task,processed=len(rows),elapsed=round(time.perf_counter()-started,1))),flush=True)
assert len(rows)==7680
with (OUT/'episodes.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'curves.json').write_text(json.dumps(curves))
summary=[]
for task in P['studies']:
 for arm in ['gaussian_final','gaussian_observed','ap_final','ap_observed','direct']:
  for group,conditions in [('Standard',['clean']),('Perturbed',['prefix_a','prefix_b']),('Perturbed 1',['prefix_a']),('Perturbed 2',['prefix_b'])]:
   sub=[r for r in rows if r['task']==task and r['arm']==arm and r['condition'] in conditions]
   result=dict(task=task,arm=arm,start=group,query_count=128,starts=len(sub),success_percent=100*sum(r['success'] for r in sub)/len(sub))
   for metric in [f'stall_W{w}' for w in SPEC['windows']]+[f'physical_detour_{d}' for d in SPEC['deltas']]+['latent_detour']:
    eligible=[r for r in sub if r[metric] is not None]
    result[metric]=dict(numerator=sum(bool(r[metric]) for r in eligible),denominator=len(eligible),percent=100*sum(bool(r[metric]) for r in eligible)/len(eligible) if eligible else None)
   summary.append(result)
(OUT/'summary.json').write_text(json.dumps(dict(complete=True,episodes=len(rows),specification=SPEC,rows=summary),indent=2))
print(json.dumps(dict(complete=True,episodes=len(rows),output=str(OUT))),flush=True)
