"""Released LeWM policy on the existing AP queries; no training or score-based tuning."""
from pathlib import Path
from types import SimpleNamespace
import argparse, contextlib, copy, importlib.metadata, io, json, os, sys, time, warnings
import numpy as np
import torch
from gymnasium.spaces import Box
from stable_worldmodel.policy import WorldModelPolicy, PlanConfig
from stable_worldmodel.solver.cem import CEMSolver

OLD=Path('runtime/ap/evidence_strengthening_20260923/results_v2')
sys.path.insert(0,str(OLD/'code'))
import common_rollout as runtime
import run_suite
from observed_target import query_seed
BASE=run_suite.BASE
OUT=Path(__file__).resolve().parent/'official_results'

def raw_info(native,rgb,goal):
 return dict(pixels=np.asarray(rgb)[None,None],goal=np.asarray(goal)[None,None],action=np.zeros((1,1,native.action_dim),np.float32))

class ProjectedCost:
 def __init__(self,model,native,low,high):
  self.model=model
  bounds=native.scaler.transform(np.stack([low,high])).astype(np.float32)
  self.low=torch.tensor(np.tile(bounds[0],5),device=native.device)
  self.high=torch.tensor(np.tile(bounds[1],5),device=native.device)
 def get_cost(self,info,actions):
  effective=torch.minimum(torch.maximum(actions,self.low),self.high)
  return self.model.get_cost(info,effective)

def policy_for(native,env,seed,project=False):
 model=ProjectedCost(native.model,native,env.action_space.low,env.action_space.high) if project else native.model
 solver=CEMSolver(model=model,batch_size=1,num_samples=300,var_scale=1.0,n_steps=30,topk=30,device=str(native.device),seed=seed)
 # Transfer the unchanged transformed image before CEM expands its sample axis.
 # Moving the expanded CPU view instead copies each identical image 300 times.
 def prepared_image(x):return native.transform(x).to(native.device)
 policy=WorldModelPolicy(solver=solver,config=PlanConfig(horizon=5,receding_horizon=5,history_len=1,action_block=5,warm_start=True),process={'action':native.scaler},transform={'pixels':prepared_image,'goal':prepared_image})
 policy.set_env(SimpleNamespace(num_envs=1,action_space=Box(low=env.action_space.low[None],high=env.action_space.high[None],dtype=np.float32)))
 return policy

def released_block(policy,info):
 """Read the released policy's fixed action buffer without re-encoding unused images."""
 actions=[policy.get_action(info)[0]]
 while policy._action_buffer:
  action=policy._action_buffer.popleft().reshape(*policy.env.action_space.shape).numpy()
  actions.append(policy.process['action'].inverse_transform(action)[0])
 return np.stack(actions).astype(np.float32)

@torch.inference_mode()
def check_policy(native,env,cached,seed):
 native.begin({},dict(initial_rgb=cached['post_rgb'],goal_rgb=cached['goal_rgb'],goal=cached['goal']))
 native.execute_steps=25
 native.solver.torch_gen.manual_seed(seed)
 with contextlib.redirect_stdout(io.StringIO()):raw,_,_=native.choose(None,0)
 policy=policy_for(native,env,seed)
 policy.transform={'pixels':native.transform,'goal':native.transform}
 with contextlib.redirect_stdout(io.StringIO()):official=np.stack([policy.get_action(raw_info(native,cached['post_rgb'],cached['goal_rgb']))[0] for _ in range(25)])
 fast_policy=policy_for(native,env,seed)
 with contextlib.redirect_stdout(io.StringIO()):buffered=released_block(fast_policy,raw_info(native,cached['post_rgb'],cached['goal_rgb']))
 report={'released_policy_full_plan_max_abs_error':float(np.abs(raw-official).max()),'released_policy_full_plan_equal':bool(np.array_equal(raw,official)),'planned_primitive_steps':25,'receding_primitive_steps':25}
 report['buffer_read_max_abs_error']=float(np.abs(buffered-official).max())
 report['device_transfer_and_buffer_equal']=bool(np.array_equal(buffered,official))
 assert report['released_policy_full_plan_max_abs_error']<1e-6,report
 assert report['buffer_read_max_abs_error']==0,report
 return report

@torch.inference_mode()
def episode(a,H,ordinal,condition,project,p,row,source,api,ew,context,native,counter,cached_root):
 env,obs,info,true_goal,setup,cached=run_suite.LockedConstruction.construct(ew,api,context,native.scaler,a.task,row,'confirm',cached_root,p,source)
 seed=query_seed(a.task,ordinal);torch.manual_seed(seed)
 policy=policy_for(native,env,seed,project)
 mode='released_raw' if not project else 'released_box'
 actions=[];trajectory=[setup['post_state']];decisions=[];encoded=[]
 success=bool(setup['prelude']['success']);stopped=bool(setup['prelude']['terminal_without_success'])
 budget=int(p['studies'][a.task]['budget']);calls=blocks=0;start=time.perf_counter()
 goal_rgb=np.ascontiguousarray(cached['goal_rgb'])
 scoring_goal=native.encode_rgb(goal_rgb)[0].cpu().numpy()
 try:
  while not success and not stopped and len(actions)<budget:
   rgb=np.ascontiguousarray(env.render())
   encoded.append(native.encode_rgb(rgb)[0].cpu().numpy())
   counter.reset()
   # The released policy buffers all 25 actions. Reading that buffer now leaves
   # the same plan as consuming it one action at a time during execution.
   with contextlib.redirect_stdout(io.StringIO()):raw=released_block(policy,raw_info(native,rgb,goal_rgb))
   assert counter.calls==150 and counter.candidates==45000,(counter.calls,counter.candidates)
   calls+=counter.calls;blocks+=counter.candidates
   effective=np.clip(raw,env.action_space.low,env.action_space.high).astype(np.float32) if project else raw
   begin=len(actions)
   for action in effective:
    obs,_,term,trunc,info=env.step(action.copy());actions.append(action.copy());trajectory.append(ew.physical_state(a.task,env))
    success=bool(ew.runtime._success_now(a.task,obs,info,true_goal,False,False));stopped=bool((term or trunc) and not success)
    if success or stopped or len(actions)>=budget:break
   decisions.append({'step':begin,'executed_actions':len(actions)-begin,'planned_actions':25,'raw_out_of_box_components':int(((raw<env.action_space.low)|(raw>env.action_space.high)).sum())})
  result=dict(task=a.task,H=H,B=budget,query_ordinal=ordinal,condition=condition,mode=mode,record_id=row['record_id'],success=success,primitive_steps=len(actions),decisions=len(decisions),predictor_calls=calls,prediction_blocks=blocks,optimizer_seed=seed,prelude=setup['prelude'],elapsed_seconds=round(time.perf_counter()-start,4))
  stem=OUT/a.task/f'H{H}'/mode/f"q{ordinal:03}_{condition}"
  stem.parent.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(stem.with_suffix('.npz'),actions=np.asarray(actions,np.float32).reshape(-1,native.action_dim),score_current=np.asarray(encoded,np.float32).reshape(-1,192),scoring_goal_encoding=scoring_goal)
  stem.with_suffix('.json').write_text(json.dumps({'compact':result,'identity':row,'decisions':decisions,'trajectory':trajectory,'true_goal':np.asarray(true_goal).tolist()},indent=2))
  return result
 finally:env.close()

def main(a):
 warnings.filterwarnings('ignore',message='.*Casting input x to numpy array.*')
 warnings.filterwarnings('ignore',message='.*Box.*precision lowered.*')
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 p0=runtime.read(BASE/'protocol.json');original_H=p0['studies'][a.task]['horizon']
 hs=sorted(set([25,50,100,original_H])) if not a.horizons else [int(h) for h in a.horizons.split(',')]
 folder=OUT/a.task;folder.mkdir(parents=True,exist_ok=True)
 config={'task':a.task,'query_range':[a.start,a.stop],'horizons':hs,'checkpoint':'unchanged MAIN task-specific frozen LeWM checkpoint','query_protocol':str(BASE/'protocol.json'),'action_scaler':'unchanged full-dataset pretrained StandardScaler','planning_blocks':5,'action_block':5,'receding_blocks':5,'history_len':1,'prediction_history':3,'cem':{'candidates':300,'iterations':30,'elites':30,'sigma':1.0,'warm_start':True},'primary_mode':'released_raw','additional_control':'released_box at original offset','budget_rule':'unchanged per-H main study budgets','query_seed_rule':'unchanged query_seed(task,ordinal)','software':{'torch':torch.__version__,'stable_worldmodel':importlib.metadata.version('stable-worldmodel')},'visible_gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}
 config_path=folder/f'protocol_q{a.start:03}-{a.stop:03}.json';config_path.write_text(json.dumps(config,indent=2))
 _,_,_,_,api,ew,context,native,_=runtime.load_runtime(a.task,'confirm')
 sources={(s['identity']['record_id'],s['condition']):s for s in runtime.read(BASE/'source_manifest.json')['rows'] if s['task']==a.task}
 counter=runtime.module('official_work_counter',runtime.REFERENCE).WorkCounter(native.model)
 records=folder/f'outcomes_q{a.start:03}-{a.stop:03}.jsonl';prior={}
 if records.exists():
  for line in records.read_text().splitlines():
   r=json.loads(line);prior[(r['H'],r['query_ordinal'],r['condition'],r['mode'])]=r
 first=True;done=0;expected=sum((2 if H==original_H else 1)*len(p0['conditions'])*(a.stop-a.start) for H in hs)
 started=time.perf_counter()
 try:
  with records.open('a') as stream:
   for H in hs:
    p=copy.deepcopy(p0);p['studies'][a.task]['horizon']=H;p['studies'][a.task]['budget']=H*(1 if a.task=='cube' else 2)
    if H==original_H:assert p['studies'][a.task]['budget']==p0['studies'][a.task]['budget']
    cached_root=BASE if H==original_H else OLD/'horizon_states'/f'H{H}'
    for ordinal in range(a.start,a.stop):
     row=copy.deepcopy(p['studies'][a.task]['confirm'][ordinal]);row.update(horizon=H,global_goal=int(row['global_start'])+H)
     for condition in p['conditions']:
      source=sources[(row['record_id'],condition)]
      if first:
       env,_,_,_,_,cached=run_suite.LockedConstruction.construct(ew,api,context,native.scaler,a.task,row,'confirm',cached_root,p,source)
       try:equivalence=check_policy(native,env,cached,query_seed(a.task,ordinal))
       finally:env.close()
       (folder/f'equivalence_q{a.start:03}.json').write_text(json.dumps(equivalence,indent=2));first=False
       print(json.dumps({'task':a.task,'equivalence':equivalence}),flush=True)
      for project in ([False,True] if H==original_H else [False]):
       mode='released_box' if project else 'released_raw'
       key=(H,ordinal,condition,mode)
       if key in prior:done+=1;continue
       result=episode(a,H,ordinal,condition,project,p,row,source,api,ew,context,native,counter,cached_root)
       stream.write(json.dumps(result)+'\n');stream.flush();done+=1
       if done%12==0:print(json.dumps({'task':a.task,'range':[a.start,a.stop],'completed':done,'expected':expected,'elapsed_seconds':round(time.perf_counter()-started,1)}),flush=True)
  final={'completed':done,'expected':expected,'complete':done==expected,'elapsed_seconds':time.perf_counter()-started}
  (folder/f'complete_q{a.start:03}-{a.stop:03}.json').write_text(json.dumps(final,indent=2));print(json.dumps(final),flush=True)
 finally:counter.handle.remove()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--stop',type=int,required=True);p.add_argument('--horizons',default='');main(p.parse_args())
