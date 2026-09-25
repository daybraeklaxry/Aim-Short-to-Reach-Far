"""Fixed-query LeWM baseline diagnosis; all outputs are separate from publication results."""
from pathlib import Path
from types import SimpleNamespace, MethodType
import argparse, contextlib, copy, io, json, os, sys, time
import numpy as np
import torch
from gymnasium.spaces import Box
from stable_worldmodel.policy import WorldModelPolicy, PlanConfig

OLD=Path('runtime/ap/evidence_strengthening_20260923/results_v2')
sys.path.insert(0,str(OLD/'code'))
import common_rollout as runtime
import run_suite
from observed_target import cem, query_seed
BASE=run_suite.BASE
OUT=Path(__file__).resolve().parent

VARIANTS=[
 dict(name='main_5',blocks=1,sigma=1/3,warm=False,history=1,clip=True,main=True),
 dict(name='released_1_sigma_third',blocks=1,sigma=1/3,warm=False,history=1,clip=True),
 dict(name='released_1_sigma_one',blocks=1,sigma=1,warm=False,history=1,clip=True),
 dict(name='released_5_history_one',blocks=5,sigma=1,warm=False,history=1,clip=True),
 dict(name='released_5_warm',blocks=5,sigma=1,warm=True,history=1,clip=True),
 dict(name='published_long',blocks=5,sigma=1,warm=True,history=3,clip=True),
 dict(name='released_no_projection',blocks=5,sigma=1,warm=True,history=3,clip=False),
 dict(name='released_execute_25',blocks=5,sigma=1,warm=True,history=3,clip=False,execute=25),
]

def raw_info(native,rgb,goal):
 return dict(pixels=np.asarray(rgb)[None,None],goal=np.asarray(goal)[None,None],action=np.zeros((1,1,native.action_dim),np.float32))

@torch.inference_mode()
def equivalence(native,env,cached,seed):
 from native_adapter import make_native_cem
 rgb,goal=cached['post_rgb'],cached['goal_rgb']
 native.begin({},dict(initial_rgb=rgb,goal_rgb=goal,goal=cached['goal']))
 native.solver.torch_gen.manual_seed(seed)
 with contextlib.redirect_stdout(io.StringIO()):actual,_,_=native.choose(None,0)
 solver=make_native_cem(native.model,native.action_dim,native.device)
 solver.torch_gen.manual_seed(seed)
 policy=WorldModelPolicy(solver=solver,config=PlanConfig(horizon=5,receding_horizon=1,history_len=1,action_block=5,warm_start=True),process={'action':native.scaler},transform={'pixels':native.transform,'goal':native.transform})
 policy.set_env(SimpleNamespace(num_envs=1,action_space=Box(low=env.action_space.low[None],high=env.action_space.high[None],dtype=np.float32)))
 with contextlib.redirect_stdout(io.StringIO()):expected=np.stack([policy.get_action(raw_info(native,rgb,goal))[0] for _ in range(5)])
 report={'released_policy_first_block_max_abs_error':float(np.max(np.abs(actual-expected))), 'released_policy_first_block_equal':bool(np.array_equal(actual,expected))}
 # One-block final-goal costs: direct official image path vs cached-encoding path.
 n=300
 gen=torch.Generator(device=native.device).manual_seed(seed)
 candidates=torch.randn((1,n,1,5*native.action_dim),device=native.device,generator=gen)/3
 current=native.encode_rgb(rgb);target=native.encode_rgb(goal)
 pixels=native.pixels(rgb)[:,None].expand(-1,n,-1,-1,-1,-1)
 info=dict(pixels=pixels,goal=native.pixels(goal)[:,None].expand(-1,n,-1,-1,-1,-1),action=torch.zeros((1,n,1,5*native.action_dim),device=native.device))
 direct=native.model.get_cost(info,candidates)
 latent=native.latent_cost.get_cost(dict(pixels=pixels,current_latent=current[:,None,None,:].expand(-1,n,-1,-1),subgoal_latent=target[:,None,:].expand(-1,n,-1)),candidates)
 report.update(one_block_cost_max_abs_error=float((direct-latent).abs().max()),one_block_elite_indices_equal=bool(torch.equal(direct.topk(30,largest=False).indices,latent.topk(30,largest=False).indices)))
 assert report['released_policy_first_block_max_abs_error']<1e-6,report
 assert report['one_block_cost_max_abs_error']<1e-5,report
 return report

@torch.inference_mode()
def episode(task,ordinal,H,cfg,p,source,row,api,ew,context,native,cached_root):
 env,obs,info,true_goal,setup,cached=run_suite.LockedConstruction.construct(ew,api,context,native.scaler,task,row,'confirm',cached_root,p,source)
 seed=query_seed(task,ordinal);torch.manual_seed(seed)
 native.solver.torch_gen.manual_seed(seed)
 native.solver.var_scale=cfg['sigma'];native.solver._config.horizon=cfg['blocks']
 native.execute_steps=cfg.get('execute',5)
 low,high=env.action_space.low,env.action_space.high
 bounds=native.scaler.transform(np.stack([low,high])).astype(np.float32)
 lower=torch.tensor(np.tile(bounds[0],5),device=native.device);upper=torch.tensor(np.tile(bounds[1],5),device=native.device)
 original_cost=native.direct_cost.get_cost
 original_rollout=native.model.rollout
 def rollout(this,info,actions,history_size=3):return original_rollout(info,actions,history_size=cfg['history'])
 native.model.rollout=MethodType(rollout,native.model)
 counts={'scored_blocks':0,'projected_blocks':0,'first_round_blocks':0,'first_round_projected':0,'cost_calls':0}
 def cost(info,candidates):
  bounded=torch.minimum(torch.maximum(candidates,lower),upper)
  changed=int((bounded!=candidates).any(-1).sum());total=candidates.numel()//(5*native.action_dim)
  counts['scored_blocks']+=total;counts['projected_blocks']+=changed
  if counts['cost_calls']%30==0:counts['first_round_blocks']+=total;counts['first_round_projected']+=changed
  counts['cost_calls']+=1
  return original_cost(info,bounded if cfg['clip'] else candidates)
 native.direct_cost.get_cost=cost
 native.begin(row,dict(initial_rgb=cached['post_rgb'],goal_rgb=cached['goal_rgb'],goal=cached['goal']))
 generator=torch.Generator(device=native.device).manual_seed(seed)
 target=native.encode_rgb(cached['goal_rgb'])
 actions=[];decisions=[];success=bool(setup['prelude']['success']);stopped=bool(setup['prelude']['terminal_without_success'])
 start=time.perf_counter();budget=int(p['studies'][task]['budget'])
 try:
  while not success and not stopped and len(actions)<budget:
   if cfg.get('main'):
    rgb=np.ascontiguousarray(env.render());current=native.encode_rgb(rgb)
    raw,details=cem(native,rgb,current,target,generator,np.zeros((5,native.action_dim),np.float32),bounds,np.stack([low,high]))
   else:
    if not cfg['warm']:native._next_init=None
    with contextlib.redirect_stdout(io.StringIO()):raw,details,_=native.choose(None,len(actions))
   effective=np.clip(raw,low,high).astype(np.float32) if cfg['clip'] else raw
   decisions.append({'step':len(actions),'raw_max':float(np.abs(raw).max()),'raw_out_of_box_components':int(((raw<low)|(raw>high)).sum()),'selected_cost':details.get('selected_cost',details.get('last_elite_mean_cost'))})
   for i,action in enumerate(effective):
    obs,_,term,trunc,info=env.step(action.copy());actions.append(action.copy())
    success=bool(ew.runtime._success_now(task,obs,info,true_goal,False,False));stopped=bool((term or trunc) and not success)
    continuing=not success and not stopped and len(actions)<budget
    native.completed_primitive(env,i,continuing)
    if not continuing:break
  result=dict(task=task,H=H,query=ordinal,variant=cfg['name'],success=success,steps=len(actions),decisions=len(decisions),elapsed_seconds=round(time.perf_counter()-start,3),**counts)
  name=f"{task}_H{H}_q{ordinal:03}_{cfg['name']}"
  np.savez_compressed(OUT/'traces'/f'{name}.npz',actions=np.asarray(actions,np.float32),goal=target.cpu().numpy())
  (OUT/'traces'/f'{name}.json').write_text(json.dumps({'result':result,'configuration':cfg,'decisions':decisions},indent=2))
  return result
 finally:
  native.direct_cost.get_cost=original_cost;native.model.rollout=original_rollout;native.execute_steps=5;env.close()

def main(a):
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 (OUT/'traces').mkdir(exist_ok=True)
 p=runtime.read(BASE/'protocol.json');original_H=p['studies'][a.task]['horizon']
 p['studies'][a.task]['horizon']=a.horizon;p['studies'][a.task]['budget']=a.horizon*(1 if a.task=='cube' else 2)
 _,_,_,_,api,ew,context,native,_=runtime.load_runtime(a.task,'confirm')
 sources={(s['identity']['record_id'],s['condition']):s for s in runtime.read(BASE/'source_manifest.json')['rows'] if s['task']==a.task}
 cached_root=BASE if a.horizon==original_H else OLD/'horizon_states'/f'H{a.horizon}'
 row=copy.deepcopy(p['studies'][a.task]['confirm'][0]);row.update(horizon=a.horizon,global_goal=int(row['global_start'])+a.horizon)
 env,_,_,_,_,cached=run_suite.LockedConstruction.construct(ew,api,context,native.scaler,a.task,row,'confirm',cached_root,p,sources[(row['record_id'],'clean')])
 try:report=equivalence(native,env,cached,query_seed(a.task,0))
 finally:env.close()
 (OUT/f'equivalence_{a.task}.json').write_text(json.dumps(report,indent=2));print(json.dumps({'equivalence':report}),flush=True)
 selected=[v for v in VARIANTS if not a.variants or v['name'] in a.variants.split(',')]
 dest=OUT/f'outcomes_{a.task}_H{a.horizon}_{a.tag}.jsonl'
 prior={}
 if dest.exists():
  for line in dest.read_text().splitlines():
   r=json.loads(line);prior[(r['query'],r['variant'])]=r
 with dest.open('a') as stream:
  for ordinal in range(a.queries):
   row=copy.deepcopy(p['studies'][a.task]['confirm'][ordinal]);row.update(horizon=a.horizon,global_goal=int(row['global_start'])+a.horizon)
   for cfg in selected:
    if (ordinal,cfg['name']) in prior:continue
    r=episode(a.task,ordinal,a.horizon,cfg,p,sources[(row['record_id'],'clean')],row,api,ew,context,native,cached_root)
    stream.write(json.dumps(r)+'\n');stream.flush();print(json.dumps(r),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task',default='pusht');p.add_argument('--horizon',type=int,default=25);p.add_argument('--queries',type=int,default=8);p.add_argument('--variants',default='');p.add_argument('--tag',default='diagnostic');main(p.parse_args())
