"""MAIN-query target/budget/memory/horizon interventions with fixed physical prefixes."""
from pathlib import Path
from types import SimpleNamespace
import argparse,copy,csv,fcntl,json,os,time
import numpy as np
import torch
import common_rollout as runtime
import suite_cem_rollout, suite_rank_rollout
import construction_fresh as original_construction
from memory_mask import apply as apply_memory_mask
from observed_target import ValidObservedCEM,cem,query_seed
from rank_reference import RecordCalibration
from train_grid import TargetMLP
import observed_target

ROOT=Path('runtime/ap/evidence_strengthening_20260923/results_v2')
BASE=ROOT.parent.parent/'fresh_target_interface_study_v2'

class LockedConstruction:
 @staticmethod
 def construct(ew,api,context,scaler,task,row,stage,root,p,source):
  if root==BASE:return original_construction.construct(ew,api,context,scaler,task,row,stage,root,p,source)
  folder=root/'locks';folder.mkdir(parents=True,exist_ok=True)
  with (folder/f"{task}_{row['record_id']}_{source['condition']}.lock").open('a') as f:
   fcntl.flock(f,fcntl.LOCK_EX)
   return original_construction.construct(ew,api,context,scaler,task,row,stage,root,p,source)

def learned(task,device):
 selected=runtime.read(ROOT/'training'/task/'selected.json')['selected']
 checkpoint=torch.load(selected['checkpoint'],map_location=device,weights_only=False);s=checkpoint['model']
 model=TargetMLP(checkpoint['width'],checkpoint['residual'],s['xmean'],s['xstd'],s['ymean'],s['ystd'],checkpoint['H']).to(device)
 model.load_state_dict(s);model.eval().requires_grad_(False)
 return model,selected

class TargetProvider:
 def __init__(self,native,handle,cfg,task):
  self.native=native;self.handle=handle;self.cfg=cfg;self.targets={};self.remaining=cfg['H']
  self.fixed_span=cfg.get('span')=='fixed';self.L=int(cfg.get('target_L',5))
  self.kind=cfg['arm'].split('_',1)[1]
  self.changed_retrieval=cfg.get('memory_fraction',1)<1 or cfg.get('key','full')!='full' or self.fixed_span or self.L>5
  self.model,self.selected=learned(task,native.device) if self.kind=='learned' else (None,None)
 def encode_record(self,index):
  i=int(index)
  if i not in self.targets:self.targets[i]=self.native.encode_rgb(np.ascontiguousarray(self.handle['pixels'][i])).detach().cpu()
  return self.targets[i].to(self.native.device)
 def set_goal_record(self,index):self.goal_record=int(index);self.goal=self.encode_record(index)
 def target(self,current,source):
  if self.kind=='final':return self.goal
  if self.kind=='learned':return self.model(current,self.goal,torch.tensor([self.remaining],device=current.device,dtype=torch.float32))
  endpoint=self.encode_record(int(source)+self.L)
  return current+endpoint-self.encode_record(source) if self.kind=='transport' else endpoint

class SuiteCEM(ValidObservedCEM):
 def __init__(self,native,handle,cfg,task):
  super().__init__(native,handle);self.provider=TargetProvider(native,handle,cfg,task);self.iterations=cfg.get('iterations',30)
  self.target_kind=self.provider.kind;self.fixed_span=self.provider.fixed_span;self.changed_retrieval=self.provider.changed_retrieval
 def set_goal_record(self,index):self.provider.set_goal_record(index)
 @torch.inference_mode()
 def choose(self,rgb,sources,raw,normalized,generator):
  prior=np.zeros((5,self.native.action_dim),np.float32);raw_prior=self.native.scaler.inverse_transform(prior).astype(np.float32)
  effective=np.clip(raw_prior,self.raw_bounds[0],self.raw_bounds[1]).astype(np.float32);changed=not np.array_equal(effective,raw_prior)
  if changed:prior=self.native.scaler.transform(effective).astype(np.float32)
  current=self.native.encode_rgb(rgb);self.provider.remaining=self.remaining;target=self.provider.target(current,sources[0])
  chosen,detail=cem(self.native,rgb,current,target,generator,prior,self.normalized_bounds,self.raw_bounds)
  if not detail['refined_mean_selected']:chosen=effective.copy()
  detail.update(prior_kind='fixed_standardized_zero',prior_source=None,prior_projected=changed,target_kind=self.provider.kind,source_prediction_misses=0,predictor_total_calls=detail['predictor_batch_calls'],predictor_total_candidate_blocks=detail['predictor_candidate_transitions'])
  arrays=dict(score_current=current[0].cpu().numpy(),score_target=target[0].cpu().numpy(),proposed_raw_actions=chosen.copy(),normalized_prior=prior.copy(),effective_raw_prior=effective.copy(),normalized_selected=np.asarray(detail['normalized_selected'],np.float32),raw_action_bounds=self.raw_bounds.copy(),normalized_action_bounds=self.normalized_bounds.copy())
  return chosen,detail,arrays

class SuiteRank(RecordCalibration):
 def __init__(self,native,handle,cfg,task):
  super().__init__(native,handle);self.provider=TargetProvider(native,handle,cfg,task)
  self.fixed_span=self.provider.fixed_span;self.changed_retrieval=self.provider.changed_retrieval
 def set_goal_record(self,index):self.provider.set_goal_record(index)
 @torch.inference_mode()
 def choose(self,arm,rgb,sources,normalized):
  current=self.native.encode_rgb(rgb)[0];self.provider.remaining=self.remaining;target=self.provider.target(current[None],sources[0])[0]
  predictions=self.predict([rgb]*8,current[None].expand(8,-1),normalized,target)
  costs=(predictions-target[None]).square().sum(-1);rank=int(costs.argmin().item())
  return rank,dict(source_prediction_misses=0,target_kind=self.provider.kind,candidate_scores=costs.cpu().tolist()),dict(score_current=current.cpu().numpy(),score_target=target.cpu().numpy(),live_predictions=predictions.cpu().numpy(),candidate_scores=costs.cpu().numpy())

def policy_bank(context,cfg):
 bank=context['bank'];variant=copy.copy(bank);variant.train_mask=bank.train_mask.copy();variant.weights=bank.weights.copy()
 variant.cached_delta=variant.cached_bank=variant.cached_squared_norm=None
 variant.option_steps=max(5,int(cfg.get('target_L',5)))
 active=np.flatnonzero(bank.train_mask);fraction=cfg.get('memory_fraction',1)
 if fraction<1:
  ordering=np.random.default_rng(26091600).permutation(active);keep=ordering[:int(len(active)*fraction)]
  variant.train_mask[:]=False;variant.train_mask[keep]=True
 if cfg.get('key')=='no_far':variant.weights=np.asarray([1,0,1],np.float32)
 elif cfg.get('key')=='no_delta':variant.weights=np.asarray([1,1,0],np.float32)
 context['policy_bank']=variant
 return dict(episodes=int(variant.train_mask.sum()),target_L=int(cfg.get('target_L',5)),valid_action_steps=variant.option_steps,weights=variant.weights.tolist(),std='recomputed from this memory subset for each h using the unchanged for_delta rule')

def main(a):
 manifest=runtime.read(ROOT/'queue-manifest.json')
 job=next(j for j in manifest['jobs'] if j['id']==a.job_id);assert job['task']==a.task;cfg=job['config']
 p=runtime.read(BASE/'protocol.json');original_H=p['studies'][a.task]['horizon'];H=cfg['H']
 if H!=original_H:p['studies'][a.task]['horizon']=H;p['studies'][a.task]['budget']=cfg['B']
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 _,_,_,_,api,ew,context,native,_=runtime.load_runtime(a.task,'confirm');membership=apply_memory_mask(context,p,a.task)
 sources={(s['identity']['record_id'],s['condition']):s for s in runtime.read(BASE/'source_manifest.json')['rows'] if s['task']==a.task}
 cached_root=BASE if H==original_H else ROOT/'horizon_states'/f'H{H}'
 args=SimpleNamespace(task=a.task,stage='confirm',reference_root=cached_root)
 out=ROOT/'jobs'/job['id'];out.mkdir(parents=True,exist_ok=True)
 specs=policy_bank(context,cfg);runtime.write(out/'protocol.json',dict(job=job,main_protocol=str(BASE/'protocol.json'),memory=membership,intervention=specs,query_and_prefix_allocation='original MAIN manifest',prediction_horizon=5,target_training='validation-only selected' if 'learned' in cfg['arm'] else None))
 counter=runtime.module('suite_counter',runtime.REFERENCE).WorkCounter(native.model)
 assignments=[(i,c) for i in range(cfg['start'],cfg['stop']) for c in cfg['conditions']]
 if a.probe:assignments=assignments[:3]
 prior={};records=out/'episodes.jsonl'
 if records.exists():
  for line in records.open():
   r=json.loads(line);prior[(r['query_ordinal'],r['condition'])]=r
 started=time.perf_counter();complete=0
 try:
  with ew.h5py.File(ew.v6._data_path(context),'r') as handle,records.open('a') as f,torch.inference_mode():
   is_cem=cfg['arm'].startswith('cem_');is_lewm=cfg['arm']=='lewm'
   scorer=None if is_lewm else SuiteCEM(native,handle,cfg,a.task) if is_cem else SuiteRank(native,handle,cfg,a.task)
   if is_lewm:
    import run_long_v2 as long_runtime
    long_runtime.BASE=cached_root;long_runtime.construction=LockedConstruction
   observed_target.ITERATIONS=cfg.get('iterations',30)
   for ordinal,condition in assignments:
    if (ordinal,condition) in prior:complete+=1;continue
    row=copy.deepcopy(p['studies'][a.task]['confirm'][ordinal]);source=sources[(row['record_id'],condition)]
    if H!=original_H:row.update(horizon=H,global_goal=int(row['global_start'])+H)
    if scorer:scorer.set_goal_record(row['global_goal'])
    torch.cuda.reset_peak_memory_stats()
    if is_lewm:trace,arrays=long_runtime.episode(a.task,row,ordinal,source,p,api,ew,context,native,counter)
    elif is_cem:
     trace,arrays=suite_cem_rollout.rollout(args,p,api,ew,context,native,LockedConstruction,scorer,counter,row,source,'gaussian_observed',ordinal,None)
    else:trace,arrays=suite_rank_rollout.rollout(args,p,api,ew,context,native,LockedConstruction,scorer,counter,row,source,'ap_observed')
    if H==original_H and len(trace['trajectory']):
     old=runtime.read(BASE/'construction'/a.task/f"confirm_{condition}_{row['record_id']}.json")
     assert trace['construction']['post_state']==old['post_state'] and trace['compact']['prelude']==old['prelude']
    if scorer is not None:
     arrays['scoring_goal_encoding']=scorer.provider.goal[0].detach().cpu().numpy()
    else:
     arrays['scoring_goal_encoding']=native.encode_rgb(np.ascontiguousarray(handle['pixels'][int(row['global_goal'])]))[0].detach().cpu().numpy()
    trace['scoring_goal_encoding_source']='MAIN frozen FP32 scoring encoder, singleton target image'
    trace['compact'].update(job_id=job['id'],arm=cfg['arm'],config=cfg,query_ordinal=ordinal,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    stem=out/'traces'/f'{ordinal:03}_{condition}';stem.parent.mkdir(exist_ok=True)
    runtime.write(stem.with_suffix('.json'),trace);np.savez_compressed(stem.with_suffix('.npz'),**arrays)
    f.write(json.dumps(trace['compact'])+'\n');f.flush();complete+=1
    with (out/'episodes.csv').open('a',newline='') as cf:
     writer=csv.writer(cf)
     if cf.tell()==0:writer.writerow(['job_id','task','query_ordinal','condition','arm','success','steps','pred_blocks'])
     writer.writerow([job['id'],a.task,ordinal,condition,cfg['arm'],int(trace['compact']['success']),trace['compact']['primitive_steps'],trace['compact']['predictor_candidate_blocks']]);cf.flush()
    print(json.dumps(dict(job=job['id'],completed=complete,expected=len(assignments),elapsed=round(time.perf_counter()-started,1),peak_reserved_bytes=trace['compact']['peak_reserved_bytes'])),flush=True)
 finally:counter.handle.remove()
 if not a.probe:runtime.write(out/'summary.json',dict(complete=True,episodes=complete,elapsed_seconds=time.perf_counter()-started))
 else:runtime.write(out/'probe.json',dict(complete=True,episodes=complete,peak_reserved_bytes=torch.cuda.max_memory_reserved()))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--job-id',required=True);ap.add_argument('--probe',action='store_true');main(ap.parse_args())
