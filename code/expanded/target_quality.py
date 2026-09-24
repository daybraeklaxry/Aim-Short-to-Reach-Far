"""Held-out-memory target quality; no evaluation queries or outcome-based selection."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from train_grid import setup,memory_path,sample_tables,sample
from run_suite import learned
import common_rollout as runtime

ROOT=Path('runtime/ap/evidence_strengthening_20260923/results_v2')

def retrieve_without_episode(bank,current,goal,h,episode):
 valid,features,mean,std=bank.for_delta(h)
 x=torch.from_numpy(np.concatenate([current,goal,goal-current],axis=-1)).float().to(bank.device)
 q=((x-mean)/std).reshape(len(current),3,-1)
 q=(q*torch.from_numpy(np.sqrt(bank.weights)).to(bank.device)[None,:,None]).reshape(len(current),-1)
 best=torch.full((len(current),),float('inf'),device=bank.device);indices=torch.zeros(len(current),dtype=torch.long,device=bank.device)
 for lower in range(0,len(valid),65536):
  upper=min(len(valid),lower+65536);block=features[lower:upper]
  score=(q.square().sum(-1)[:,None]+bank.cached_squared_norm[lower:upper][None]-2*q@block.T).clamp_min(0)
  episodes=torch.as_tensor(bank.episode_idx[valid[lower:upper]],device=bank.device)
  score.masked_fill_(torch.as_tensor(episode,device=bank.device)[:,None]==episodes[None],float('inf'))
  values,pos=score.min(-1);take=values<best;best=torch.where(take,values,best);indices=torch.where(take,pos+lower,indices)
 assert torch.isfinite(best).all()
 result=valid[indices.cpu().numpy()]
 assert np.all(bank.episode_idx[result]!=episode)
 return result

def nearest_distance(z,targets,indices,device):
 result=torch.full((len(targets),),float('inf'),device=device)
 for lower in range(0,len(indices),32768):
  memory=torch.as_tensor(np.asarray(z[indices[lower:lower+32768]]),device=device)
  for i in range(0,len(targets),128):
   q=targets[i:i+128]
   d=(q.square().sum(-1)[:,None]+memory.square().sum(-1)[None]-2*q@memory.T).clamp_min(0)
   result[i:i+len(q)]=torch.minimum(result[i:i+len(q)],d.min(-1).values)
 return result.cpu().numpy()

def main(task):
 p,ew,ctx,native,boundary=setup(task);bank=ctx['bank'];model,selected=learned(task,native.device)
 protocol=runtime.read(Path(selected['checkpoint']).parent/'protocol.json')
 val=np.asarray(protocol['validation_episodes'],np.int64);H=p['studies'][task]['horizon']
 si,hs=sample(sample_tables(val,bank.lengths,bank.offsets,H),bank.offsets,np.random.default_rng(26092302),4096)
 z=np.load(memory_path(task)/'native_fp32_memory.npy',mmap_mode='r')
 out=ROOT/'target_quality'/task;out.mkdir(parents=True,exist_ok=True)
 output=out/'summary.json'
 if output.exists():assert runtime.read(output)['complete'];return
 pred=np.empty((4096,192),np.float32);picked=np.empty(4096,np.int64);started=time.perf_counter()
 with torch.inference_mode():
  for h in sorted(set(hs)):
   positions=np.flatnonzero(hs==h)
   for lo in range(0,len(positions),32):
    idx=positions[lo:lo+32];s=si[idx]
    current=torch.as_tensor(np.asarray(z[s]),device=native.device);goal=torch.as_tensor(np.asarray(z[s+h]),device=native.device)
    pred[idx]=model(current,goal,torch.full((len(idx),),float(h),device=native.device)).cpu().numpy()
    # Retrieval uses the same cached representation and normalization as MAIN.
    picked[idx]=retrieve_without_episode(bank,bank.latents[s],bank.latents[s+h],int(h),bank.episode_idx[s])
   torch.cuda.empty_cache()
   print(json.dumps(dict(task=task,phase='validation_targets',h=int(h),rows=int((hs<=h).sum()),total=4096)),flush=True)
  truth=np.asarray(z[si+5]);retrieved=np.asarray(z[picked+5])
  mlp_error=np.square(pred-truth).sum(-1);retrieved_error=np.square(retrieved-truth).sum(-1)
  # Observed targets are actual memory entries, so their nearest-memory distance is exactly zero.
  memory_indices=np.flatnonzero(bank.train_mask[bank.episode_idx])
  distance=nearest_distance(z,torch.as_tensor(pred,device=native.device),memory_indices,native.device)
 rows=dict(source=si,horizon=hs,retrieved_source=picked,mlp_squared_error=mlp_error,retrieved_squared_error=retrieved_error,mlp_nearest_memory_squared_distance=distance,retrieved_nearest_memory_squared_distance=np.zeros(4096))
 np.savez_compressed(out/'samples.npz',**rows)
 stats=lambda x:dict(mean=float(np.mean(x)),median=float(np.median(x)))
 summary=dict(complete=True,task=task,validation_samples=4096,query_episodes_used=False,checkpoint=selected['checkpoint'],retrieval='MAIN cached pair-feature normalization; candidate source episode differs from each held-out validation query episode',target_coordinates='same frozen FP32 LeWM scoring encoder for network inputs, true successors, and retrieved target observations',mlp_successor_squared_l2=stats(mlp_error),retrieved_successor_squared_l2=stats(retrieved_error),mlp_nearest_memory_squared_l2=stats(distance),retrieved_nearest_memory_squared_l2=dict(mean=0.0,median=0.0),nearest_memory='All allowable MAIN memory observations, including validation memory; no control-query observations. Observed targets have an exact zero-distance member.',peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_seconds=time.perf_counter()-started)
 runtime.write(output,summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--task',required=True);main(parser.parse_args().task)
