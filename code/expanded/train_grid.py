"""Prespecified MAIN-memory target regression; no control-query model selection."""
from pathlib import Path
import argparse,datetime,json,math,os,time
import numpy as np
import torch
from torch import nn
import common_rollout as runtime
from memory_mask import apply as apply_memory_mask

BASE=Path('runtime/ap/fresh_target_interface_study_v2')
ROOT=Path('runtime/ap/evidence_strengthening_20260923/results_v2')
LEGACY=ROOT.parent/'learned_target/cube'
TASKS=('cube','pusht','reacher','tworoom')

def save(path,data):runtime.write(path,data)
def memory_path(task):return (LEGACY if task=='cube' else ROOT/'features'/task)
def setup(task):
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 p=runtime.read(BASE/'protocol.json')
 *_,api,ew,ctx,native,construction=runtime.load_runtime(task,'confirm')
 boundary=apply_memory_mask(ctx,p,task)
 return p,ew,ctx,native,boundary

def sample_tables(eps,lengths,offsets,H):
 result={}
 for h in range(5,H+1):
  e=eps[lengths[eps]>h];n=lengths[e]-h
  assert len(e);result[h]=(e,np.cumsum(n))
 return result

def sample(spec,offsets,rng,n):
 hs=rng.integers(min(spec),max(spec)+1,size=n);ss=np.empty(n,np.int64)
 for h in np.unique(hs):
  mask=hs==h;e,ends=spec[int(h)];v=rng.integers(0,int(ends[-1]),size=int(mask.sum()))
  j=np.searchsorted(ends,v,side='right');prev=np.where(j==0,0,ends[np.maximum(0,j-1)])
  ss[mask]=offsets[e[j]]+v-prev
 return ss,hs

class TargetMLP(nn.Module):
 def __init__(self,width,residual,xmean,xstd,ymean,ystd,H):
  super().__init__();self.residual=bool(residual);self.H=H
  self.register_buffer('xmean',xmean);self.register_buffer('xstd',xstd)
  self.register_buffer('ymean',ymean);self.register_buffer('ystd',ystd)
  layers=[];d=577
  for _ in range(3):layers.extend([nn.Linear(d,width),nn.LayerNorm(width),nn.GELU()]);d=width
  layers.append(nn.Linear(width,192));self.net=nn.Sequential(*layers)
 def forward(self,current,goal,h):
  x=torch.cat([current,goal,goal-current,h.reshape(-1,1)/self.H],-1)
  y=self.net((x-self.xmean)/self.xstd)*self.ystd+self.ymean
  return y+current if self.residual else y

def cache(task):
 p,ew,ctx,native,boundary=setup(task);bank=ctx['bank'];out=memory_path(task);out.mkdir(parents=True,exist_ok=True)
 receipt=out/'native_fp32_memory.json';path=out/'native_fp32_memory.npy'
 if receipt.exists():
  r=runtime.read(receipt);assert r['complete'] and r['included_episodes']==np.flatnonzero(bank.train_mask).tolist()
  return dict(reused=True,path=str(path),receipt=str(receipt))
 assert task!='cube','Cube cache is owned by the already-running encoding job.'
 mask=bank.train_mask[bank.episode_idx];temporary=out/'native_fp32_memory.building.npy'
 assert not temporary.exists(),temporary
 z=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=bank.latents.shape)
 n=0;t=time.perf_counter()
 with ew.h5py.File(ew.v6._data_path(ctx),'r') as handle,torch.inference_mode():
  for lo in range(0,len(mask),256):
   hi=min(len(mask),lo+256);chosen=mask[lo:hi]
   if not chosen.any():continue
   images=np.asarray(handle['pixels'][lo:hi])[chosen]
   pixels=native.transform(torch.from_numpy(np.ascontiguousarray(images)).permute(0,3,1,2)).to(native.device)
   encoded=native.model.encode(dict(pixels=pixels[:,None]))['emb'][:,-1].float().cpu().numpy()
   z[lo:hi][chosen]=encoded;n+=len(encoded)
   if lo%(256*100)==0:print(json.dumps(dict(task=task,phase='encode',frames=n,total=int(mask.sum()),elapsed=round(time.perf_counter()-t,1))),flush=True)
 assert n==int(mask.sum());z.flush();del z;os.replace(temporary,path)
 save(receipt,dict(complete=True,frames=n,dimension=192,dtype='float32',encoder='MAIN NativeAdapter FP32 projection and preprocessing',included_episodes=np.flatnonzero(bank.train_mask).tolist(),query_episodes_excluded=True,elapsed_seconds=time.perf_counter()-t))
 return dict(reused=False,path=str(path),receipt=str(receipt))

def train(a):
 p,ew,ctx,native,boundary=setup(a.task);bank=ctx['bank'];H=int(p['studies'][a.task]['horizon'])
 receipt=runtime.read(memory_path(a.task)/'native_fp32_memory.json')
 episodes=np.flatnonzero(bank.train_mask);assert receipt['complete'] and receipt['included_episodes']==episodes.tolist()
 ordered=np.random.default_rng(26092301).permutation(episodes);nval=int(round(.05*len(ordered)))
 val_eps,train_eps=ordered[:nval],ordered[nval:]
 assert not set(episodes)&{r['episode'] for r in p['studies'][a.task]['confirm']}
 mapped=np.load(memory_path(a.task)/'native_fp32_memory.npy',mmap_mode='r')
 z=torch.tensor(np.asarray(mapped),device=native.device,dtype=torch.float32)
 train_spec=sample_tables(train_eps,bank.lengths,bank.offsets,H);val_spec=sample_tables(val_eps,bank.lengths,bank.offsets,H)
 si,sh=sample(train_spec,bank.offsets,np.random.default_rng(26092303),262144)
 vi,vh=sample(val_spec,bank.offsets,np.random.default_rng(26092302),4096)
 # Every selected target is inside a memory episode; the control query set is disjoint.
 for ids,hs,allowed in [(si,sh,train_eps),(vi,vh,val_eps)]:
  assert np.isin(bank.episode_idx[ids],allowed).all()
  np.testing.assert_array_equal(bank.episode_idx[ids],bank.episode_idx[ids+hs])
 with torch.no_grad():
  sums=[torch.zeros(n,device=z.device,dtype=torch.float64) for n in [577,577,192,192]]
  for lo in range(0,len(si),4096):
   ii=torch.as_tensor(si[lo:lo+4096],device=z.device);hh=torch.as_tensor(sh[lo:lo+4096],device=z.device)
   cur=z[ii];goal=z[ii+hh];y=z[ii+5]-(cur if a.residual else 0)
   x=torch.cat([cur,goal,goal-cur,hh.float()[:,None]/H],-1).double();y=y.double()
   sums[0]+=x.sum(0);sums[1]+=x.square().sum(0);sums[2]+=y.sum(0);sums[3]+=y.square().sum(0)
  xm=sums[0]/len(si);xs=(sums[1]/len(si)-xm.square()).clamp_min(0).sqrt().clamp_min(1e-4)
  ym=sums[2]/len(si);ys=(sums[3]/len(si)-ym.square()).clamp_min(0).sqrt().clamp_min(1e-4)
  xm,xs,ym,ys=[v.float() for v in [xm,xs,ym,ys]]
 seed=26092311+TASKS.index(a.task)
 torch.manual_seed(seed);rng=np.random.default_rng(seed)
 model=TargetMLP(a.width,a.residual,xm,xs,ym,ys,H).to(z.device)
 opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
 out=ROOT/'training'/a.task/f'w{a.width}_{"residual" if a.residual else "absolute"}'
 out.mkdir(parents=True,exist_ok=True);summary=out/'summary.json'
 if summary.exists():assert runtime.read(summary)['complete'];return runtime.read(summary)
 state=out/'resume.pt';step0=0;best=float('inf');stale=0;best_step=0
 if state.exists():
  d=torch.load(state,map_location=z.device,weights_only=False);model.load_state_dict(d['model']);opt.load_state_dict(d['optimizer']);step0=d['step'];best=d['best'];stale=d['stale'];best_step=d['best_step'];rng.bit_generator.state=d['rng_state']
 protocol=dict(task=a.task,width=a.width,residual=a.residual,hidden_layers=3,activation='GELU',normalization='LayerNorm after each hidden linear; training-only per-dimension input/output standardization',train_episodes=train_eps.tolist(),validation_episodes=val_eps.tolist(),validation_samples=4096,standardization_samples=262144,maximum_updates=50000,batch=1024,learning_rate=3e-4,weight_decay=1e-4,optimizer='AdamW',decay='cosine to zero over 50000 steps',early_stopping='10 consecutive validation checks without improvement after at least 5000 updates; checked every 500',training_loss='MSE divided by the output training standard deviation',selection_metric='Raw latent-coordinate MSE on fixed memory validation transitions; common scale for residual and absolute networks',seed=seed,world_model_frozen=True,query_episodes_excluded=True,cache=str(memory_path(a.task)),boundary=boundary)
 save(out/'protocol.json',protocol)
 vi=torch.as_tensor(vi,device=z.device);vh=torch.as_tensor(vh,device=z.device)
 with ew.h5py.File(ew.v6._data_path(ctx),'r') as handle,torch.inference_mode():
  for index in vi[:3].cpu().tolist():torch.testing.assert_close(native.encode_rgb(np.ascontiguousarray(handle['pixels'][index]))[0],z[index],rtol=1e-4,atol=1e-5)
 t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
 for step in range(step0+1,50001):
  ii,hh=sample(train_spec,bank.offsets,rng,1024);ii=torch.as_tensor(ii,device=z.device);hh=torch.as_tensor(hh,device=z.device)
  model.train();pred=model(z[ii],z[ii+hh],hh.float());loss=((pred-z[ii+5])/ys).square().mean()
  assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  for group in opt.param_groups:group['lr']=.5*3e-4*(1+math.cos(math.pi*step/50000))
  if step%500==0:
   model.eval()
   with torch.no_grad():mse=float((model(z[vi],z[vi+vh],vh.float())-z[vi+5]).square().mean())
   if mse<best:
    best=mse;best_step=step;stale=0
    torch.save(dict(model=model.state_dict(),width=a.width,residual=a.residual,H=H,validation_mse=best,step=step,seed=seed),out/'best.pt')
   else:stale+=1
   torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,best=best,best_step=best_step,stale=stale,rng_state=rng.bit_generator.state),state)
   progress=dict(task=a.task,width=a.width,residual=a.residual,step=step,mse=mse,best=best,best_step=best_step,elapsed=time.perf_counter()-t,peak_reserved_bytes=torch.cuda.max_memory_reserved())
   with (out/'learning.jsonl').open('a') as f:f.write(json.dumps(progress)+'\n');f.flush()
   print(json.dumps(progress),flush=True)
   if step>=5000 and stale>=10:break
 result=dict(complete=True,task=a.task,width=a.width,residual=a.residual,updates=step,best_step=best_step,validation_mse=best,checkpoint=str(out/'best.pt'),elapsed_seconds=time.perf_counter()-t,peak_reserved_bytes=torch.cuda.max_memory_reserved())
 save(summary,result);return result

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--task',choices=TASKS,required=True);ap.add_argument('--mode',choices=['cache','train'],required=True);ap.add_argument('--width',type=int,choices=[512,1024],default=512);ap.add_argument('--residual',type=int,choices=[0,1],default=1)
 a=ap.parse_args();print(json.dumps(cache(a.task) if a.mode=='cache' else train(a)),flush=True)
