from pathlib import Path
import argparse,inspect,json,sys,time
import torch
R=Path(__file__).parent
sys.path.insert(0,str(R.parent/'baseline'))
import official_baseline as baseline
p=argparse.ArgumentParser();p.add_argument('--task',required=True);a=p.parse_args()
start=time.time()
root,main_protocol,protocol,native_protocol,api,ew,context,native,construction=baseline.runtime.load_runtime(a.task,'confirm')
cache=R/'cache';cache.mkdir(exist_ok=True)
dataset_names={'cube':'ogbench/cube_single_expert','pusht':'pusht_expert_train','reacher':'dmc/reacher_random','tworoom':'tworoom'}
main=json.loads((baseline.BASE/'protocol.json').read_text())
data=Path(main['studies'][a.task]['data_path'])
link=cache/(dataset_names[a.task]+'.h5');link.parent.mkdir(parents=True,exist_ok=True)
if not link.exists():link.symlink_to(data)
dest=cache/a.task/'lewm_object.ckpt';dest.parent.mkdir(parents=True,exist_ok=True)
model=native.model
meta={'task':a.task,'checkpoint_class':str(type(model)),'checkpoint_source':inspect.getsourcefile(type(model)),'state_tensors':len(model.state_dict()),'parameters':sum(x.numel() for x in model.parameters()),'dataset':str(data),'dataset_name':dataset_names[a.task],'dataset_bytes':data.stat().st_size,'model_object':str(dest),'transform':str(native.transform),'action_mean':native.scaler.mean_.tolist(),'action_scale':native.scaler.scale_.tolist()}
torch.save(model.cpu(),dest)
restored=torch.load(dest,map_location='cpu',weights_only=False)
meta['serialization_exact']=all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
meta['seconds']=time.time()-start
(R/'reference'/f'assets_{a.task}.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta),flush=True)
