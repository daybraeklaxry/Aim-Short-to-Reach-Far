"""Encode missing scoring goals for offline analysis; no environment steps or policy changes."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from train_grid import setup
import common_rollout as runtime
ROOT=Path('runtime/ap/evidence_strengthening_20260923/results_v2')
def main(task):
 missing=runtime.read(ROOT/'mechanism_all/summary.json')['missing_goal_encodings']
 traces=[]
 for entry in missing:
  stem=Path(entry['trace']);info=runtime.read(stem.with_suffix('.json'))
  if info['compact']['task']==task:traces.append((stem,info['compact']))
 if not traces:return
 _,ew,context,native,_=setup(task)
 out=ROOT/'goal_encodings'/task;out.mkdir(parents=True,exist_ok=True)
 values={};mapping={}
 with ew.h5py.File(ew.v6._data_path(context),'r') as f,torch.inference_mode():
  for stem,c in traces:
   index=int(c['identity']['global_goal']);key=f'frame_{index}'
   if key not in values:values[key]=native.encode_rgb(np.ascontiguousarray(f['pixels'][index]))[0].detach().cpu().numpy()
   mapping[str(stem)]=key
 np.savez_compressed(out/'scoring_goals.npz',**values)
 runtime.write(out/'summary.json',dict(complete=True,task=task,frames=len(values),trace_mapping=mapping,encoder='same frozen FP32 singleton scoring encoder',environment_steps=0,old_traces_modified=False,checkpoint_scope='MAIN task-specific LeWM',peak_reserved_bytes=torch.cuda.max_memory_reserved()))
 print(json.dumps(dict(task=task,frames=len(values),traces=len(mapping),complete=True)),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);main(p.parse_args().task)
