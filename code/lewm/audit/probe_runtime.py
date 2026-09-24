from pathlib import Path
import argparse, copy, inspect, json, os, sys
import numpy as np
import torch

OLD=Path('runtime/ap/evidence_strengthening_20260923/results_v2')
sys.path.insert(0,str(OLD/'code'))
import common_rollout as runtime
import run_suite
import run_long_v2
BASE=run_suite.BASE
OUT=Path(__file__).resolve().parent

def main(task):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    _,_,_,_,api,ew,context,native,_=runtime.load_runtime(task,'confirm')
    definitions={name:inspect.getsource(getattr(native.model,name)) for name in ['get_cost','rollout','predict','encode']}
    definitions['shared_cache']=inspect.getsource(api.shared_cache)
    definitions['make_live']=inspect.getsource(ew.runtime._make_live)
    for name,code in definitions.items():
        (OUT/f'source_{name}.py').write_text(code)
    result={'task':task,'model_class':str(type(native.model)), 'model_source':inspect.getsourcefile(type(native.model)), 'native_class_source':inspect.getsourcefile(type(native)), 'encoder_interpolation':getattr(native.model,'interpolate_pos_encoding',None), 'action_mean':native.mean.tolist(), 'action_std':native.std.tolist(), 'source_sizes':{k:len(v) for k,v in definitions.items()},'goals':[]}
    p0=runtime.read(BASE/'protocol.json')
    sources={(s['identity']['record_id'],s['condition']):s for s in runtime.read(BASE/'source_manifest.json')['rows'] if s['task']==task}
    with ew.h5py.File(ew.v6._data_path(context),'r') as handle, torch.inference_mode():
        for h in [25,p0['studies'][task]['horizon']]:
            p=copy.deepcopy(p0)
            p['studies'][task]['horizon']=h
            p['studies'][task]['budget']=h*(1 if task=='cube' else 2)
            row=copy.deepcopy(p['studies'][task]['confirm'][0])
            row.update(horizon=h,global_goal=int(row['global_start'])+h)
            source=sources[(row['record_id'],'clean')]
            cached_root=BASE if h==p0['studies'][task]['horizon'] else OLD/'horizon_states'/f'H{h}'
            env,obs,info,true_goal,setup,cached=run_suite.LockedConstruction.construct(ew,api,context,native.scaler,task,row,'confirm',cached_root,p,source)
            actual=np.asarray(handle['pixels'][row['global_goal']])
            actual_encoded=native.encode_rgb(np.ascontiguousarray(actual))[0].cpu().numpy()
            cached_encoded=native.encode_rgb(np.ascontiguousarray(cached['goal_rgb']))[0].cpu().numpy()
            native.begin(row,dict(initial_rgb=cached['post_rgb'],goal_rgb=cached['goal_rgb'],goal=cached['goal']))
            result['goals'].append({'H':h,'global_goal':row['global_goal'],'cached_rgb_matches_goal_frame':bool(np.array_equal(actual,cached['goal_rgb'])),'cached_goal_max_encoding_difference':float(np.abs(actual_encoded-cached_encoded).max()),'true_goal':np.asarray(true_goal).tolist(),'low':env.action_space.low.tolist(),'high':env.action_space.high.tolist(),'initial_success':bool(setup['prelude']['success'])})
            env.close()
    (OUT/f'runtime_probe_{task}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',default='pusht');a=parser.parse_args();main(a.task)
