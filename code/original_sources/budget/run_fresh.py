"""Final five-arm target-interface evaluation. No historical policy reuse."""
import argparse,json,os,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import common_rollout as runtime
import construction_fresh as construction
from memory_mask import apply as apply_memory_mask
from baseline_rollout import rollout as discrete_rollout
from target_scorers import TargetRank,TargetGaussian,ARMS

def main(a):
    start=time.perf_counter();p=runtime.read(a.study_root/'protocol.json')
    assert Path(__file__).resolve().parent==a.study_root/'frozen_code'
    assert p['arms']==list(ARMS) and 0<=a.shard_index<a.shards
    profile=p['technical_profile']
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    original_root,_,old,native_protocol,api,ew,context,native,_=runtime.load_runtime(a.task,'profile' if profile else 'confirm')
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    assert native_protocol['frozen_utc']==p['native_frozen_utc']
    memory=apply_memory_mask(context,p,a.task) if not profile else dict(
        original_train_episodes=int(context['bank'].train_mask.sum()),
        active_train_episodes=int(context['bank'].train_mask.sum()),
        fresh_memory_mask_applied=False,technical_profile_old_exposed_queries=True)
    rows=p['studies'][a.task]['confirm'];conditions=p['conditions']
    sources={(s['identity']['record_id'],s['condition']):s for s in runtime.read(a.study_root/'source_manifest.json')['rows'] if s['task']==a.task}
    args=SimpleNamespace(task=a.task,stage='confirm',reference_root=a.study_root)
    counter=runtime.module('target_interface_counter',runtime.REFERENCE).WorkCounter(native.model)
    a.output.mkdir(parents=True,exist_ok=True)
    tag=f'{a.task}_shard{a.shard_index}of{a.shards}'
    runtime.write(a.output/f'run_{tag}.json',dict(study=p['study'],task=a.task,profile=profile,
        frozen_utc=p['frozen_utc'],arms=p['arms'],query_map=rows,conditions=conditions,
        memory_boundary=memory,numeric_protocol=p['numeric_protocol'],
        action_scaling=native_protocol['native_action_stats'][a.task],
        source_runtime_root=str(original_root),frozen_code=str(Path(__file__).resolve().parent),
        visible_gpu=os.environ['CUDA_VISIBLE_DEVICES'],shard_index=a.shard_index,shards=a.shards,
        setup_seconds=time.perf_counter()-start))
    outcomes=[]
    try:
        with ew.h5py.File(ew.v6._data_path(context),'r') as handle,torch.inference_mode():
            rank=TargetRank(native,handle)
            gaussian={kind:TargetGaussian(native,handle,kind) for kind in ('observed','final')}
            for ordinal,row in enumerate(rows):
                rank.set_goal_record(row['global_goal'])
                for scorer in gaussian.values():scorer.set_goal_record(row['global_goal'])
                for ci,condition in enumerate(conditions):
                    owner=(ordinal*len(conditions)+ci)%a.shards if profile else ordinal%a.shards
                    if owner!=a.shard_index:continue
                    source=sources[(row['record_id'],condition)]
                    construction.prepare(ew,api,context,native.scaler,a.task,row,'confirm',a.study_root,p,source)
                    for arm in ARMS:
                        if arm.startswith('gaussian_'):
                            trace,arrays=runtime.rollout(args,p,api,ew,context,native,construction,
                                gaussian[arm.removeprefix('gaussian_')],counter,row,source,arm,ordinal,None)
                        else:
                            trace,arrays=discrete_rollout(args,p,api,ew,context,native,construction,
                                rank,counter,row,source,arm)
                        trace['compact']['study']=p['study']
                        trace['compact']['technical_profile']=profile
                        if profile and trace['decisions'] and arm!='direct':
                            index=trace['decisions'][0]['target_index']
                            expected=row['global_goal'] if arm.endswith('_final') else int(
                                arrays['retrieved_sources' if arm.startswith('gaussian_') else 'candidate_sources'][0,0])+5
                            assert index==expected
                            target=native.encode_rgb(np.ascontiguousarray(handle['pixels'][index]))[0].cpu().numpy()
                            np.testing.assert_array_equal(target,arrays['score_target'][0])
                            trace['profile_target_image_verified']=True
                        stem=a.output/'traces'/a.task/f"confirm_{row['record_id']}_{condition}_{arm}"
                        stem.parent.mkdir(parents=True,exist_ok=True)
                        with stem.with_suffix('.npz.tmp').open('wb') as f:np.savez_compressed(f,**arrays)
                        os.replace(stem.with_suffix('.npz.tmp'),stem.with_suffix('.npz'))
                        runtime.write(stem.with_suffix('.json'),trace)
                        outcomes.append(trace['compact'])
                        (a.output/f'outcomes_{tag}.jsonl').write_text(''.join(json.dumps(v,allow_nan=False)+'\n' for v in outcomes))
                        print(json.dumps(dict(completed=len(outcomes),task=a.task,shard=a.shard_index)),flush=True)
    finally:counter.handle.remove()
    expected=(len(range(a.shard_index,len(rows)*len(conditions),a.shards))*5 if profile else
              len(range(a.shard_index,len(rows),a.shards))*len(conditions)*5)
    assert len(outcomes)==expected
    runtime.write(a.output/f'summary_{tag}.json',dict(completed=True,outcomes=len(outcomes),expected=expected,
        elapsed_seconds=time.perf_counter()-start,task=a.task,shard_index=a.shard_index))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task',choices=('cube','pusht','reacher','tworoom'),required=True)
    p.add_argument('--shards',type=int,required=True);p.add_argument('--shard-index',type=int,required=True)
    main(p.parse_args())
