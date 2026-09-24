"""One-factor LeWM checks on the H=25 main protocol, without running AP."""
from pathlib import Path
import argparse,copy,json,sys,time
import h5py,numpy as np,torch
R=Path(__file__).parent
sys.path.insert(0,str(R.parent/'audit'))
import official_baseline as baseline

p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--factor',required=True,choices=['budget50','official_queries','goal_frame24','dataset_initial_image','cube_full_state']);p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=128)
a=p.parse_args()
if a.factor in ['budget50','cube_full_state']:assert a.task=='cube'
root=R/'gap_results'/a.factor;root.mkdir(parents=True,exist_ok=True);baseline.OUT=root
torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
protocol=baseline.runtime.read(baseline.BASE/'protocol.json');study=protocol['studies'][a.task]
H=24 if a.factor=='goal_frame24' else 25
study['horizon']=H;study['budget']=50 if a.task!='cube' or a.factor=='budget50' else 25
_,_,_,_,api,ew,context,native,_=baseline.runtime.load_runtime(a.task,'confirm')
baseline.run_suite.apply_memory_mask(context,protocol,a.task)
sources={(s['identity']['record_id'],s['condition']):s for s in baseline.runtime.read(baseline.BASE/'source_manifest.json')['rows'] if s['task']==a.task}
cached_root=baseline.OLD/'horizon_states/H25' if a.factor in ['budget50','dataset_initial_image'] else R/'gap_states'/a.factor
counter=baseline.runtime.module('gap_counter',baseline.runtime.REFERENCE).WorkCounter(native.model)
records=root/f'{a.task}_{a.start:03}-{a.stop:03}.jsonl';assert not records.exists(),str(records)
with h5py.File(study['data_path'],'r') as data:
    selected=None
    if a.factor=='official_queries':
        ep=data['episode_idx' if 'episode_idx' in data else 'ep_idx'][:];steps=data['step_idx'][:]
        ids,inverse=np.unique(ep,return_inverse=True);maximum=np.full(len(ids),-1,np.int64);np.maximum.at(maximum,inverse,steps)
        valid=np.flatnonzero(steps<=maximum[inverse]-25)
        selected=np.sort(valid[np.random.default_rng(42).choice(len(valid)-1,size=128,replace=False)])
        assert np.all(ep[selected]==ep[selected+25])
        # The common state-cache builder also prepares unused retrieval metadata.
        # Its holdout assertion must cover this new query set; LeWM never reads
        # the retrieval bank and its pretrained action scaler stays unchanged.
        context['bank'].train_mask[np.unique(ep[selected]).astype(np.int64)]=False
    if a.factor=='cube_full_state':
        original_make_live=ew.runtime._make_live
        def full_state(context,row):
            env,obs,info,goal=original_make_live(context,row)
            index=int(row['global_start']);e=env.unwrapped
            e.set_state(np.asarray(data['qpos'][index],np.float64),np.asarray(data['qvel'][index],np.float64))
            e.pre_step();e.post_step()
            return env,np.asarray(e.compute_observation(),np.float32),e.get_step_info(),goal
        ew.runtime._make_live=full_state
    config={'task':a.task,'factor':a.factor,'range':[a.start,a.stop],'actual_goal_frame_offset':H,'budget':study['budget'],'query_sampling_seed':42 if selected is not None else 26091391,'optimizer_seed':'unchanged paper query_seed(task,ordinal)','controller':'unchanged released 25-action LeWM','changed_setting':a.factor,'other_target_methods_run':False}
    (root/f'{a.task}_{a.start:03}-{a.stop:03}_config.json').write_text(json.dumps(config,indent=2))
    with records.open('w') as stream:
        try:
            for ordinal in range(a.start,a.stop):
                row=copy.deepcopy(study['confirm'][ordinal]);source=copy.deepcopy(sources[(row['record_id'],'clean')])
                row.update(horizon=H,global_goal=int(row['global_start'])+H)
                if selected is not None:
                    for key in ['initial_distance','canonical_record_id','cohort_selection_seed','cohort_index']:row.pop(key,None)
                    index=int(selected[ordinal]);row.update(global_start=index,global_goal=index+25,episode=int(ep[index]),local_start=int(steps[index]),record_id=f'audit-official-rows-{a.task}-{ordinal:03}',record_index=ordinal,split='released_full_dataset_rows',sample_seed=42)
                    source['identity']={key:row[key] for key in ['record_id','episode','global_start','global_goal','environment_seed','horizon']}
                original_raw_info=baseline.raw_info
                first=[True];initial_image=None
                if a.factor=='dataset_initial_image':
                    initial_image=np.ascontiguousarray(data['pixels'][int(row['global_start'])])
                    def dataset_first(native,rgb,goal):
                        if first[0]:rgb=initial_image;first[0]=False
                        return original_raw_info(native,rgb,goal)
                    baseline.raw_info=dataset_first
                try:
                    with torch.inference_mode():
                        result=baseline.episode(a,H,ordinal,'clean',False,protocol,row,source,api,ew,context,native,counter,cached_root)
                    result.update(factor=a.factor,main_comparison_H=25,actual_goal_frame_offset=H)
                    if initial_image is not None and result['decisions']:
                        file=root/a.task/f'H{H}'/'released_raw'/f'q{ordinal:03}_clean.npz'
                        with np.load(file) as z:arrays={k:z[k] for k in z.files}
                        with torch.inference_mode():arrays['score_current'][0]=native.encode_rgb(initial_image)[0].cpu().numpy()
                        np.savez_compressed(file,**arrays)
                    stream.write(json.dumps(result)+'\n');stream.flush()
                    if (ordinal-a.start+1)%8==0:print(json.dumps({'task':a.task,'factor':a.factor,'completed':ordinal-a.start+1,'expected':a.stop-a.start}),flush=True)
                finally:baseline.raw_info=original_raw_info
        finally:counter.handle.remove()
print(json.dumps({'task':a.task,'factor':a.factor,'complete':True}),flush=True)
