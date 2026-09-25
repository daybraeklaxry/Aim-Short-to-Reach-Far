"""Same main queries/actions: simulator endpoints, then a shared Direct continuation."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import common_rollout as runtime
import construction_fresh as construction
from action_domain import prepare_candidates
from memory_mask import apply as apply_memory_mask
from target_scorers import TargetRank
from exact_effect import summarize_effects as base_summarize, SELECTORS as BASE_SELECTORS
from run_suite import learned
ROOT=Path('runtime/ap/evidence_strengthening_20260923/results_v2')
SELECTORS=BASE_SELECTORS+('predictor_mlp','exact_mlp')

def summarize_effects(np,arrays):
    result=base_summarize(np,arrays)
    result['selected_ranks'].update(predictor_mlp=int(arrays['predictor_mlp_scores'].argmin()),exact_mlp=int(arrays['exact_mlp_scores'].argmin()))
    result['continuation_ranks']=sorted(set(result['selected_ranks'].values()))
    result['continuation_selector_groups']={str(i):[s for s in SELECTORS if result['selected_ranks'][s]==i] for i in result['continuation_ranks']}
    return result

BASE=Path('runtime/ap/fresh_target_interface_study_v2')

def factory(api,ew,context,native,task,row,p,source):
    return construction.construct(ew,api,context,native.scaler,task,row,'confirm',BASE,p,source)

def same_start(ew,task,env,goal,setup,cached,reference,reference_arrays):
    assert ew.equal_state(ew.physical_state(task,env),reference['post_state'])
    assert construction.extra_state(ew,task,env)==reference['post_extra']
    assert setup['prelude']==reference['prelude']
    np.testing.assert_array_equal(goal,reference['true_goal'])
    for key in ['post_rgb','goal_rgb','initial_sources','initial_raw','initial_normalized']:
        np.testing.assert_array_equal(cached[key],reference_arrays[key])

@torch.inference_mode()
def evaluate(task,row,ordinal,source,p,api,ew,context,native,scorer,cached_result=None,cached_arrays=None):
    make=lambda:factory(api,ew,context,native,task,row,p,source)
    env,obs,info,goal,setup,cached=make()
    low,high=env.action_space.low.copy(),env.action_space.high.copy()
    env.close()
    meta=dict(task=task,identity=row,query_ordinal=ordinal,condition=source['condition'],prelude=setup['prelude'],completed=True)
    if setup['prelude']['category']!='policy_entered':
        result=dict(meta,entered_policy=False,fixed_five_diagnostic_eligible=False,no_candidate_clipping=True,prediction_diagnostics=None,selected_ranks={k:None for k in SELECTORS},continuations={k:dict(success=bool(setup['prelude']['success']),primitive_steps=0,first_rank=None) for k in SELECTORS})
        return result,{}
    if cached_arrays is not None:
        arrays={k:v.copy() for k,v in cached_arrays.items()}
        sources=cached['initial_sources'];raw=cached['initial_raw']
        effective,normalized,projection=prepare_candidates(raw,low,high,native.scaler)
        np.testing.assert_array_equal(arrays['candidate_sources'],sources)
        np.testing.assert_array_equal(arrays['candidate_raw_actions'],effective)
        np.testing.assert_array_equal(arrays['current_latent'],scorer.native.encode_rgb(cached['post_rgb'])[0].cpu().numpy())
        branches=cached_result['branches'];step_counts=arrays['executed_primitive_count']
        success,terms,truncs=arrays['success'],arrays['terminated'],arrays['truncated']
    else:
        sources=cached['initial_sources']
        raw=cached['initial_raw']
        effective,normalized,projection=prepare_candidates(raw,low,high,native.scaler)
        scorer.set_goal_record(row['global_goal'])
        local_rank,_,local=scorer.choose('ap_observed',cached['post_rgb'],sources,normalized)
        far_rank,_,far=scorer.choose('ap_final',cached['post_rgb'],sources,normalized)
        np.testing.assert_array_equal(local['live_predictions'],far['live_predictions'])
        # This check ties every query to the actual main-study prediction and action.
        for arm,scoring,rank in [('ap_observed',local,local_rank),('ap_final',far,far_rank)]:
            stem=BASE/'evaluation/traces'/task/f"confirm_{row['record_id']}_{source['condition']}_{arm}"
            old=runtime.read(stem.with_suffix('.json'))
            assert old['decisions'][0]['selected_rank']==rank
            with np.load(stem.with_suffix('.npz')) as prior:
                for key in ['score_current','score_target','live_predictions','candidate_scores']:
                    np.testing.assert_array_equal(scoring[key],prior[key][0],err_msg=f'{task}/{ordinal}/{arm}/{key}')
                np.testing.assert_array_equal(effective,prior['candidate_effective_raw_actions'][0])
        ends=[];branches=[]
        step_counts=np.zeros(8,np.int64)
        success,terms,truncs=(np.zeros((8,5),bool) for _ in range(3))
        for rank in range(8):
            env,obs,info,goal,replay,replay_arrays=make()
            try:
                same_start(ew,task,env,goal,replay,replay_arrays,setup,cached)
                for t,action in enumerate(effective[rank]):
                    obs,_,term,trunc,info=env.step(action.copy())
                    won=bool(ew.runtime._success_now(task,obs,info,goal,False,False))
                    step_counts[rank]+=1
                    success[rank,t],terms[rank,t],truncs[rank,t]=won,bool(term),bool(trunc)
                    if won or term or trunc:break
                ends.append(native.encode_rgb(np.ascontiguousarray(env.render()))[0].cpu().numpy())
                branches.append(dict(rank=rank,primitive_steps=int(step_counts[rank]),endpoint_state=ew.physical_state(task,env),endpoint_extra=construction.extra_state(ew,task,env)))
            finally:
                env.close()
                torch.cuda.empty_cache()
        recorded_starts=np.stack([scorer.encoding(i).cpu().numpy() for i in sources])
        recorded_ends=np.stack([scorer.encoding(int(i)+5).cpu().numpy() for i in sources])
        arrays=dict(candidate_sources=sources.copy(),original_candidate_actions=raw.copy(),candidate_raw_actions=effective.copy(),candidate_normalized_actions=normalized.copy(),current_latent=local['score_current'],local_target=local['score_target'],far_target=far['score_target'],recorded_start_latents=recorded_starts,recorded_end_latents=recorded_ends,predicted_endpoints=local['live_predictions'],exact_endpoints=np.stack(ends),learned_local_scores=local['candidate_scores'],learned_far_scores=far['candidate_scores'],executed_primitive_count=step_counts,complete_five_steps=step_counts==5,success=success,terminated=terms,truncated=truncs)
        arrays['recorded_delta_endpoints']=arrays['current_latent'][None]+recorded_ends-recorded_starts
        arrays['recorded_delta_local_scores']=np.square(arrays['recorded_delta_endpoints']-arrays['local_target']).sum(-1)
        arrays['exact_local_scores']=np.square(arrays['exact_endpoints']-arrays['local_target']).sum(-1)
        arrays['exact_far_scores']=np.square(arrays['exact_endpoints']-arrays['far_target']).sum(-1)
    tensor=lambda x:torch.as_tensor(x,device=native.device,dtype=torch.float32)
    mlp_target=scorer.target_model(tensor(arrays['current_latent'])[None],tensor(arrays['far_target'])[None],tensor([row['horizon']]))[0].cpu().numpy()
    arrays['mlp_target']=mlp_target
    arrays['predictor_mlp_scores']=np.square(arrays['predicted_endpoints']-mlp_target).sum(-1)
    arrays['exact_mlp_scores']=np.square(arrays['exact_endpoints']-mlp_target).sum(-1)
    result=dict(meta,entered_policy=True,action_projection=projection,no_candidate_clipping=projection['coordinates']==0,branches=branches,**summarize_effects(np,arrays))
    # All selectors differ only in their first block; later steps use the same Direct rule.
    continuations={}
    previous_by_rank={} if cached_result is None else {v['first_rank']:v for v in cached_result['continuations'].values()}
    for rank in result['continuation_ranks']:
        if rank in previous_by_rank:
            continuations[str(rank)]=previous_by_rank[rank]
            continue
        env,obs,info,goal,replay,replay_arrays=make()
        actions=[];won=False;stopped=False
        try:
            same_start(ew,task,env,goal,replay,replay_arrays,setup,cached)
            for t,action in enumerate(effective[rank]):
                obs,_,term,trunc,info=env.step(action.copy())
                won=bool(ew.runtime._success_now(task,obs,info,goal,False,False));stopped=bool((term or trunc) and not won)
                actions.append(action.copy())
                assert (won,bool(term),bool(trunc))==(bool(success[rank,t]),bool(terms[rank,t]),bool(truncs[rank,t]))
                if won or stopped:break
            assert len(actions)==step_counts[rank]
            assert ew.equal_state(ew.physical_state(task,env),branches[rank]['endpoint_state'])
            assert construction.extra_state(ew,task,env)==branches[rank]['endpoint_extra']
            while not won and not stopped and len(actions)<p['studies'][task]['budget']:
                current=ew.runtime._encode(context,np.ascontiguousarray(env.render())[None]).astype(np.float32)
                h=max(5,int(row['horizon'])-len(actions))
                found,_,_=context['bank'].nearest_topk(current,cached['goal'].reshape(1,-1),h,8)
                candidate_raw=ew.runtime._raw_actions(context,np.asarray(found[0],np.int64),5).astype(np.float32)
                chunk=np.clip(candidate_raw[0],low,high).astype(np.float32)
                for action in chunk:
                    obs,_,term,trunc,info=env.step(action.copy())
                    won=bool(ew.runtime._success_now(task,obs,info,goal,False,False));stopped=bool((term or trunc) and not won)
                    actions.append(action.copy())
                    if won or stopped or len(actions)>=p['studies'][task]['budget']:break
                torch.cuda.empty_cache()
            continuations[str(rank)]=dict(success=won,primitive_steps=len(actions),first_rank=rank)
            arrays[f'continuation_actions_rank{rank}']=np.asarray(actions,np.float32)
            if rank==0:
                stem=BASE/'evaluation/traces'/task/f"confirm_{row['record_id']}_{source['condition']}_direct"
                old=runtime.read(stem.with_suffix('.json'))['compact']
                assert won==old['success'] and len(actions)==old['primitive_steps']
                with np.load(stem.with_suffix('.npz')) as prior:np.testing.assert_array_equal(arrays[f'continuation_actions_rank{rank}'],prior['actions'])
        finally:
            env.close()
            torch.cuda.empty_cache()
    result['continuations']={name:continuations[str(rank)] for name,rank in result['selected_ranks'].items()}
    result['main_prediction_replay_exact']=True
    result['direct_continuation_replay_exact']=True
    return result,arrays

def main(a):
    start=time.perf_counter();p=runtime.read(BASE/'protocol.json')
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    _,_,_,_,api,ew,context,native,_=runtime.load_runtime(a.task,'confirm')
    boundary=apply_memory_mask(context,p,a.task)
    sources={(s['identity']['record_id'],s['condition']):s for s in runtime.read(BASE/'source_manifest.json')['rows'] if s['task']==a.task}
    job=next(j for j in runtime.read(ROOT/'queue-manifest.json')['jobs'] if j['id']==a.job_id);cfg=job['config']
    output=ROOT/'jobs'/job['id'];output.mkdir(parents=True,exist_ok=True)
    assignments=[(i,c) for i in range(cfg['start'],cfg['stop']) for c in p['conditions']]
    if a.probe:assignments=assignments[:3]
    runtime.write(output/'protocol.json',dict(task=a.task,base_study=str(BASE),memory_boundary=boundary,conditions=p['conditions'],selectors=SELECTORS,selector_labels=dict(direct='Direct',learned_far='Predictor-Final',learned_local='Predictor-Observed',predictor_mlp='Predictor-Learned',exact_far='Simulator-Final',exact_local='Simulator-Observed',exact_mlp='Simulator-Learned',recorded_delta_local='Displacement-Observed'),action_mapping='Same clipping to the action bounds as main study, before prediction and physical execution.',diagnostic_population='All eight candidates complete five primitives; additionally report the no-candidate-clipping subset.',continuation='Selector changes only the first block, then identical live-state Direct retrieval to the original total budget.',replication='Exact main first-decision predictions/ranks and full Direct actions are checked for every policy-entered query.',mode='MAIN unified diagnostic'))
    prior={}
    if (output/'outcomes.jsonl').exists():
        for line in (output/'outcomes.jsonl').open():
            r=__import__('json').loads(line);prior[(r['query_ordinal'],r['condition'])]=r
    with ew.h5py.File(ew.v6._data_path(context),'r') as handle,(output/'outcomes.jsonl').open('a') as records:
        scorer=TargetRank(native,handle)
        scorer.target_model,selected=learned(a.task,native.device)
        for count,(ordinal,condition) in enumerate(assignments,1):
            if (ordinal,condition) in prior:continue
            row=p['studies'][a.task]['confirm'][ordinal]
            oldstem=ROOT.parent/'exact_smoke'/a.task/'traces'/f"{row['record_id']}_{condition}"
            cached_result=runtime.read(oldstem.with_suffix('.json')) if oldstem.with_suffix('.json').exists() else None
            cached_arrays=None
            if cached_result is not None:
                with np.load(oldstem.with_suffix('.npz')) as z:cached_arrays={k:z[k] for k in z.files}
            torch.cuda.reset_peak_memory_stats()
            result,arrays=evaluate(a.task,row,ordinal,sources[(row['record_id'],condition)],p,api,ew,context,native,scorer,cached_result,cached_arrays)
            result.update(allocator_cleanup='unused cached tensors released after each Direct decision and simulator branch',peak_reserved_bytes=torch.cuda.max_memory_reserved(),target_checkpoint=selected['checkpoint'],reused_initial_effects=str(oldstem) if cached_arrays is not None else None)
            stem=output/'traces'/f"{row['record_id']}_{condition}";stem.parent.mkdir(exist_ok=True)
            runtime.write(stem.with_suffix('.json'),result);np.savez_compressed(stem.with_suffix('.npz'),**arrays)
            records.write(json.dumps(result)+'\n');records.flush()
            print(json.dumps(dict(task=a.task,mode="main",completed=count,expected=len(assignments),elapsed_seconds=round(time.perf_counter()-start,1))),flush=True)
    runtime.write(output/('probe.json' if a.probe else 'summary.json'),dict(completed=True,outcomes=len(assignments),main_prediction_replay_exact=True,direct_continuation_replay_exact=True,elapsed_seconds=time.perf_counter()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=('cube','pusht','reacher','tworoom'),required=True)
    p.add_argument('--job-id',required=True);p.add_argument('--probe',action='store_true')
    main(p.parse_args())
