"""LeWM's complete 25-action planner with main-study action projection and starts."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import common_rollout as runtime
import construction_fresh as construction
from observed_target import query_seed
from memory_mask import apply as apply_memory_mask

BASE = Path('runtime/ap/fresh_target_interface_study_v2')

@torch.inference_mode()
def episode(task, row, ordinal, source, p, api, ew, context, native, counter):
    env, obs, info, true_goal, setup, cached = construction.construct(ew, api, context, native.scaler, task, row, 'confirm', BASE, p, source)
    low, high = np.asarray(env.action_space.low), np.asarray(env.action_space.high)
    bounds = native.scaler.transform(np.stack([low, high])).astype(np.float32)
    lower, upper = [torch.as_tensor(np.tile(v, 5), device=native.device) for v in bounds]
    original_cost = native.direct_cost.get_cost
    projection_count = 0
    def projected_cost(info, candidates):
        nonlocal projection_count
        effective = torch.minimum(torch.maximum(candidates, lower), upper)
        projection_count += int((effective != candidates).any(-1).sum())
        return original_cost(info, effective)
    native.direct_cost.get_cost = projected_cost
    native.begin(row, dict(initial_rgb=cached['post_rgb'], goal_rgb=cached['goal_rgb'], goal=cached['goal']))
    seed = query_seed(task, ordinal)
    torch.manual_seed(seed)
    native.solver.torch_gen.manual_seed(seed)
    actions, decisions, trajectory, encoded_states = [], [], [setup['post_state']], []
    success = bool(setup['prelude']['success'])
    stopped = bool(setup['prelude']['terminal_without_success'])
    calls = blocks = 0
    started = time.perf_counter()
    budget = int(p['studies'][task]['budget'])
    try:
        while not success and not stopped and len(actions) < budget:
            step = len(actions)
            counter.reset()
            projection_count = 0
            raw, details, inputs = native.choose(None, step)
            assert counter.calls == 150 and counter.candidates == 45000
            calls += counter.calls
            blocks += counter.candidates
            effective = np.clip(raw, low, high).astype(np.float32)
            assert np.isfinite(effective).all() and (effective >= low).all() and (effective <= high).all()
            encoded_states.append(np.asarray(inputs['current'], np.float32))
            details.update(candidate_clipping=True, action_clipping=True, projected_candidate_blocks=projection_count, current_latent=np.asarray(inputs['current']).tolist(), incoming_plan_horizon=25)
            for i, action in enumerate(effective):
                obs, _, term, trunc, info = env.step(action.copy())
                success = bool(ew.runtime._success_now(task, obs, info, true_goal, False, False))
                stopped = bool((term or trunc) and not success)
                actions.append(action.copy())
                trajectory.append(ew.physical_state(task, env))
                continuing = not success and not stopped and len(actions) < budget
                native.completed_primitive(env, i, continuing)
                if not continuing:
                    break
            decisions.append(dict(t=step, executed_actions=len(actions)-step, **details))
        compact = dict(study='unified_lewm_25_20260923', task=task, identity=row, query_ordinal=ordinal, condition=source['condition'], arm='lewm_25', completed=True, success=success, prelude=setup['prelude'], primitive_steps=len(actions), decisions=len(decisions), predictor_calls=calls, predictor_candidate_blocks=blocks, optimizer_seed=seed, elapsed_seconds=time.perf_counter()-started, budget=budget, horizon=row['horizon'])
        trace = dict(compact=compact, construction=setup, decisions=decisions, trajectory=trajectory, final_state=ew.physical_state(task,env), true_goal=np.asarray(true_goal).tolist())
        arrays = dict(actions=np.asarray(actions,np.float32).reshape(-1,native.action_dim), score_current=np.asarray(encoded_states,np.float32).reshape(-1,192), goal_encoding=np.asarray(cached['goal'],np.float32), scoring_goal_encoding=native.encode_rgb(np.ascontiguousarray(cached['goal_rgb']))[0].detach().cpu().numpy(), raw_action_bounds=np.stack([low,high]))
        trace['scoring_goal_encoding_source']='MAIN frozen FP32 scoring encoder, singleton target image'
        return trace, arrays
    finally:
        native.direct_cost.get_cost = original_cost
        env.close()

def main(a):
    started=time.perf_counter()
    p=runtime.read(BASE/'protocol.json')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    _,_,_,_,api,ew,context,native,_=runtime.load_runtime(a.task,'confirm')
    membership=apply_memory_mask(context,p,a.task)
    sources={(s['identity']['record_id'],s['condition']):s for s in runtime.read(BASE/'source_manifest.json')['rows'] if s['task']==a.task}
    counter=runtime.module('long_counter',runtime.REFERENCE).WorkCounter(native.model)
    output=a.output/a.task/f'q{a.start:03}-{a.stop:03}'
    output.mkdir(parents=True,exist_ok=True)
    protocol=dict(task=a.task,base_study=str(BASE),memory_boundary=membership,query_count=128,conditions=p['conditions'],planner='LeWM 25-action CEM, 300 candidates, 30 iterations, 30 elites, sigma 1, retained four-block warm-start, rolling prediction history',action_mapping='Project each normalized candidate onto normalized raw-action Box before official get_cost; project selected de-normalized actions to raw Box before execution.',differences_from_5_action_cem='Horizon, warm start, initial variance, solver mean selection, and rolling prediction history; complete planner comparison.',visible_gpu=os.environ['CUDA_VISIBLE_DEVICES'])
    runtime.write(output/'protocol.json',protocol)
    completed=0
    try:
        existing={}
        if (output/'outcomes.jsonl').exists():
            for line in (output/'outcomes.jsonl').open():
                prior=json.loads(line);existing[(prior['query_ordinal'],prior['condition'])]=prior
        with (output/'outcomes.jsonl').open('a') as records:
            assignments=[(o,c) for o in range(a.start,a.stop) for c in p['conditions']]
            for ordinal,condition in assignments:
                row=p['studies'][a.task]['confirm'][ordinal]
                if (ordinal,condition) in existing:
                    completed+=1;continue
                torch.cuda.reset_peak_memory_stats()
                trace,arrays=episode(a.task,row,ordinal,sources[(row['record_id'],condition)],p,api,ew,context,native,counter)
                trace['compact']['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
                stem=output/'traces'/f"{row['record_id']}_{condition}"
                stem.parent.mkdir(parents=True,exist_ok=True)
                runtime.write(stem.with_suffix('.json'),trace)
                np.savez_compressed(stem.with_suffix('.npz'),**arrays)
                records.write(json.dumps(trace['compact'])+'\n');records.flush()
                completed+=1
                print(json.dumps(dict(task=a.task,mode=a.mode,completed=completed,expected=len(assignments),elapsed_seconds=round(time.perf_counter()-started,1))),flush=True)
    finally:
        counter.handle.remove()
    runtime.write(output/'summary.json',dict(completed=True,outcomes=completed,peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_seconds=time.perf_counter()-started))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=('cube','pusht','reacher','tworoom'),required=True)
    p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=128);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=('smoke','evaluate'),required=True)
    main(p.parse_args())
