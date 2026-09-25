"""Only change the released policy's execution horizon; all main queries fixed."""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import argparse,copy,json,sys,time
import torch
R=Path(__file__).parent
sys.path.insert(0,str(R.parent/'baseline'))
import official_baseline as baseline

p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--stop',type=int,required=True);p.add_argument('--conditions',default='clean,prefix_a,prefix_b');p.add_argument('--profile',action='store_true')
a=p.parse_args()
root=R/('replan5_profile' if a.profile else 'replan5_results')
root.mkdir(exist_ok=True);baseline.OUT=root
torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
original_policy=baseline.policy_for
def replan_policy(native,env,seed,project=False):
    policy=original_policy(native,env,seed,project)
    policy.cfg=replace(policy.cfg,receding_horizon=1)
    policy.set_env(policy.env)
    return policy
baseline.policy_for=replan_policy
protocol=baseline.runtime.read(baseline.BASE/'protocol.json');H=protocol['studies'][a.task]['horizon']
conditions=a.conditions.split(',');assert all(c in protocol['conditions'] for c in conditions)
_,_,_,_,api,ew,context,native,_=baseline.runtime.load_runtime(a.task,'confirm')
sources={(s['identity']['record_id'],s['condition']):s for s in baseline.runtime.read(baseline.BASE/'source_manifest.json')['rows'] if s['task']==a.task}
counter=baseline.runtime.module('replan5_counter',baseline.runtime.REFERENCE).WorkCounter(native.model)
records=root/f'{a.task}_{a.start:03}-{a.stop:03}.jsonl'
assert not records.exists(),str(records)
started=time.time()
with records.open('w') as stream:
    try:
        for ordinal in range(a.start,a.stop):
            row=copy.deepcopy(protocol['studies'][a.task]['confirm'][ordinal])
            for condition in conditions:
                with torch.inference_mode():
                    result=baseline.episode(a,H,ordinal,condition,False,protocol,row,sources[(row['record_id'],condition)],api,ew,context,native,counter,baseline.BASE)
                result.update(variant='LeWM (5-action replanning)',planned_actions_per_decision=25,maximum_executed_actions_per_decision=5)
                # The reused writer's per-decision metadata hard-codes25; correct
                # only that execution limit in this new variant's output record.
                detail=root/a.task/f'H{H}'/'released_raw'/f'q{ordinal:03}_{condition}.json'
                trace=json.loads(detail.read_text());trace['compact']=result
                for decision in trace['decisions']:
                    assert decision['executed_actions']<=5
                    decision['maximum_executed_actions']=5
                detail.write_text(json.dumps(trace,indent=2))
                stream.write(json.dumps(result)+'\n');stream.flush();print(json.dumps(result),flush=True)
    finally:counter.handle.remove()
print(json.dumps({'task':a.task,'elapsed_seconds':time.time()-started,'complete':True}),flush=True)
