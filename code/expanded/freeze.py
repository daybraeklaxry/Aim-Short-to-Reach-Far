"""Freeze all five policies and fixed identities before any fresh outcome."""
import argparse,json,shutil
from pathlib import Path
from datetime import datetime,timezone
BASE=Path('external/le-wm/research')
ORAL=BASE/'ap_rank_oral_revision_20260913'
CONF=BASE/'ap_rank_confirmatory_20260912/confirmation'
PREP=ORAL/'fresh_query_preparation_20260914'
TASKS=('cube','pusht','reacher','tworoom')
ARMS=['direct','ap_observed','ap_final','gaussian_observed','gaussian_final']
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main(a):
    assert not a.output.exists()
    old=read(CONF/'protocol.json');c=read(PREP/'cohort.json');manifest=read(PREP/'source_manifest.json')
    profile=a.stage=='profile';conditions=['clean','prefix_a'] if profile else ['clean','prefix_a','prefix_b']
    if profile:
        studies={};sources=[]
        for task in TASKS:
            root=Path(old['development_evidence'][task]['development_protocol']).parent
            previous=read(root/'protocol.json');study=previous['studies'][task]
            rows=study['profile'][:old['development_evidence'][task]['queries']];assert len(rows)==1
            studies[task]=dict(study,confirm=rows)
            ids={r['record_id'] for r in rows}
            sources.extend(s for s in read(root/'source_manifest.json')['rows'] if s['task']==task and
                s['stage']=='profile' and s['condition'] in conditions and s['identity']['record_id'] in ids)
        assert len(sources)==8
        manifest=dict(rows=sources,scope='Already exposed development episodes only; no fresh queries or mask')
    else:
        studies=c['studies']
        assert c['environment_steps']==c['model_calls']==c['outcomes_read']==0
        assert manifest['environment_steps']==manifest['policy_calls']==manifest['model_calls']==0
        expected={(t,r['record_id'],k) for t in TASKS for r in studies[t]['confirm'] for k in conditions}
        actual={(s['task'],s['identity']['record_id'],s['condition']) for s in manifest['rows']}
        assert len(expected)==len(actual)==len(manifest['rows'])==1536 and expected==actual
        for task in TASKS:
            query={r['episode'] for r in studies[task]['confirm']}
            assert len(query)==128 and not query&set(c['bank_train_episodes'][task])
    stamp=datetime.now(timezone.utc).isoformat()
    p=dict(study='fresh_target_interface_20260914',frozen_utc=stamp,technical_profile=profile,
        native_root=old['native_root'],native_frozen_utc=old['native_frozen_utc'],studies=studies,
        arms=ARMS,main_method='ap_observed',conditions=conditions,
        scheduling='Formal whole query ordinal modulo shards; all conditions and arms stay with the same cache creator. Profile old query-condition shards.',
        action_chunk=5,candidates=8,expected_outcomes=(40 if profile else 7680),
        bank_train_episodes=c['bank_train_episodes'],removed_memory_episodes=c['removed_memory_episodes'],
        source_manifest_frozen_utc=manifest.get('frozen_utc'),
        native_numeric_source='Frozen native Adapter and scaler; all arm predictions float32',
        numeric_protocol=dict(dtype='float32',matmul_precision='highest',cuda_matmul_tf32=False,cudnn_tf32=False),
        action_domain='Global raw environment Box P for every fixed TRAIN prelude, discrete candidate, Gaussian prior/proposal, and execution; unprojected elite update retained.',
        targets=dict(observed='Native E of original leading TRAIN source+5 pixels',final='Native E of supplied row.global_goal pixels'),
        gaussian=dict(initial_mean_and_incumbent='fixed standardized zero projected through raw Box; never recorded action initialization',
            sigma=1/3,iterations=30,samples=300,elites=30,std_floor=1e-5,strict_final_prior_mean=True,
            query_seed='PushT:26091331+1000*ordinal; Cube/Reacher/TwoRoom:26091351+1000000*taskindex+1000*ordinal; same for both target arms',
            dynamics_calls_per_decision=31,candidate_blocks_per_decision=9002),
        ap=dict(dynamics_calls_per_decision=1,candidate_blocks_per_decision=8,argmin='first minimum; no rescore or tolerance ties'),
        direct=dict(dynamics_calls_per_decision=0,candidate_blocks_per_decision=0,action='P(top1 retrieved raw block)'),
        prefix='Original preassigned TRAIN sources; common raw Box projection; no replacement; outside unchanged policy budget',
        observed_projection_boundary='Original recorded successor need not equal physical endpoint of projected recorded action.',
        dynamics_source_predictions=0,old_policy_outcomes_reused=0,new_training=False,
        fresh_selection_receipt=str(PREP/'decision_receipt.json'),
        split_scope=c['scope'],preprocessing_rule=c['preprocessing_rule'],
        planned_contrasts=dict(primary=[['ap_observed','ap_final'],['gaussian_observed','gaussian_final']],
            secondary=[['ap_observed','direct']],conditions=['clean','prefix_mean'],
            macro='Equal weight across four tasks; prefix mean inside each episode',
            bootstrap_resamples=10000,bootstrap_seed=20260913,
            bootstrap_unit='Independently within each task, sample128 episodes; same indices for all arms and clean/prefixA/prefixB',
            intervals='Six marginal pointwise95% percentile CIs; no p values/Holm/joint gate',
            task_tables='All5 arms all4 tasks clean/A/B/prefixmean and task-level pointwise intervals'),
        finality='AP-observed fixed main implementation before fresh outcomes; no outcome-based changes or new variants')
    a.output.mkdir(parents=True)
    write(a.output/'protocol.json',p);write(a.output/'source_manifest.json',manifest)
    write(a.output/'cohort.json',c)
    code=a.output/'frozen_code';shutil.copytree(Path(__file__).resolve().parent,code,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ORAL/'science/fresh_five_arm_statistics_frozen.md',a.output/'statistics_protocol.md')
    print(json.dumps(dict(output=str(a.output),expected_outcomes=p['expected_outcomes'],frozen_utc=stamp,environment_steps=0)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['profile','confirm'],required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
