"""Summarize complete checks, without claiming pending runs passed."""
from pathlib import Path
import csv,json
import numpy as np
R=Path(__file__).parent
out=R/'lewm_checks';out.mkdir(exist_ok=True)
reference=[];parity=[]
for task in ['cube','pusht','reacher','tworoom']:
    for seed in [42,43,44]:
        dirs={m:R/(m+'_runs')/f'{task}_seed{seed}_n50' for m in ['official','wrapper']}
        reports={m:json.loads((d/'run.json').read_text()) for m,d in dirs.items() if (d/'run.json').exists()}
        ref=reports.get('official',{})
        if ref.get('status')!='complete':continue
        reference.append(dict(task=task,seed=seed,seed_source='released config' if seed==42 else 'predeclared audit repeat',episodes=50,successes=int(sum(ref['metrics']['episode_successes'])),success_rate=ref['metrics']['success_rate'],episode_ids=json.dumps(ref['eval_episodes']),start_steps=json.dumps(ref['eval_start_idx']),goal_steps=json.dumps([int(s)+24 for s in ref['eval_start_idx']]),configured_goal_offset=25,actual_frame_offset=24,budget=50,solver_calls=ref['solver_calls']))
        wrapper=reports.get('wrapper',{})
        if wrapper.get('status')!='complete':continue
        assert ref['eval_episodes']==wrapper['eval_episodes'] and ref['eval_start_idx']==wrapper['eval_start_idx']
        x=np.asarray(ref['metrics']['episode_successes'],bool);y=np.asarray(wrapper['metrics']['episode_successes'],bool)
        with np.load(dirs['official']/'observed_runtime.npz') as ra,np.load(dirs['wrapper']/'observed_runtime.npz') as wa:
            ax=np.stack([ra[f'actions_{i}'] for i in range(50)]);ay=np.stack([wa[f'actions_{i}'] for i in range(50)])
            maxerror=float(np.abs(ax-ay).max());equal=bool(np.array_equal(ax,ay))
            input_equal=all(np.array_equal(ra[f'input_{k}_0'],wa[f'input_{k}_0']) for k in ['pixels','goal'])
        delta=int(y.sum())-int(x.sum());agreement=float((x==y).mean())
        parity.append(dict(task=task,seed=seed,episodes=50,official_successes=int(x.sum()),wrapper_successes=int(y.sum()),success_count_difference=delta,episode_agreement=agreement,input_images_equal=input_equal,actions_exactly_equal=equal,max_action_abs_difference=maxerror,predeclared_parity_pass=abs(delta)<=3 and agreement>=.9,checkpoint_tensors_equal=wrapper['wrapper_checkpoint_tensors_equal'],scaler_equal=wrapper['wrapper_scaler_equal']))
for name,rows in [('official_repro.csv',reference),('parity.csv',parity)]:
    if rows:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={'official_complete':len(reference),'parity_complete':len(parity),'expected':12,'parity_failures':[x for x in parity if not x['predeclared_parity_pass']]}
(out/'status.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
