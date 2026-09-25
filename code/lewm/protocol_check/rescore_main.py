"""Apply both physical success predicates to saved H25 LeWM states (no rollout)."""
from pathlib import Path
import csv,json,math
import numpy as np
import torch
import gymnasium as gym
import mujoco
import stable_worldmodel

R=Path(__file__).parent
BASE=R.parent/'baseline/official_results'
out=R/'lewm_checks';out.mkdir(exist_ok=True)
rows=[];cube=None
for task in ['cube','pusht','reacher','tworoom']:
    if task=='cube':
        cube=gym.make('swm/OGBCube-v0',env_type='single',ob_type='states',mode='data_collection',visualize_info=False,terminate_at_goal=True,width=224,height=224,disable_env_checker=True)
        cube.reset(seed=0,options={'variation':[]})
    for ordinal in range(128):
        path=BASE/task/'H25/released_raw'/f'q{ordinal:03}_clean.json'
        trace=json.loads(path.read_text());goal=np.asarray(trace['true_goal'],np.float64)
        paper=[];released=[]
        for state in trace['trajectory']:
            if task=='cube':
                e=cube.unwrapped
                mujoco.mj_setState(e.model,e.data,np.asarray(state['integration'],np.float64),mujoco.mjtState.mjSTATE_INTEGRATION)
                q=np.empty(e.model.nq);mujoco.mj_getState(e.model,e.data,q,mujoco.mjtState.mjSTATE_QPOS)
                target=goal[19:22]/10+np.asarray([.425,0,0])
                # Direct physical condition vs the released environment method.
                x=bool(np.linalg.norm(q[14:17]-target)<=.04)
                y=bool(all(e._compute_successes()))
            elif task=='pusht':
                v=np.asarray(state['state'],np.float64)
                angle=abs(float(goal[4])-float(v[4]))%(2*math.pi)
                x=bool(np.linalg.norm(goal[:4]-v[:4])<20 and min(angle,2*math.pi-angle)<math.pi/9)
                diff=np.abs(goal[4]-v[4]);y=bool(np.linalg.norm(goal[:4]-v[:4])<20 and np.minimum(diff,2*np.pi-diff)<np.pi/9)
            elif task=='reacher':
                q=np.asarray(state['physics'],np.float64)[:2]
                x=bool(np.all(np.abs(q.astype(np.float32)-goal[:2])<.05))
                y=bool(np.all(np.abs(q-goal[:2])<.05))
            else:
                v=np.asarray(state['state'],np.float32)
                distance=float(torch.norm(torch.as_tensor(v[:2])-torch.as_tensor(v[2:4])))
                x=y=distance<16
            paper.append(x);released.append(y)
        expected=bool(trace['compact']['success'])
        row={'task':task,'query_ordinal':ordinal,'states_checked':len(paper),'stored_success':expected,'paper_rescored_success':any(paper),'released_predicate_success':any(released),'paper_matches_record':expected==any(paper),'per_state_predicate_disagreements':sum(x!=y for x,y in zip(paper,released)),'initial_success':paper[0]}
        rows.append(row)
    if cube is not None:cube.close();cube=None
with (out/'success_rescoring.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for task in ['cube','pusht','reacher','tworoom']:
    group=[r for r in rows if r['task']==task]
    summary.append({'task':task,'episodes':len(group),'states_checked':sum(r['states_checked'] for r in group),'stored_successes':sum(r['stored_success'] for r in group),'rescored_successes':sum(r['paper_rescored_success'] for r in group),'released_predicate_successes':sum(r['released_predicate_success'] for r in group),'record_disagreements':sum(not r['paper_matches_record'] for r in group),'per_state_predicate_disagreements':sum(r['per_state_predicate_disagreements'] for r in group),'initial_successes':sum(r['initial_success'] for r in group)})
(out/'success_rescoring.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
