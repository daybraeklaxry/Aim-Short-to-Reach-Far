"""Metadata helpers for the explicitly scoped fresh control-query cohort."""
import datetime
import json
from pathlib import Path
import numpy as np

TASKS=('cube','pusht','reacher','tworoom')
SETTINGS={'cube':(150,150),'pusht':(140,280),'reacher':(150,300),'tworoom':(100,200)}
CONDITIONS=('clean','prefix_a','prefix_b')

def read(path):
    path=Path(path)
    path.stat()  # Inspect existence/size before reading this explicit metadata file.
    return json.loads(path.read_text(encoding='utf-8-sig'))

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def generator(seed,task_index):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,task_index])))

def episode_train(lengths,source,task):
    if task=='pusht':
        order=np.random.default_rng(3072).permutation(len(lengths))
        return np.asarray(order[:int(.9*len(lengths))],np.int64)
    return np.asarray(source['split']['train'],np.int64)

def eligible_episodes(task,source,lengths,horizon,excluded,train):
    if task=='pusht':
        candidates=None
        eligible=sorted(int(e) for e in train if e not in excluded and lengths[e]>horizon)
    else:
        candidates={int(r['episode']):r for r in source['by_horizon'][str(horizon)]['validation']['32']['rows']}
        heldout=set(source['split']['validation'])
        eligible=sorted(e for e in candidates if e in heldout and e not in excluded)
        assert not set(eligible)&set(train)
    return eligible,candidates

def valid_prefix_starts(lengths,offsets,train,finite_actions):
    """Exactly the old five-finite-actions plus an in-episode endpoint rule."""
    episode=np.repeat(np.arange(len(lengths)),lengths)
    positions=np.arange(len(episode),dtype=np.int64)
    mask=np.zeros(len(lengths),bool);mask[np.asarray(train,np.int64)]=True
    ends=offsets[episode]+lengths[episode]
    valid=mask[episode] & (positions+5<ends)
    prefix=np.r_[0,np.cumsum(~finite_actions,dtype=np.int64)]
    valid[:len(positions)-4] &= (prefix[5:]-prefix[:-5])==0
    valid[len(positions)-4:]=False
    return np.flatnonzero(valid),episode
