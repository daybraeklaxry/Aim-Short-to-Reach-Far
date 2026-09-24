"""Frozen episode-cluster bootstrap used for the 7,680-outcome main study."""
import numpy as np

TASKS = ('cube','pusht','reacher','tworoom')
ARMS = ('direct','ap_observed','ap_final','gaussian_observed','gaussian_final')
CONDITIONS = ('clean','prefix_a','prefix_b')
DISPLAY_CONDITIONS = (*CONDITIONS,'prefix_mean')
COMPARISONS = (('ap_observed','ap_final','primary'),
               ('gaussian_observed','gaussian_final','primary'),
               ('ap_observed','direct','secondary'))
N, REPLICATES, SEED = 128, 10000, 20260913


def ci(values):
    return (100*np.quantile(values,[.025,.975])).tolist()


def bootstrap(y):
    """One fixed independent draw stream per task; shared across all its cells."""
    rng=np.random.default_rng(SEED)
    draws=np.empty((4,REPLICATES,4,5),dtype=np.float64)
    for t in range(4):
        indices=rng.integers(0,N,size=(REPLICATES,N))
        counts=np.zeros((REPLICATES,N),dtype=np.int16)
        np.add.at(counts,(np.arange(REPLICATES)[:,None],indices),1)
        means=(counts.astype(np.float64) @ y[t].reshape(N,15)/N).reshape(REPLICATES,3,5)
        draws[t,:,:3]=means
        draws[t,:,3]=(means[:,1]+means[:,2])/2
    points=np.concatenate((y.mean(axis=1),y[:,:,1:3].mean(axis=(1,2))[:,None,:]),axis=1)
    return points,draws


def estimate_tables(y):
    points,draws=bootstrap(y)
    cells,contrasts=[],[]
    for t,task in enumerate(TASKS):
        for c,condition in enumerate(DISPLAY_CONDITIONS):
            cell=dict(task=task,condition=condition,n_episode_clusters=N,
                success_percent={a:float(100*points[t,c,k]) for k,a in enumerate(ARMS)},
                success_ci95_percent={a:ci(draws[t,:,c,k]) for k,a in enumerate(ARMS)})
            if c<3:
                cell['n']=N
                cell['success_counts_author_evidence']={a:int(y[t,:,c,k].sum()) for k,a in enumerate(ARMS)}
            cells.append(cell)
            for left,right,role in COMPARISONS:
                l,r=ARMS.index(left),ARMS.index(right)
                row=dict(task=task,condition=condition,left=left,right=right,n_episode_clusters=N,
                    difference_pp=float(100*(points[t,c,l]-points[t,c,r])),
                    ci95_pp=ci(draws[t,:,c,l]-draws[t,:,c,r]))
                if c<3:
                    ll,rr=y[t,:,c,l].astype(bool),y[t,:,c,r].astype(bool)
                    row['discordance']=dict(left_only=int((ll&~rr).sum()),right_only=int((rr&~ll).sum()),
                        both_success=int((ll&rr).sum()),both_failure=int((~ll&~rr).sum()))
                contrasts.append(row)
    macro_points,macro_draws=points.mean(axis=0),draws.mean(axis=0)
    macro_rates,macro_contrasts=[],[]
    for c,condition in enumerate(DISPLAY_CONDITIONS):
        macro_rates.append(dict(condition=condition,task_weight=.25,n_episode_clusters_per_task=N,
            success_percent={a:float(100*macro_points[c,k]) for k,a in enumerate(ARMS)},
            success_ci95_percent={a:ci(macro_draws[:,c,k]) for k,a in enumerate(ARMS)}))
        for left,right,role in COMPARISONS:
            l,r=ARMS.index(left),ARMS.index(right)
            macro_contrasts.append(dict(condition=condition,left=left,right=right,
                prespecified_role=role if condition in ('clean','prefix_mean') else 'descriptive_prefix',
                difference_pp=float(100*(macro_points[c,l]-macro_points[c,r])),
                ci95_pp=ci(macro_draws[:,c,l]-macro_draws[:,c,r])))
    planned=[r for r in macro_contrasts if r['prespecified_role']!='descriptive_prefix']
    assert sum(r['prespecified_role']=='primary' for r in planned)==4
    assert sum(r['prespecified_role']=='secondary' for r in planned)==2
    return dict(cells=cells,paired_contrasts=contrasts,equal_task_macro=macro_rates,
        macro_contrasts=macro_contrasts,planned_macro_contrasts=planned)
