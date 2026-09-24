from pathlib import Path
import datetime,json,collections
ROOT=Path('runtime/ap')
OUT=ROOT/'evidence_strengthening_20260923/results_v2'
BASE=ROOT/'fresh_target_interface_study_v2'
OUT.mkdir(parents=True,exist_ok=True)
p=json.loads((BASE/'protocol.json').read_text())
reuse=[]
for task in p['studies']:
 files=sorted((BASE/'evaluation').glob(f'outcomes_{task}_shard*.jsonl'))
 seen={};counts=collections.Counter()
 for f in files:
  for line in f.open():
   r=json.loads(line);k=(r['identity']['record_id'],r['condition'],r['arm'])
   assert k not in seen and r['completed'];seen[k]=r;counts[r['arm']]+=1
 assert len(seen)==1920 and set(counts.values())=={384}
 samples={}
 for arm in p['arms']:
  for (qid,condition,which),r in seen.items():
   if which==arm and r['decisions']>0:
    stem=BASE/'evaluation/traces'/task/f'confirm_{qid}_{condition}_{arm}'
    import numpy as np
    with np.load(stem.with_suffix('.npz')) as z:keys=z.files
    t=json.loads(stem.with_suffix('.json').read_text())
    samples[arm]=dict(trace=str(stem),arrays=keys,has_decisions=bool(t['decisions']),has_physical_trajectory=bool(t['trajectory']))
    break
 reuse.append(dict(task=task,episodes=len(seen),arms=dict(counts),logs=samples,sources=[str(f) for f in files]))
receipt=json.loads(Path(p['fresh_selection_receipt']).read_text())
report=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_protocol=str(BASE/'protocol.json'),source_manifest=str(BASE/'source_manifest.json'),fresh_selection_receipt=str(p['fresh_selection_receipt']),memory_sizes={t:len(es) for t,es in p['bank_train_episodes'].items()},base_reuse=reuse,original_budget_path=str(ROOT/'evidence_strengthening_20260923/budget'),replay_checks=str(ROOT/'evidence_strengthening_20260923/replay'),original_transport_study=str(ROOT/'anchor_transport_20260915'),supporting_study='external/le-wm/research/ap_rank_confirmatory_20260912/confirmation',source_paths=dict(main=str(BASE/'frozen_code'),runtime=str(Path(p['native_root'])/'frozen_code'),budget=str(ROOT/'evidence_strengthening_20260923/budget_code'),lewm=str(ROOT/'evidence_strengthening_20260923/long_code'),exact=str(ROOT/'evidence_strengthening_20260923/exact_code')))
(OUT/'reuse.json').write_text(json.dumps(report,indent=2))
text='''# MAIN experiment reuse inventory\n\nCurrent request: pasted-text-1.txt (aa76f888). This replaces the older, narrower PLAN.md.\n\nAll 7,680 completed main outcomes are retained without rerunning. The existing JSON/NPZ traces contain per-decision encodings for scored controllers and complete physical trajectories. Mechanism analysis can therefore reuse these logs; new diagnostics must not be inferred from the older latent-only plateau metric.\n\n## Authoritative sources\n'''
text+='\n'.join(f'- {k}: `{v}`' for k,v in report.items() if k in ('base_protocol','source_manifest','fresh_selection_receipt','original_budget_path','replay_checks','original_transport_study','supporting_study'))
text+='\n\n## Reuse decisions\n- Main five-controller outcomes: reuse all four tasks and three starts.\n- CEM 30 iterations, full-memory/default AP, and default task horizon: references to the same main records, not recomputed duplicates.\n- Four running budget jobs: adopt their completed episodes and live process handles.\n- Main logs: reuse; no blanket mechanism replay is necessary.\n- Existing transport/supporting planner/diagnostics: keep as separately labelled evidence; new MAIN counterparts remain required.\n- The old Cube target-network run supplies its FP32 memory cache. Its two-hidden-layer, three-seed training recipe is not the requested four-configuration baseline and will not be substituted for it.\n\n## Protocol decisions\n- Retain original query, perturbation, optimizer seeds, checkpoints, action projection, and stopping rules.\n- Train all four tasks, widths 512/1024 crossed with residual/absolute output; select only by held-out memory MSE. No control-query tuning.\n- Report no confidence intervals, error bars or significance tests, as requested.\n- L=1/3/10 changes the observed target timestamp; the frozen predictor and action block stay at five steps. This is a target-time ablation, not a different trained dynamics horizon.\n- A flat failure trace is descriptive stalling, not a certificate that search found a global or local optimum.\n- Shared caches/outputs remain under results_v2, except explicitly adopted read-only prior evidence.\n'
(OUT/'REUSE.md').write_text(text)
print(json.dumps(dict(main_outcomes=sum(r['episodes'] for r in reuse),memory_sizes=report['memory_sizes'],logs_reusable=True,output=str(OUT))))
