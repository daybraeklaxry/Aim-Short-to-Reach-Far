"""Check numerical identity and timing of transfer-before-expand on real queries."""
import contextlib, io, json, sys, time
from pathlib import Path
import numpy as np
import torch
import official_baseline as ob

task = sys.argv[1]
torch.set_num_threads(2)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
p = ob.runtime.read(ob.BASE / 'protocol.json')
_, _, _, _, api, ew, context, native, _ = ob.runtime.load_runtime(task, 'confirm')
row = p['studies'][task]['confirm'][0]
source = next(s for s in ob.runtime.read(ob.BASE / 'source_manifest.json')['rows'] if s['task'] == task and s['condition'] == 'clean' and s['identity']['record_id'] == row['record_id'])
env, _, _, _, _, cached = ob.run_suite.LockedConstruction.construct(ew, api, context, native.scaler, task, row, 'confirm', ob.BASE, p, source)
try:
    results = []
    with torch.inference_mode():
        for cpu_input in [True, False]:
            policy = ob.policy_for(native, env, ob.query_seed(task, 0))
            if cpu_input:
                policy.transform = {'pixels': native.transform, 'goal': native.transform}
            torch.cuda.synchronize()
            start = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                actions = ob.released_block(policy, ob.raw_info(native, cached['post_rgb'], cached['goal_rgb']))
            torch.cuda.synchronize()
            results.append((actions, time.perf_counter() - start))
    error = float(np.abs(results[0][0] - results[1][0]).max())
    assert error == 0, (task, error)
    report = {'task': task, 'full_plan_max_abs_error': error, 'cpu_expansion_seconds': results[0][1], 'gpu_expansion_seconds': results[1][1]}
    (Path(__file__).resolve().parent / f'input_transfer_{task}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
finally:
    env.close()
