"""Run final offline analysis once every registered experiment has completed."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time

ROOT = Path('runtime/ap/evidence_strengthening_20260923/results_v2')
manifest = json.loads((ROOT / 'queue-manifest.json').read_text())
ids = [job['id'] for job in manifest['jobs']]
scheduler_pid = json.loads((ROOT / 'scheduler-launch.json').read_text())['pid']
last_report = 0.0
while True:
    state = json.loads((ROOT / 'queue-state.json').read_text())
    counts = Counter(state['jobs'][job]['status'] for job in ids)
    if counts.get('failed'):
        failed = [job for job in ids if state['jobs'][job]['status'] == 'failed']
        raise RuntimeError(f'Final analysis requires all jobs: failed={failed}')
    if counts['complete'] == len(ids):
        break
    os.kill(scheduler_pid, 0)
    if time.monotonic() - last_report >= 1800:
        print(json.dumps({'utc': datetime.now(timezone.utc).isoformat(),
                          'waiting_for_registered_jobs': dict(counts)}), flush=True)
        last_report = time.monotonic()
    time.sleep(30)

env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
stages = ['collect_interventions.py', 'collect_exact.py', 'analyze_all_mechanism.py',
          'collect_progress.py', 'emit_artifacts.py']
completed = []
for name in stages:
    print(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'stage': name}), flush=True)
    subprocess.run([sys.executable, str(ROOT / 'code' / name)], env=env, check=True)
    completed.append(name)

artifact = json.loads((ROOT / 'artifact-status.json').read_text())
assert artifact['all_requested_outputs_ready'], artifact['outputs']
interventions = json.loads((ROOT / 'collected/interventions.json').read_text())
assert interventions['complete']
mechanism = json.loads((ROOT / 'mechanism_all/summary.json').read_text())
assert mechanism['complete'], 'Mechanism extraction still has missing rows or coordinates'
receipt = {'completed_utc': datetime.now(timezone.utc).isoformat(),
           'registered_jobs': len(ids), 'stages': completed,
           'intervention_cells': interventions['total_cells'],
           'publication_outputs': len(artifact['outputs']),
           'all_requested_outputs_ready': True,
           'manuscript_review_and_delivery_remaining': True}
(ROOT / 'final-analysis-complete.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)
