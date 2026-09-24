"""Finish this run after its registered workers complete; no new experiments."""
from pathlib import Path
import json, shutil, subprocess, sys, tarfile, time

ROOT = Path(__file__).resolve().parent
jobs = json.loads((ROOT / 'official_launch.json').read_text())
print(json.dumps({'waiting_for_registered_shards': len(jobs)}), flush=True)
markers = [ROOT / 'official_results' / j['task'] / f'complete_q{j["start"]:03}-{j["stop"]:03}.json' for j in jobs]
while not all(p.exists() and json.loads(p.read_text())['complete'] for p in markers):
    time.sleep(30)
subprocess.run([sys.executable, str(ROOT / 'collect_official.py')], check=True)
subprocess.run(['bash', str(ROOT / 'run.sh'), 'analyze_official_mechanism.py'], check=True)
inputs = ROOT / 'evaluation_inputs'
inputs.mkdir(exist_ok=True)
base = ROOT.parent / 'fresh_target_interface_study_v2'
for name in ['protocol.json', 'source_manifest.json', 'native_action_stats.json']:
    if (base / name).is_file():
        shutil.copy2(base / name, inputs / name)
durable = Path('external/le-wm/research/ap_rank_oral_revision_20260913/lewm_baseline_audit_20260923')
durable.mkdir(parents=True, exist_ok=True)
archive = durable / 'official_baseline_results.tar.gz'
members = [ROOT / 'official_results', inputs, ROOT / 'traces']
members += list(ROOT.glob('*.py')) + list(ROOT.glob('*.json')) + list(ROOT.glob('*.jsonl')) + [ROOT / 'run.sh']
with tarfile.open(archive, 'w:gz', compresslevel=3) as tar:
    for path in sorted(set(members)):
        tar.add(path, arcname=str(path.relative_to(ROOT)))
for name in ['official_summary.json', 'official_mechanism.json', 'pilot_trace_equivalence.json', 'official_launch.json']:
    shutil.copy2(ROOT / name, durable / name)
result = {'complete': True, 'episodes': 7296, 'archive': str(archive), 'archive_bytes': archive.stat().st_size,
          'registered_shards': len(jobs), 'mechanism_episodes': 1536}
(ROOT / 'finalization.json').write_text(json.dumps(result, indent=2))
(durable / 'finalization.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
