"""Record package metadata under the same launcher as an audit task."""
from pathlib import Path
import argparse, importlib.metadata as metadata, json, platform, sys

R = Path(__file__).parent
sys.path.insert(0, str(R / 'deps'))
parser = argparse.ArgumentParser()
parser.add_argument('--task', required=True)
args = parser.parse_args()
names = ['torch', 'torchvision', 'stable-worldmodel', 'stable-pretraining',
         'mujoco', 'dm-control', 'gymnasium', 'numpy', 'h5py', 'hydra-core',
         'PyOpenGL', 'pymunk']
versions = {name: metadata.version(name) for name in names}
file = R / 'lewm_checks' / 'runtime_versions.json'
report = json.loads(file.read_text()) if file.exists() else {}
report.pop('packages', None)
report.setdefault('task_launchers', {})[args.task] = {
    'python': platform.python_version(), 'packages': versions}
file.write_text(json.dumps(report, indent=2))
print(json.dumps({'task': args.task, 'python': platform.python_version(), 'packages': versions}))
