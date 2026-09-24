"""Nonpublication layout proof using only the four complete MAIN PushT points."""
from pathlib import Path
import argparse
import json
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
from matplotlib.text import Text

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.root/'code'))
from publication_plots import teaser_figure
for path in (args.root/'fonts').glob('*.ttf'):
    font_manager.fontManager.addfont(str(path))
plt.rcParams.update({'font.family':'Times New Roman', 'mathtext.fontset':'stix',
    'font.size':8, 'axes.titlesize':8, 'axes.labelsize':8, 'xtick.labelsize':8,
    'ytick.labelsize':8, 'legend.fontsize':8, 'axes.spines.top':False,
    'axes.spines.right':False, 'axes.linewidth':.6, 'lines.linewidth':1.2})
data = json.loads((args.root/'collected/main.json').read_text())['rows']
names = {'CEM (final goal)':'cem_final', 'CEM (learned target)':'cem_learned',
         'AP-CEM':'cem_observed', 'LeWM planner':'lewm'}
rows = [r for r in data if r['task']=='pusht' and r['start']=='Standard' and r['arm'] in names]
assert len(rows) == 4 and all(r['complete'] for r in rows)
points = [dict(arm=names[r['arm']], H=140, success=r['success_percent']) for r in rows]
fig = teaser_figure(points)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
box = fig.bbox
clipped = []
for obj in fig.findobj(Text):
    if not obj.get_visible() or not obj.get_text():
        continue
    b = obj.get_window_extent(renderer)
    if b.x0 < box.x0-.5 or b.y0 < box.y0-.5 or b.x1 > box.x1+.5 or b.y1 > box.y1+.5:
        clipped.append(obj.get_text())
out = args.root/'layout_proofs'
out.mkdir(exist_ok=True)
fig.savefig(out/'teaser-layout-only.png', dpi=180)
plt.close(fig)
receipt = {'scope':'Layout only; the full horizon sweep is unfinished.',
           'used_points':points, 'clipped_labels':clipped, 'print_size_points':[196,150]}
(out/'teaser-layout-only.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
