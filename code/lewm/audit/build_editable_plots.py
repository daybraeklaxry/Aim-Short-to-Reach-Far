"""Rebuild the updated LeWM comparison plots from the supplied complete data."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pypdf import PdfReader, PdfWriter, Transformation
from xml.etree import ElementTree as ET
from publication_plots import compute_figure, horizon_figure, teaser_figure

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
OUT = DATA.parent
plot = json.loads((DATA / 'plot-data.json').read_text())
main = json.loads((DATA / 'main-results.json').read_text())
refs = [dict(task=r['task'], start=r['start'], work=r['mean_pred_blocks'], success=r['success_percent'])
        for r in main if r['arm'] == 'LeWM planner' and r['start'] in ['Standard', 'Perturbed']]
assert len(refs) == 8
plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix', 'font.size': 8,
    'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
    'lines.linewidth': 1.2, 'savefig.pad_inches': .04})
def save(fig, name):
    stem = OUT / name
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(stem.with_suffix('.' + ext), **({'dpi': 160} if ext == 'png' else {}))
    plt.close(fig)
save(compute_figure(plot['compute'], refs, ['Standard']), 'Figure_5_Compute')
save(horizon_figure(plot['horizon'], ['Standard']), 'Figure_3_Horizon')
save(compute_figure(plot['compute_perturbed'], refs, ['Perturbed']), 'appendix/compute_perturbed')
save(horizon_figure(plot['horizon_perturbed'], ['Perturbed']), 'appendix/horizon_perturbed')
save(teaser_figure(plot['teaser']), 'editable_source/figure1/teaser')
concept, teaser = DATA / 'figure1/concept.pdf', DATA / 'figure1/teaser.pdf'
writer = PdfWriter()
page = writer.add_blank_page(width=396, height=150)
page.merge_page(PdfReader(concept).pages[0])
page.merge_transformed_page(PdfReader(teaser).pages[0], Transformation().translate(tx=200, ty=0))
writer.write(OUT / 'Figure_1_Mechanism.pdf')
ns = 'http://www.w3.org/2000/svg'
ET.register_namespace('', ns)
root = ET.Element(f'{{{ns}}}svg', {'width': '396pt', 'height': '150pt', 'viewBox': '0 0 396 150'})
for path, x, width in [(DATA / 'figure1/concept.svg', 0, 190), (DATA / 'figure1/teaser.svg', 200, 196)]:
    panel = ET.parse(path).getroot()
    panel.attrib.update(x=str(x), y='0', width=str(width), height='150')
    root.append(panel)
ET.ElementTree(root).write(OUT / 'Figure_1_Mechanism.svg', encoding='utf-8', xml_declaration=True)
print('Rebuilt Figures 1, 3, 5 and the perturbed-start comparison plots.')
