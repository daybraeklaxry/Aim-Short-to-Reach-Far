"""Replace the LeWM baseline using complete released-policy outcomes only."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
from xml.etree import ElementTree as ET
import json, shutil, sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pypdf import PdfReader, PdfWriter, Transformation

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / 'paper'
sys.path.insert(0, str(ROOT / 'publication_code'))
from publication_tables import TASKS, NAMES, STARTS, fmt, metric_cell, main_table, main_facts, full_appendix
from publication_plots import compute_figure, horizon_figure, teaser_figure

summary = json.loads((ROOT / 'official_summary.json').read_text())
assert summary['complete'] and summary['episodes'] == summary['expected'] == 7296
lookup = {(r['task'], r['H'], r['start'], r['mode']): r for r in summary['rows']}
original = summary['original_offsets']
old_main = json.loads((ROOT / 'publication_inputs/main.json').read_text())['rows']
main = deepcopy(old_main)
old_inter = json.loads((ROOT / 'publication_inputs/interventions.json').read_text())['rows']
inter = deepcopy(old_inter)
old_plot = json.loads((ROOT / 'publication_inputs/plot_data.json').read_text())
plot = deepcopy(old_plot)

def cell(task, start, h=None, mode='released_raw'):
    return lookup[(task, original[task] if h is None else h, start, mode)]

for row in main:
    if row['arm'] == 'LeWM planner':
        c = cell(row['task'], row['start'])
        for key in ['success_percent', 'mean_pred_blocks']:
            row[key] = c[key]
        row['baseline_configuration'] = 'released_raw; plan 25, execute 25'
for old, new in zip(old_main, main):
    if old['arm'] != 'LeWM planner':
        assert old == new
for row in inter:
    if row['config']['arm'] == 'lewm':
        assert row['kind'] == 'horizon'
        c = cell(row['task'], row['start'], row['config']['H'])
        for key in ['success_percent', 'mean_pred_blocks', 'mean_steps']:
            row[key] = c[key]
        row['source_files'] = ['official_outcomes.jsonl']
for old, new in zip(old_inter, inter):
    if old['config']['arm'] != 'lewm':
        assert old == new
for name in ['horizon', 'horizon_perturbed']:
    for row in plot[name]:
        if row['arm'] == 'lewm':
            row['success'] = cell(row['task'], row['start'], row['H'])['success_percent']
for row in plot['teaser']:
    if row['arm'] == 'lewm':
        row['success'] = cell('pusht', 'Standard', row['H'])['success_percent']

(ROOT / 'publication_outputs').mkdir(exist_ok=True)
for name, value in [('main', main), ('interventions', inter), ('plot_data', plot)]:
    (ROOT / 'publication_outputs' / (name + '.json')).write_text(json.dumps(value, indent=2))
(PAPER / 'tables_v2/main.tex').write_text(main_table(main))
(PAPER / 'manuscript_fragments/main_facts.tex').write_text(main_facts(main))
full = full_appendix(inter, main).replace('MAIN', 'main')
full = full.replace('All rows use the main query and action protocol.', 'All rows use the main queries and the specified controller configurations.')
(PAPER / 'tables_v2/appendix_full.tex').write_text(full)

mechanism = json.loads((ROOT / 'official_mechanism.json').read_text())
assert mechanism['complete'] and mechanism['episodes'] == 1536
behavior = {(NAMES[r['task']], r['start']): r for r in mechanism['rows']}
for filename in ['mechanism.tex', 'mechanism_sensitivity.tex']:
    path = PAPER / 'tables_v2' / filename
    lines = path.read_text().splitlines()
    metrics = ['stall_W10', 'physical_detour_0.5', 'latent_detour']
    count = 0
    for i, line in enumerate(lines):
        if 'tab:v2-mechanism-window' in line:
            metrics = ['stall_W5', 'stall_W10', 'stall_W20']
        if 'tab:v2-mechanism-threshold' in line:
            metrics = ['physical_detour_0.25', 'physical_detour_0.5', 'physical_detour_1.0']
        if ' & LeWM-R5 & ' in line:
            parts = line.split(' & ')
            row = behavior[(parts[0], parts[2])]
            lines[i] = ' & '.join([parts[0], 'LeWM planner', parts[2]] + [metric_cell(row[k]) for k in metrics]) + r' \\'
            count += 1
    assert count == (16 if filename == 'mechanism.tex' else 32), (filename, count)
    path.write_text('\n'.join(lines) + '\n')

old_lookup = {(r['task'], r['arm'], r['start']): r for r in old_main}
plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix', 'font.size': 8,
    'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
    'lines.linewidth': 1.2, 'savefig.pad_inches': .04})
figures = ROOT / 'figures'
figures.mkdir(exist_ok=True)
refs = [dict(task=t, start=s, success=cell(t, s)['success_percent'], work=cell(t, s)['mean_pred_blocks']) for t in TASKS for s in ['Standard', 'Perturbed']]
def save(fig, name):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(figures / f'{name}.{ext}', **({'dpi': 160} if ext == 'png' else {}))
    plt.close(fig)
    shutil.copy2(figures / f'{name}.pdf', PAPER / 'figures_v2' / f'{name}.pdf')
for start, suffix in [('Standard', ''), ('Perturbed', '_perturbed')]:
    save(compute_figure(plot['compute' + suffix], refs, [start]), 'compute' + suffix)
    save(horizon_figure(plot['horizon' + suffix], [start]), 'horizon' + suffix)
save(teaser_figure(plot['teaser']), 'teaser')
pages = [PdfReader(ROOT / 'figure1/concept.pdf').pages[0], PdfReader(figures / 'teaser.pdf').pages[0]]
assert [(round(float(p.mediabox.width)), round(float(p.mediabox.height))) for p in pages] == [(190, 150), (196, 150)]
writer = PdfWriter()
page = writer.add_blank_page(width=396, height=150)
page.merge_page(pages[0])
page.merge_transformed_page(pages[1], Transformation().translate(tx=200, ty=0))
writer.write(figures / 'Figure_1.pdf')
shutil.copy2(figures / 'Figure_1.pdf', PAPER / 'figures_v2/Figure_1.pdf')
ns = 'http://www.w3.org/2000/svg'
ET.register_namespace('', ns)
svg = ET.Element(f'{{{ns}}}svg', {'width': '396pt', 'height': '150pt', 'viewBox': '0 0 396 150'})
for source, x, width in [(ROOT / 'figure1/concept.svg', 0, 190), (figures / 'teaser.svg', 200, 196)]:
    panel = ET.parse(source).getroot()
    panel.attrib.update(x=str(x), y='0', width=str(width), height='150')
    svg.append(panel)
ET.ElementTree(svg).write(figures / 'Figure_1.svg', encoding='utf-8', xml_declaration=True)

comparisons = []
for t in TASKS:
    for s in ['Standard', 'Perturbed']:
        ap = old_lookup[(t, 'AP-CEM', s)]
        lewm = cell(t, s)
        comparisons.append({'task': t, 'start': s, 'AP_CEM': ap['success_percent'],
            'LeWM': lewm['success_percent'], 'old_LeWM_R5': old_lookup[(t, 'LeWM planner', s)]['success_percent'],
            'LeWM_Box': cell(t, s, mode='released_box')['success_percent'],
            'AP_work_percent_of_LeWM': 100 * ap['mean_pred_blocks'] / lewm['mean_pred_blocks']})
horizon = []
for row in plot['horizon'] + plot['horizon_perturbed']:
    if row['arm'] == 'cem_observed':
        baseline = cell(row['task'], row['start'], row['H'])
        horizon.append({'task': row['task'], 'start': row['start'], 'H': row['H'],
                        'AP_CEM': row['success'], 'LeWM': baseline['success_percent']})
claims = {'main': comparisons, 'horizons': horizon,
          'all_other_main_results_unchanged': True, 'all_other_intervention_results_unchanged': True,
          'generated_utc': datetime.now(timezone.utc).isoformat()}
(ROOT / 'result_comparison.json').write_text(json.dumps(claims, indent=2))
print(json.dumps(claims['main']))
