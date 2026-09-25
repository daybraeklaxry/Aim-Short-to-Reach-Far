"""Readable, paginated tables from complete experimental cells."""
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
import json

TASKS = ['cube', 'pusht', 'reacher', 'tworoom']
NAMES = dict(cube='Cube', pusht='PushT', reacher='Reacher', tworoom='TwoRoom')
STARTS = ['Standard', 'Perturbed 1', 'Perturbed 2', 'Perturbed']
MAIN_ARMS = ['LeWM planner', 'CEM (final goal)', 'CEM (learned target)', 'AP-CEM', 'Direct', 'Rank (final goal)', 'Rank (learned target)', 'AP-rank']
DISPLAY = dict(cem_final='CEM (final goal)', cem_learned='CEM (learned target)', cem_observed='AP-CEM', cem_transport='CEM (transported)', rank_final='Rank (final goal)', rank_learned='Rank (learned target)', rank_observed='AP-rank', lewm='LeWM planner', lewm_25='LeWM planner', gaussian_final='CEM (final goal)', gaussian_observed='AP-CEM', ap_final='Rank (final goal)', ap_observed='AP-rank', direct='Direct')

def fmt(value):
    return str(Decimal(str(value)).quantize(Decimal('.1'), rounding=ROUND_HALF_UP))

def main_facts(records):
    """Unrounded task means, formatted once for the manuscript's numeric claims."""
    lookup = {(r['task'], r['arm'], r['start']): r for r in records}
    names = {'LeWM planner': 'LeWM', 'CEM (final goal)': 'CemFinal',
             'CEM (learned target)': 'CemLearned', 'AP-CEM': 'CemObserved',
             'Direct': 'Direct', 'Rank (final goal)': 'RankFinal',
             'Rank (learned target)': 'RankLearned', 'AP-rank': 'RankObserved'}
    lines = ['% MAIN task means, generated from complete unrounded outcomes.']
    for start in ('Standard', 'Perturbed'):
        means = {}
        for arm, name in names.items():
            rows = [lookup[(t, arm, start)] for t in TASKS]
            assert all(r['complete'] for r in rows)
            means[arm] = sum(r['success_percent'] for r in rows) / len(TASKS)
            command = 'VTwoMean' + name + start
            lines.append('\\newcommand{\\' + command + '}{' + fmt(means[arm]) + '}')
        for family, final, observed, learned in [
                ('Cem', 'CEM (final goal)', 'AP-CEM', 'CEM (learned target)'),
                ('Rank', 'Rank (final goal)', 'AP-rank', 'Rank (learned target)')]:
            for target, arm in [('Observed', observed), ('Learned', learned)]:
                command = 'VTwo' + family + target + 'Gain' + start
                lines.append('\\newcommand{\\' + command + '}{'
                             + fmt(means[arm] - means[final]) + '}')
    return '\n'.join(lines) + '\n'


def main_table(records, starts=('Standard', 'Perturbed')):
    lookup = {(r['task'], r['arm'], r['start']): r for r in records}
    assert all(lookup[(t, a, s)]['complete'] for t in TASKS for a in MAIN_ARMS for s in starts)
    lines = [r'\begingroup', r'\small', r'\setlength{\tabcolsep}{2pt}', r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrrrrrrr@{}}', r'\toprule', r' & & No memory & \multicolumn{3}{c}{Observation-only} & \multicolumn{4}{c}{Recorded actions} \\', r'\cmidrule(lr){3-3}\cmidrule(lr){4-6}\cmidrule(l){7-10}', r'Task & Start & \shortstack{LeWM\\planner} & \shortstack{CEM\\final} & \shortstack{CEM\\learned} & AP-CEM & Direct & \shortstack{Rank\\final} & \shortstack{Rank\\learned} & AP-rank \\', r'\midrule']
    for task in TASKS + ['Mean']:
        if task == 'Mean':
            lines.append(r'\midrule')
        elif task != TASKS[0]:
            lines.append(r'\addlinespace[4pt]')
        for si, start in enumerate(starts):
            values = [sum(lookup[(t, a, start)]['success_percent'] for t in TASKS) / 4 if task == 'Mean' else lookup[(task, a, start)]['success_percent'] for a in MAIN_ARMS]
            cells = [fmt(v) for v in values]
            for group in [(1, 2, 3), (4, 5, 6, 7)]:
                best = max(values[i] for i in group)
                for i in group:
                    if values[i] == best:
                        cells[i] = r'\textbf{' + cells[i] + '}'
            label = NAMES.get(task, task) if si == 0 else ''
            lines.append(' & '.join([label, start] + cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular*}', r'\endgroup']
    return '\n'.join(lines) + '\n'

def diagnostic_table(records, starts=('Standard', 'Perturbed')):
    lookup = {(r['task'], r['start']): r for r in records}
    assert all(lookup[(t, s)]['complete'] for t in TASKS for s in starts)
    metrics = ['predictor_error', 'displacement_error', 'predictor_regret', 'displacement_regret', 'direct_regret']
    lines = [r'\begingroup', r'\small', r'\setlength{\tabcolsep}{2.5pt}',
             r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llcrrrrr@{}}', r'\toprule',
             r' & & Eligible & \multicolumn{2}{c}{Endpoint error} & \multicolumn{3}{c}{Selection regret} \\',
             r'\cmidrule(lr){4-5}\cmidrule(l){6-8}',
             r'Task & Start & starts / queries & Predictor & Displ. & Predictor & Displ. & Direct \\', r'\midrule']
    for ti, task in enumerate(TASKS):
        if ti:
            lines.append(r'\addlinespace[4pt]')
        for si, start in enumerate(starts):
            row = lookup[(task, start)]
            values = [row[k] for k in metrics]
            cells = [str(Decimal(str(v)).quantize(Decimal('.001'), rounding=ROUND_HALF_UP)) for v in values]
            for group in [(0, 1), (2, 3, 4)]:
                best = min(values[i] for i in group)
                for i in group:
                    if values[i] == best:
                        cells[i] = r'\textbf{' + cells[i] + '}'
            counts = f"{row['eligible_starts']} / {row['eligible_queries']}"
            lines.append(' & '.join([NAMES[task] if si == 0 else '', start, counts] + cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular*}', r'\endgroup']
    return '\n'.join(lines) + '\n'


SELECTORS = ['Direct', 'Predictor-Final', 'Predictor-Observed', 'Predictor-Learned', 'Simulator-Final', 'Simulator-Observed', 'Simulator-Learned', 'Displacement-Observed']


def first_block_table(records):
    lookup = {(r['task'], r['start'], r['selector']): r for r in records}
    conditions = [(t, s) for t in TASKS for s in ('Standard', 'Perturbed')]
    assert all(lookup[(t, s, a)]['complete'] for t, s in conditions for a in SELECTORS)
    lines = [r'\begingroup', r'\small', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrrr@{}}', r'\toprule',
             ' & ' + ' & '.join(r'\multicolumn{2}{c}{' + NAMES[t] + '}' for t in TASKS) + r' \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}',
             r'First-block selector & Std. & Pert. & Std. & Pert. & Std. & Pert. & Std. & Pert. \\', r'\midrule']
    for ai, arm in enumerate(SELECTORS):
        if ai in [1, 4, 7]:
            lines.append(r'\addlinespace[3pt]')
        cells = []
        for task, start in conditions:
            value = lookup[(task, start, arm)]['success_percent']
            best = max(lookup[(task, start, a)]['success_percent'] for a in SELECTORS)
            cell = fmt(value)
            cells.append(r'\textbf{' + cell + '}' if value == best else cell)
        lines.append(' & '.join([arm.replace('-', '--')] + cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular*}', r'\endgroup']
    return '\n'.join(lines) + '\n'


def first_block_full_table(records):
    lookup = {(r['task'], r['start'], r['selector']): r for r in records}
    assert all(lookup[(t, s, a)]['complete'] for t in TASKS for s in STARTS for a in SELECTORS)
    rows = [[NAMES[t], a.replace('-', '--')] + [fmt(lookup[(t, s, a)]['success_percent']) for s in STARTS] for t in TASKS for a in SELECTORS]
    return longtable('Success (\\%) after changing only the first action block and then using Direct. Each task uses the same 128 MAIN queries. Perturbed averages the two fixed prefixes within query.',
                     'tab:v2-first-block-full', ['Task', 'First-block selector', 'Standard', r'\shortstack{Perturbed\\1}', r'\shortstack{Perturbed\\2}', 'Perturbed'], rows, r'@{}llrrrr@{}')


def learned_target_tables(selections, qualities):
    assert set(selections) == set(qualities) == set(TASKS)
    model = [r'\begingroup', r'\small', r'\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrllrr@{}}', r'\toprule',
             r'Task & Width & Output & Best step & Validation MSE & Worker min. \\', r'\midrule']
    quality = [r'\begingroup', r'\small', r'\setlength{\tabcolsep}{4pt}',
               r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrrr@{}}', r'\toprule',
               r' & & \multicolumn{2}{c}{Successor error} & \multicolumn{2}{c}{Nearest memory distance} \\',
               r'\cmidrule(lr){3-4}\cmidrule(l){5-6}',
               r'Task & Target & Mean & Median & Mean & Median \\', r'\midrule']
    for ti, task in enumerate(TASKS):
        selected, q = selections[task], qualities[task]
        assert not selected['control_results_used'] and q['complete'] and not q['query_episodes_used']
        if ti:
            model.append(r'\addlinespace[4pt]')
            quality.append(r'\addlinespace[4pt]')
        for ci, candidate in enumerate(selected['candidates']):
            mse = f"{candidate['validation_mse']:.6f}"
            if candidate['checkpoint'] == selected['selected']['checkpoint']:
                mse = r'\textbf{' + mse + '}'
            cells = [NAMES[task] if ci == 0 else '', str(candidate['width']), 'Residual' if candidate['residual'] else 'Absolute',
                     f"{candidate['best_step']:,}", mse, fmt(candidate['elapsed_seconds'] / 60)]
            model.append(' & '.join(cells) + r' \\')
        for qi, (name, prefix) in enumerate([('Learned', 'mlp'), ('Observed', 'retrieved')]):
            cells = [NAMES[task] if qi == 0 else '', name]
            for metric in ['successor_squared_l2', 'nearest_memory_squared_l2']:
                cells.extend(f"{q[prefix + '_' + metric][stat]:.3f}" for stat in ['mean', 'median'])
            quality.append(' & '.join(cells) + r' \\')
    result = {}
    for name, lines in [('target_models.tex', model), ('target_quality.tex', quality)]:
        lines += [r'\bottomrule', r'\end{tabular*}', r'\endgroup']
        result[name] = '\n'.join(lines) + '\n'
    return result


def ablation_table(rows):
    best = [max(row[col] for row in rows if row[col] is not None) for col in range(1, 5)]
    lines = [r'\begingroup', r'\small', r'\setlength{\tabcolsep}{7pt}',
             r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
             r' & \multicolumn{2}{c}{AP-rank} & \multicolumn{2}{c}{AP-CEM} \\',
             r'\cmidrule(lr){2-3}\cmidrule(l){4-5}',
             r'Variant & Standard & Perturbed & Standard & Perturbed \\', r'\midrule']
    for ri, row in enumerate(rows):
        if ri in [1, 4, 7]:
            lines.append(r'\addlinespace[4pt]')
        cells = [row[0].replace('%', r'\%')]
        for ci, value in enumerate(row[1:]):
            if value is None:
                cells.append('--')
            else:
                text = fmt(value)
                cells.append(r'\textbf{' + text + '}' if value == best[ci] else text)
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\endgroup']
    return '\n'.join(lines) + '\n'


def transport_table(records):
    lookup = {(r['task'], r['start'], r['config']['arm']): r for r in records if r['kind'] == 'transport'}
    starts, arms = ['Standard', 'Perturbed'], ['cem_observed', 'cem_transport']
    assert all(lookup[(t, s, a)]['complete'] for t in TASKS for s in starts for a in arms)
    lines = [r'\begingroup', r'\small', r'\setlength{\tabcolsep}{8pt}',
             r'\begin{tabular}{@{}llrr@{}}', r'\toprule',
             r'Task & Start & Observed target & Transported target \\', r'\midrule']
    for ti, task in enumerate(TASKS + ['Mean']):
        if task == 'Mean':
            lines.append(r'\midrule')
        elif ti:
            lines.append(r'\addlinespace[4pt]')
        for si, start in enumerate(starts):
            values = [sum(lookup[(t, start, a)]['success_percent'] for t in TASKS) / 4 if task == 'Mean' else lookup[(task, start, a)]['success_percent'] for a in arms]
            cells = [r'\textbf{' + fmt(v) + '}' if v == max(values) else fmt(v) for v in values]
            lines.append(' & '.join([NAMES.get(task, task) if si == 0 else '', start] + cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\endgroup']
    return '\n'.join(lines) + '\n'


def longtable(caption, label, headers, rows, columns, spacing=3):
    n = len(headers)
    head = ' & '.join(headers) + r' \\'
    lines = [r'\begingroup', r'\small', r'\setlength{\tabcolsep}{' + str(spacing) + 'pt}', r'\setlength{\LTcapwidth}{\linewidth}', r'\begin{longtable}{' + columns + '}', r'\caption{' + caption + r'}\label{' + label + r'}\\', r'\toprule', head, r'\midrule', r'\endfirsthead', r'\toprule', head, r'\midrule', r'\endhead', r'\bottomrule', r'\endfoot']
    for row in rows:
        assert len(row) == n
        lines.append(' & '.join(row) + r' \\')
    lines += [r'\end{longtable}', r'\endgroup']
    return '\n'.join(lines) + '\n'

def config_label(kind, cfg):
    if kind == 'budget':
        return r'$I=' + str(cfg['iterations']) + '$'
    if kind == 'horizon':
        return r'$H=' + str(cfg['H']) + r',\ B=' + str(cfg['B']) + '$'
    if kind == 'transport':
        return r'$H=' + str(cfg['H']) + '$'
    if cfg.get('default'):
        return 'Default'
    if 'target_L' in cfg:
        return r'Target $\ell=' + str(cfg['target_L']) + '$'
    if 'memory_fraction' in cfg:
        return f"Memory {100 * cfg['memory_fraction']:.0f}" + r'\%'
    if 'key' in cfg:
        return {'no_far': 'No far endpoint', 'no_delta': 'No displacement term'}[cfg['key']]
    if cfg.get('span') == 'fixed':
        return 'Fixed retrieval span'
    raise ValueError((kind, cfg))

def intervention_order(kind, cfg):
    """Use the main controller order and numeric, prescribed setting order."""
    arms = MAIN_ARMS.copy()
    arms.insert(arms.index('AP-CEM') + 1, 'CEM (transported)')
    arm = arms.index(DISPLAY[cfg['arm']])
    if kind == 'budget':
        return arm, 0, cfg['iterations']
    if kind == 'horizon':
        return arm, 0, cfg['H']
    if kind == 'transport' or cfg.get('default'):
        return arm, 0, 0
    if 'target_L' in cfg:
        return arm, 1, cfg['target_L']
    if 'memory_fraction' in cfg:
        return arm, 2, cfg['memory_fraction']
    if 'key' in cfg:
        return arm, 3, {'no_far': 0, 'no_delta': 1}[cfg['key']]
    if cfg.get('span') == 'fixed':
        return arm, 4, 0
    raise ValueError((kind, cfg))


def full_appendix(interventions, main):
    assert interventions and all(r['complete'] for r in interventions)
    chunks = [r'\clearpage', r'\begin{table}[H]', r'\centering', r'\caption{Success (\%) on the MAIN queries. Perturbed 1 and 2 use the two fixed prefixes; Perturbed averages them within query. Means weight tasks equally.}', r'\label{tab:v2-main-full}', main_table(main, STARTS).strip(), r'\end{table}', r'\FloatBarrier']
    grouped = defaultdict(dict)
    for r in interventions:
        if r['kind'] == 'main':
            continue  # These learned-controller cells are already in the complete MAIN table.
        key = (r['kind'], r['task'], json.dumps(r['config'], sort_keys=True))
        assert r['start'] not in grouped[key]
        grouped[key][r['start']] = r
    assert all(set(group) == set(STARTS) for group in grouped.values())
    titles = dict(transport='Observed and transported targets', budget='CEM search budgets', ablation='Anchored Planning design choices', horizon='Goal offsets')
    descriptions = dict(transport='observed and transported targets', budget='the CEM budget sweep', ablation='the Anchored Planning ablations', horizon='the goal-offset sweep')
    for kind in ['transport', 'budget', 'ablation', 'horizon']:
        selected = [(key, value) for key, value in grouped.items() if key[0] == kind]
        selected.sort(key=lambda pair: (TASKS.index(pair[0][1]), intervention_order(kind, pair[1]['Standard']['config'])))
        assert selected, kind
        chunks.append(r'\subsection{' + titles[kind] + '}')
        for metric, suffix, caption in [('success_percent', 'success', 'Success (\\%)'), ('mean_pred_blocks', 'work', 'Mean predicted five-action blocks per episode, counted until the episode ends')]:
            rows = []
            for (_, task, _), group in selected:
                cfg = group['Standard']['config']
                values = [fmt(group[s][metric]) for s in STARTS]
                if metric == 'mean_pred_blocks':
                    values = [format(Decimal(value), ',f') for value in values]
                rows.append([NAMES[task], DISPLAY[cfg['arm']], config_label(kind, cfg)] + values)
            headers = ['Task', 'Controller', 'Setting', 'Standard', r'\shortstack{Perturbed\\1}', r'\shortstack{Perturbed\\2}', 'Perturbed']
            chunks.append(longtable(caption + ' for ' + descriptions[kind] + '. All rows use the MAIN query and action protocol.', 'tab:v2-' + kind + '-' + suffix, headers, rows, r'@{}lp{.20\linewidth}p{.19\linewidth}rrrr@{}', spacing=2.5 if metric == 'mean_pred_blocks' else 3))
    return '\n'.join(chunks) + '\n'

def mechanism_records(original, additional):
    """Require every prescribed main controller and the transported controller."""
    if not original.get('complete'):
        return None
    rows = []
    for row in original['rows']:
        rows.append(dict(task=row['task'], arm=row['arm'], start=row['start'], metrics=row))
    for task in TASKS:
        for arm, kind in [('cem_learned', 'main'), ('rank_learned', 'main'), ('lewm_25', 'main'), ('cem_transport', 'transport')]:
            for start in STARTS:
                matches = [r for r in additional.get('rows', []) if r['group']['task'] == task and r['group']['kind'] == kind and r['group']['config']['arm'] == arm and r['start'] == start]
                if len(matches) != 1 or not matches[0]['complete']:
                    return None
                rows.append(dict(task=task, arm=arm, start=start, metrics=matches[0]))
    keys = {(r['task'], r['arm'], r['start']) for r in rows}
    assert len(keys) == len(rows) == 4 * 9 * 4
    order = MAIN_ARMS.copy()
    order.insert(order.index('AP-CEM') + 1, 'CEM (transported)')
    rows.sort(key=lambda r: (TASKS.index(r['task']), order.index(DISPLAY[r['arm']]), STARTS.index(r['start'])))
    return rows

def metric_cell(metric):
    n, d = metric['numerator'], metric['denominator']
    assert 0 <= n <= d
    return '-- (0/0)' if d == 0 else fmt(metric['percent']) + f' ({n}/{d})'

def mechanism_tables(original, additional):
    records = mechanism_records(original, additional)
    if records is None:
        return None
    columns = r'@{}lp{.22\linewidth}lrrr@{}'
    prefix = 'Values are percentages, followed by positive/eligible episode counts. Stalling conditions on failed episodes with decisions; detours condition on policy-entered successes. Perturbed pools eligible episodes from both prefixes. '
    base = longtable(r'\textbf{Physical and latent behavior during control.} ' + prefix + 'Physical detours use $\\delta=0.5$ and stalling uses $W=10$. Latent detours are unavailable for Direct.', 'tab:v2-mechanism', ['Task', 'Controller', 'Start', 'Stalling', r'\shortstack{Physical\\detour}', r'\shortstack{Latent\\detour}'], [[NAMES[r['task']], DISPLAY[r['arm']], r['start']] + [('--' if r['arm'] == 'direct' and k == 'latent_detour' else metric_cell(r['metrics'][k])) for k in ['stall_W10', 'physical_detour_0.5', 'latent_detour']] for r in records], columns)
    sensitivity = []
    for label, heading, metrics in [('window', 'Stalling across decision-window lengths.', ['stall_W5', 'stall_W10', 'stall_W20']), ('threshold', 'Physical detours across error-rise thresholds.', ['physical_detour_0.25', 'physical_detour_0.5', 'physical_detour_1.0'])]:
        headers = ['$W=5$', '$W=10$', '$W=20$'] if label == 'window' else [r'$\delta=0.25$', r'$\delta=0.5$', r'$\delta=1.0$']
        sensitivity.append(longtable(r'\textbf{' + heading + '} ' + prefix, 'tab:v2-mechanism-' + label, ['Task', 'Controller', 'Start'] + headers, [[NAMES[r['task']], DISPLAY[r['arm']], r['start']] + [metric_cell(r['metrics'][k]) for k in metrics] for r in records], columns))
    return dict(mechanism=base, sensitivity='\n'.join(sensitivity))
