"""Report completed experiments, including learned-target wins and pending work."""
from pathlib import Path
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
import json
import os
from publication_tables import DISPLAY, config_label, intervention_order

ROOT = Path('runtime/ap/evidence_strengthening_20260923/results_v2')
TASKS = ['cube', 'pusht', 'reacher', 'tworoom']
ARMS = ['LeWM planner', 'CEM (final goal)', 'CEM (learned target)', 'AP-CEM', 'Direct', 'Rank (final goal)', 'Rank (learned target)', 'AP-rank']

def read(name):
    p = ROOT / name
    return json.loads(p.read_text()) if p.exists() else None

def save(name, value):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(value, indent=2))
    os.replace(q, p)

def fmt(x):
    return str(Decimal(str(x)).quantize(Decimal('.1'), rounding=ROUND_HALF_UP))


def retry_history(state):
    history = []
    for job, entry in state['jobs'].items():
        for attempt in range(1, entry.get('attempts', 0)):
            path = ROOT/'logs'/f'{job}-attempt{attempt}.log'
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size-128000))
                text = stream.read().decode(errors='replace')
            reason = 'CUDA out of memory' if 'out of memory' in text.lower() else 'See attempt log'
            history.append(dict(job=job, attempt=attempt, reason=reason,
                                current_status=entry['status'], log=str(path)))
    save('worker-attempt-history.json', history)
    if not history:
        return ''
    text = '## Retried worker attempts\n\nEarlier worker errors are retained even when their jobs subsequently complete. The scheduler reduces per-GPU concurrency on CUDA OOM; the study configuration stays fixed.\n\n'
    text += '| Job | Earlier attempt | Cause | Current status | Log |\n|---|---:|---|---|---|\n'
    for row in history:
        text += '| ' + ' | '.join([row['job'], str(row['attempt']), row['reason'], row['current_status'], '`'+row['log']+'`']) + ' |\n'
    return text + '\n'


def intervention_report():
    source = read('collected/interventions.json')
    if not source:
        return '\n## Intervention results\n\nCollection is pending.\n'
    starts = ['Standard', 'Perturbed 1', 'Perturbed 2', 'Perturbed']
    grouped = defaultdict(dict)
    for row in source['rows']:
        if row['kind'] != 'main':
            grouped[(row['kind'], row['task'], json.dumps(row['config'], sort_keys=True))][row['start']] = row
    names = dict(transport='Observed and transported targets', budget='CEM search budgets', ablation='Anchored Planning design choices', horizon='Goal offsets')
    text = '\n## Intervention success rates (%)\n\nEvery registered setting appears below. Pending cells show completed / assigned starts instead of a success estimate. Mean rows weight the four tasks equally and are available only when all four task cells are complete. Goal-offset means use the common offsets 25, 50, and 100; each task retains its prescribed allowance-to-offset ratio. Means at the task-specific original offsets are the MAIN means above. Offsets 140 and 150 retain task-specific results.\n\n'
    for kind in ['transport', 'budget', 'ablation', 'horizon']:
        text += '### ' + names[kind] + '\n\n'
        text += '| Task | Controller | Setting | Standard | Perturbed 1 | Perturbed 2 | Perturbed |\n|---|---|---|---:|---:|---:|---:|\n'
        selected = [(key, group) for key, group in grouped.items() if key[0] == kind]
        selected.sort(key=lambda pair: (TASKS.index(pair[0][1]), intervention_order(kind, json.loads(pair[0][2]))))
        mean_groups = defaultdict(dict)
        for (_, task, cfg_json), group in selected:
            cfg = json.loads(cfg_json)
            setting = config_label(kind, cfg).replace(r'\%', '%')
            values = []
            for start in starts:
                row = group[start]
                values.append(fmt(row['success_percent']) if row['complete'] else f"pending ({row['completed']}/{row['expected']})")
            text += '| ' + ' | '.join([task, DISPLAY[cfg['arm']], setting] + values) + ' |\n'
            omitted = ['B'] if kind == 'horizon' else ['H', 'B']
            key = json.dumps({k: v for k, v in cfg.items() if k not in omitted}, sort_keys=True)
            mean_groups[key][task] = group
        for cfg_json, groups in sorted(mean_groups.items(), key=lambda pair: intervention_order(kind, json.loads(pair[0]))):
            if kind == 'horizon' and set(groups) != set(TASKS):
                continue
            assert set(groups) == set(TASKS)
            cfg = json.loads(cfg_json)
            if kind == 'horizon':
                setting = r'$H=' + str(cfg['H']) + '$'
            else:
                setting = config_label(kind, cfg) if kind != 'transport' else 'Task-specific goal offset'
            values = []
            for start in starts:
                rows = [groups[t][start] for t in TASKS]
                values.append(fmt(sum(r['success_percent'] for r in rows) / 4) if all(r['complete'] for r in rows) else 'pending')
            text += '| ' + ' | '.join(['Mean', DISPLAY[cfg['arm']], setting.replace(r'\%', '%')] + values) + ' |\n'
        text += '\n'
    return text

def training_audit():
    base = json.loads((ROOT.parent.parent / 'fresh_target_interface_study_v2/protocol.json').read_text())
    state = read('queue-state.json')['jobs']
    jobs = read('queue-manifest.json')['jobs']
    checks = []
    for task in TASKS:
        selection = read(f'training/{task}/selected.json')
        if not selection:
            checks.append(dict(task=task, complete=False))
            continue
        candidates = selection['candidates']
        assert len(candidates) == 4 and {(c['width'], c['residual']) for c in candidates} == {(w, r) for w in [512, 1024] for r in [0, 1]}
        assert all(c['complete'] for c in candidates)
        best = min(candidates, key=lambda c: c['validation_mse'])
        assert selection['selected']['checkpoint'] == best['checkpoint'] and not selection['control_results_used']
        queries = {q['episode'] for q in base['studies'][task]['confirm']}
        splits = []
        for candidate in candidates:
            protocol = json.loads((Path(candidate['checkpoint']).parent / 'protocol.json').read_text())
            train, valid = set(protocol['train_episodes']), set(protocol['validation_episodes'])
            receipt = json.loads((Path(protocol['cache']) / 'native_fp32_memory.json').read_text())
            memory = set(receipt['included_episodes'])
            assert not train & valid and not queries & (train | valid) and train | valid == memory
            assert len(valid) == round(.05 * len(memory))
            assert protocol['hidden_layers'] == 3 and protocol['batch'] == 1024 and protocol['maximum_updates'] == 50000
            assert protocol['learning_rate'] == 3e-4 and protocol['weight_decay'] == 1e-4
            assert receipt['complete'] and receipt['dtype'] == 'float32'
            splits.append((train, valid))
        assert all(s == splits[0] for s in splits)
        starts = [state[j['id']]['started_utc'] for j in jobs if j['kind'] == 'main' and j['task'] == task and state[j['id']].get('started_utc')]
        assert starts and datetime.fromisoformat(selection['selected_utc']) < min(datetime.fromisoformat(t) for t in starts)
        checks.append(dict(task=task, complete=True, configurations=4, training_episodes=len(splits[0][0]), validation_episodes=len(splits[0][1]), query_episodes=len(queries), overlapping_query_episodes=0, identical_validation_split=True, selected_by_validation_mse=True, selected_before_main_evaluation=True, width=best['width'], residual=best['residual'], best_step=best['best_step'], validation_mse=best['validation_mse']))
    result = dict(checked_utc=datetime.now(timezone.utc).isoformat(), complete=all(x['complete'] for x in checks), checks=checks)
    save('collected/training-audit.json', result)
    return result

def render_report():
    main = read('collected/main.json')
    rows = main['rows']
    def cell(task, arm, start):
        found = [r for r in rows if r['task'] == task and r['arm'] == arm and r['start'] == start]
        assert len(found) == 1
        return found[0]
    means = {}
    for start in ['Standard', 'Perturbed', 'Perturbed 1', 'Perturbed 2']:
        means[start] = {}
        for arm in ARMS:
            cells = [cell(t, arm, start) for t in TASKS]
            means[start][arm] = sum(c['success_percent'] for c in cells) / 4 if all(c['complete'] for c in cells) else None
    wins = []
    for task in TASKS:
        for start in ['Standard', 'Perturbed']:
            for learned, observed in [('CEM (learned target)', 'AP-CEM'), ('Rank (learned target)', 'AP-rank')]:
                a, b = cell(task, learned, start), cell(task, observed, start)
                if a['complete'] and b['complete'] and a['success_percent'] > b['success_percent']:
                    wins.append(dict(task=task, start=start, learned=learned, observed=observed, learned_success=a['success_percent'], observed_success=b['success_percent'], difference_pp=a['success_percent'] - b['success_percent']))
    saved = dict(collected_utc=main['collected_utc'], main_complete=main['complete'], means=means, learned_target_wins=wins)
    save('collected/main-interpretation.json', saved)
    text = '# Experiment summary\n\n'
    text += datetime.now(timezone.utc).isoformat() + '\n\n'
    text += '## Learned-target results that change the comparison\n\n'
    if main['complete']:
        standard, perturbed = means['Standard'], means['Perturbed']
        text += f"**The learned target is not uniformly worse than the observed target.** Standard-start Rank (learned target) averages {fmt(standard['Rank (learned target)'])}% versus {fmt(standard['AP-rank'])}% for AP-rank. Under perturbed starts these means are {fmt(perturbed['Rank (learned target)'])}% and {fmt(perturbed['AP-rank'])}%. AP-CEM averages {fmt(standard['AP-CEM'])}% / {fmt(perturbed['AP-CEM'])}% at standard / perturbed starts, versus {fmt(standard['CEM (learned target)'])}% / {fmt(perturbed['CEM (learned target)'])}% for learned-target CEM. These are descriptive task means, not significance claims.\n\n"
    text += '| Task | Start | Learned controller | Learned | Observed | Difference (pp) |\n|---|---|---|---:|---:|---:|\n'
    for w in wins:
        text += f"| {w['task']} | {w['start']} | {w['learned']} | {fmt(w['learned_success'])} | {fmt(w['observed_success'])} | +{fmt(w['difference_pp'])} |\n"
    text += '\nThese results assess a matched target-regression MLP; they do not reproduce SAGE or establish a general ranking of learned and retrieved subgoals. Both target sources can improve on final-goal scoring. The scientific claim is about target choice, not universal retrieval superiority.\n\n'
    text += '## Complete MAIN success rates (%)\n\nThe original 7,680 outcomes are reused. Each task/start uses its original 128 queries. Perturbed averages the two original perturbations within each query; Mean weights the four tasks equally.\n\n'
    if (ROOT/'collected/main_episodes.csv').exists():
        text += 'The individual binary outcomes, executed steps, prediction counts, and source files are available in `collected/main_episodes.csv`.\n\n'
    text += '| Task | Start | ' + ' | '.join(ARMS) + ' |\n|---|---|' + '---:|' * len(ARMS) + '\n'
    for task in TASKS + ['Mean']:
        for start in ['Standard', 'Perturbed', 'Perturbed 1', 'Perturbed 2']:
            values = [means[start][a] if task == 'Mean' else cell(task, a, start)['success_percent'] for a in ARMS]
            text += '| ' + ' | '.join([task, start] + [fmt(v) if v is not None else 'pending' for v in values]) + ' |\n'
    train = training_audit()
    text += '\n## Training and held-out-memory target quality\n\nAll four configurations per task use the same held-out memory episodes. Their training and validation episodes are disjoint from the control queries; selection precedes control evaluation and minimizes validation MSE in common latent units. See collected/training-audit.json for the checked episode counts and selected configurations.\n\n'
    text += '| Task | Width / output | Validation MSE per coordinate | Learned successor L2: mean / median | Retrieved successor L2: mean / median | Learned nearest-memory L2: mean / median |\n|---|---|---:|---|---|---|\n'
    for check in train['checks']:
        if not check['complete']:
            continue
        q = read(f"target_quality/{check['task']}/summary.json")
        assert q and q['complete'] and q['validation_samples'] == 4096 and not q['query_episodes_used']
        values = [' / '.join(f'{q[k][m]:.4f}' for m in ['mean', 'median']) for k in ['mlp_successor_squared_l2', 'retrieved_successor_squared_l2', 'mlp_nearest_memory_squared_l2']]
        output = 'residual' if check['residual'] else 'absolute'
        text += f"| {check['task']} | {check['width']} / {output} | {check['validation_mse']:.6f} | " + ' | '.join(values) + ' |\n'
    text += '\nSuccessor and nearest-memory distances are squared L2 sums over the latent coordinates. Retrieved targets have zero nearest-memory distance by construction; this is not independent evidence of current-state reachability. The learned target has lower mean held-out successor error on all four tasks, but does not have higher control success in every task or setting. Prediction-to-successor error and control utility should therefore be reported separately.\n\n'
    text += '### All target-model configurations\n\nValidation MSE is measured per latent coordinate at the best validation checkpoint. Worker minutes are measured training-process time under concurrent execution, excluding feature preparation; they are not end-to-end deployment cost.\n\n| Task | Width | Output | Best step | Validation MSE | Worker minutes | Selected |\n|---|---:|---|---:|---:|---:|---|\n'
    for task in TASKS:
        selection = read(f'training/{task}/selected.json')
        for candidate in selection['candidates']:
            text += '| ' + ' | '.join([task, str(candidate['width']), 'residual' if candidate['residual'] else 'absolute', str(candidate['best_step']), f"{candidate['validation_mse']:.6f}", fmt(candidate['elapsed_seconds'] / 60), 'yes' if candidate['checkpoint'] == selection['selected']['checkpoint'] else '']) + ' |\n'
    text += intervention_report()
    text += '## Simulator diagnostics and first-block interventions\n\n'
    for name in ['diagnostic', 'first_block']:
        result = read(f'collected/{name}.json')
        text += f"- {name}: {result['source_outcomes']} assigned task/query/start outcomes collected; complete={result['complete']}. Detailed rows: collected/{name}.json.\n"
    text += '\nThe collector checks exact replay of the original predictions and Direct continuations on policy-entered episodes. Endpoint diagnostics retain only starts for which all eight candidates complete five primitive actions, and report the eligible denominators.\n\n'
    diagnostic = read('collected/diagnostic.json')
    text += '| Task | Start | Eligible starts / queries | Predictor error | Displacement error | Predictor regret | Displacement regret | Direct regret |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for row in diagnostic['rows']:
        metrics = [f"{row[key]:.3f}" if row['complete'] else 'pending' for key in ['predictor_error', 'displacement_error', 'predictor_regret', 'displacement_regret', 'direct_regret']]
        text += '| ' + ' | '.join([row['task'], row['start'], f"{row['eligible_starts']} / {row['eligible_queries']}"] + metrics) + ' |\n'
    first = read('collected/first_block.json')
    selectors = ['Direct', 'Predictor-Final', 'Predictor-Observed', 'Predictor-Learned', 'Simulator-Final', 'Simulator-Observed', 'Simulator-Learned', 'Displacement-Observed']
    lookup = {(row['task'], row['start'], row['selector']): row for row in first['rows']}
    text += '\nFirst-block success (%) uses all assigned starts; subsequent actions come from Direct.\n\n'
    text += '| Task | Start | ' + ' | '.join(selectors) + ' |\n|---|---|' + '---:|' * len(selectors) + '\n'
    for task in TASKS:
        for start in ['Standard', 'Perturbed 1', 'Perturbed 2', 'Perturbed']:
            rows = [lookup[(task, start, arm)] for arm in selectors]
            text += '| ' + ' | '.join([task, start] + [fmt(row['success_percent']) if row['complete'] else 'pending' for row in rows]) + ' |\n'
    aliases = read('collected/reuse-equalities.json')
    text += f"## Reuse checks\n\nThe intervention collector verified {len(aliases['checks'])} explicit aliases to existing episode records (verified={aliases['verified']}). These include original-budget, original-horizon, default-ablation, and observed transport-reference cells. They are references to the same outcomes, not independent replications. Mechanism extraction reuses existing trajectories; no blanket replay of the original study was necessary.\n\n"
    state, manifest = read('queue-state.json'), read('queue-manifest.json')
    counts = defaultdict(Counter)
    for job in manifest['jobs']:
        counts[job['kind']][state['jobs'][job['id']]['status']] += 1
    text += '## Remaining execution\n\nScheduler status is a job count, not an experimental result. Incomplete cells have no reported success estimate.\n\n| Kind | Complete | Running | Queued | Failed |\n|---|---:|---:|---:|---:|\n'
    for kind, count in counts.items():
        text += '| ' + kind + ' | ' + ' | '.join(str(count[k]) for k in ['complete', 'running', 'queued', 'failed']) + ' |\n'
    failed = [{**{'job': k}, **s} for k, s in state['jobs'].items() if s['status'] == 'failed']
    text += '\nCurrent failures: ' + (json.dumps(failed, indent=2) if failed else 'none') + '.\n\n'
    text += retry_history(state)
    text += 'All prescribed transport, budget, ablation and goal-offset cells, the expanded mechanism analysis, final vector figures/tables, and the revised manuscript remain in scope. No partial results are substituted for these deliverables. The result cutoff remains 2026-09-26 12:00 Beijing time.\n'
    target = ROOT / 'REPORT.md'
    temporary = target.with_suffix('.md.tmp')
    temporary.write_text(text, encoding='utf-8')
    os.replace(temporary, target)
    return dict(main_complete=main['complete'], learned_target_wins=len(wins), training_audit_complete=train['complete'])

if __name__ == '__main__':
    print(json.dumps(render_report()))
