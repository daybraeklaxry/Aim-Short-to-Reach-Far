"""Publication-width plots; callers decide which complete studies to export."""
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from publication_tables import TASKS, NAMES, DISPLAY

COLORS = {'final': '#8C8C8C', 'learned': '#D98C2B', 'observed': '#1B8A8A', 'lewm': '#111111'}
STARTS = ['Standard', 'Perturbed']
HORIZONS = {'cube': [25, 50, 100, 150], 'pusht': [25, 50, 100, 140],
            'reacher': [25, 50, 100, 150], 'tworoom': [25, 50, 100]}
TARGET_ARMS = ['cem_final', 'cem_learned', 'cem_observed']
HORIZON_ARMS = TARGET_ARMS + ['lewm', 'rank_final', 'rank_learned', 'rank_observed']


def success_axis(ax):
    ax.set_ylim(-2, 102)
    ax.set_yticks([0, 50, 100])
    ax.grid(axis='y', color='#E9E9E9', linewidth=.5)
    ax.set_axisbelow(True)


def task_start_grid(height, starts):
    fig, axes = plt.subplots(len(starts), 4, figsize=(5.5, height), sharey=True,
                             squeeze=False, layout='constrained')
    for i, start in enumerate(starts):
        for j, task in enumerate(TASKS):
            ax = axes[i, j]
            success_axis(ax)
            if i == 0:
                ax.set_title(NAMES[task])
            if j == 0:
                ax.set_ylabel('Success (%)' if len(starts) == 1 else start + '\nSuccess (%)')
    return fig, axes


def compute_figure(points, references, starts=STARTS):
    fig, axes = task_start_grid(2.2 if len(starts) == 1 else 3.7, starts)
    for i, start in enumerate(starts):
        for j, task in enumerate(TASKS):
            ax = axes[i, j]
            for arm in TARGET_ARMS:
                rows = sorted((r for r in points if r['task'] == task and r['start'] == start
                               and r['arm'] == arm), key=lambda r: r['iterations'])
                ax.plot([r['work'] for r in rows], [r['success'] for r in rows],
                        color=COLORS[arm.split('_')[1]], marker='o', markersize=3,
                        label=DISPLAY[arm])
            ref = next(r for r in references if r['task'] == task and r['start'] == start)
            ax.plot(ref['work'], ref['success'], color=COLORS['lewm'], marker='*',
                    markersize=7, linestyle='none', label='LeWM planner')
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=3))
            ax.xaxis.set_minor_formatter(NullFormatter())
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=2,
               frameon=False, fontsize=8, columnspacing=1.6)
    fig.supxlabel('Mean prediction blocks per episode', fontsize=8)
    return fig


def horizon_figure(points, starts=STARTS):
    fig, axes = task_start_grid(2.45 if len(starts) == 1 else 3.95, starts)
    for i, start in enumerate(starts):
        for j, task in enumerate(TASKS):
            ax = axes[i, j]
            for arm in HORIZON_ARMS:
                rows = sorted((r for r in points if r['task'] == task and r['start'] == start
                               and r['arm'] == arm), key=lambda r: r['H'])
                ax.plot([r['H'] for r in rows], [r['success'] for r in rows],
                        color=COLORS['lewm' if arm == 'lewm' else arm.split('_')[1]],
                        linestyle='--' if arm.startswith('rank') else '-',
                        marker='*' if arm == 'lewm' else 's' if arm.startswith('rank') else 'o',
                        markersize=5 if arm == 'lewm' else 3, label=DISPLAY[arm])
            ax.set_xticks(HORIZONS[task])
            ax.set_xlim(18, max(HORIZONS[task]) + 7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    lookup = dict(zip(HORIZON_ARMS, zip(handles, labels)))
    # Each legend column groups the synthesis and ranking rules for one target.
    order = ['cem_final', 'rank_final', 'lewm', 'cem_learned', 'rank_learned',
             'cem_observed', 'rank_observed']
    fig.legend([lookup[a][0] for a in order], [lookup[a][1] for a in order],
               loc='outside upper center', ncol=3, frameon=False, fontsize=8,
               columnspacing=1.2, handlelength=2)
    fig.supxlabel(r'Goal offset $H$', fontsize=8)
    return fig


def teaser_figure(points):
    # Embedded at native size beside the 190-point shared-prediction concept.
    fig, ax = plt.subplots(figsize=(196 / 72, 150 / 72))
    fig.subplots_adjust(left=.20, right=.975, bottom=.21, top=.685)
    for arm in TARGET_ARMS + ['lewm']:
        rows = sorted((r for r in points if r['arm'] == arm), key=lambda r: r['H'])
        ax.plot([r['H'] for r in rows], [r['success'] for r in rows],
                marker='*' if arm == 'lewm' else 'o', markersize=6 if arm == 'lewm' else 3,
                color=COLORS['lewm' if arm == 'lewm' else arm.split('_')[1]], label=DISPLAY[arm])
    success_axis(ax)
    ax.set_xlabel(r'Goal offset $H$')
    ax.set_ylabel('Success (%)')
    ax.set_xticks(HORIZONS['pusht'])
    ax.set_xlim(18, 147)
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 2, 1, 3]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               frameon=False, fontsize=8, loc='upper center',
               bbox_to_anchor=(.54, .955), ncol=2, handlelength=1.5,
               columnspacing=.8, handletextpad=.45, labelspacing=.25)
    fig.text(0, .94, '(b) PushT', fontsize=8, fontweight='bold', va='baseline')
    return fig
