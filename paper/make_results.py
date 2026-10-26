"""Generate manuscript tables and presentation figures from saved results."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = Path(__file__).resolve().parent
METHODS = ['fixed', 'ibm', 'gat', 'optimized', 'rws']
NAMES = ['Fixed donor', 'Nearest donor', 'GAT', 'Target trained', 'Warm start']
PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']


def write_table(path, caption, label, header, body):
    text = '\\begin{table}[tb]\n\\centering\n\\small\n'
    text += f'\\caption{{{caption}}}\n\\label{{{label}}}\n'
    text += '\\begin{tabular}{llrrrr}\n\\toprule\n'
    text += header + ' \\\\\n\\midrule\n'
    text += '\n'.join(body)
    text += '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    path.write_text(text)


def plot_transfer(ax, transfer, reps, metric, ylabel, labels):
    """One point per method and graph, with means over the same six graphs."""
    pivot = transfer[transfer.reps == reps].pivot(
        index='group', columns='method', values=metric)[METHODS].sort_index()
    for (group, values), color in zip(pivot.iterrows(), PALETTE):
        ax.plot(range(5), values, marker='o', markersize=3.5, linewidth=1,
                color=color, alpha=0.75, label=group)
    ax.scatter(range(5), pivot.mean(), marker='D', s=35, color='black',
               label='Mean', zorder=10)
    ax.set_xticks(range(5), labels, rotation=25)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.2)


def main():
    generated = PAPER / 'generated'
    figures = PAPER / 'figures'
    generated.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    transfer = pd.read_csv(ROOT / 'results/parameter_transfer.csv', comment='#')
    metadata = json.loads((ROOT / 'results/parameter_transfer.json').read_text())
    assert metadata['complete'] and len(transfer) == 60
    assert not transfer.duplicated(['group', 'reps', 'method']).any()
    rows = []
    for reps in (1, 2):
        for method, name in zip(METHODS, NAMES):
            data = transfer[(transfer.reps == reps) & (transfer.method == method)]
            assert len(data) == 6
            mean = data.mean(numeric_only=True)
            rows.append(f'{reps} & {name} & {mean.gap_closed:.4f} & '
                        f'{mean.p_optimal:.4f} & {mean.p_feasible:.4f} & '
                        f'{mean.target_evaluations:.1f} \\\\')
        if reps == 1:
            rows.append('\\midrule')
    write_table(generated / 'transfer_table.tex',
                'Statevector comparison. Each row averages the same six held-out graphs. '
                '$P_{\\mathrm{opt}}$ uses exact state probabilities; '
                '$P_{\\mathrm{feas}}$ uses 4096 sampled bit strings. '
                'Evaluations count target quantum-angle optimizer calls.',
                'tab:transfer',
                '$p$ & Method & $G_E$ & $P_{\\mathrm{opt}}$ & '
                '$P_{\\mathrm{feas}}$ & Evaluations', rows)

    mid = pd.read_csv(ROOT / 'results/qaoa_mid.csv', comment='#')
    common = mid.loc[mid.angles == 'train', ['instance', 'B']].drop_duplicates()
    paired = mid.merge(common, on=['instance', 'B'], validate='many_to_one')
    assert len(common) == 10 and len(paired) == 60
    rows = []
    for reps in (1, 2, 3):
        for angles, name in (('donor', 'Donor'), ('train', 'Target trained')):
            data = paired[(paired.reps == reps) & (paired.angles == angles)]
            mean = data.mean(numeric_only=True)
            optimal = int((data.flow_after_repair == data.opt_nip).sum())
            rows.append(f'{reps} & {name} & {mean.gap_closed:.4f} & '
                        f'{mean.shot_p_optimal:.4f} & {mean.p_feasible:.4f} & '
                        f'{optimal}/10 \\\\')
    write_table(generated / 'mid_table.tex',
                'Paired MPS comparison on ten instance-budget pairs with at most 81 qubits. '
                'The maximum bond dimension is 32. Both probabilities are sampled estimates. '
                'The last column counts optimal repaired solutions.',
                'tab:mid',
                '$p$ & Schedule & $G_E$ & $P_{\\mathrm{opt}}$ & '
                '$P_{\\mathrm{feas}}$ & Repaired', rows)

    plt.rcParams.update({'font.family': 'serif', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    metrics = ['gap_closed', 'p_optimal']
    ylabels = ['Energy gap closed', 'Optimal-interdiction probability']
    for reps, suffix in ((1, 'one'), (2, 'two')):
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), layout='constrained')
        for ax, metric, ylabel in zip(axes, metrics, ylabels):
            plot_transfer(ax, transfer, reps, metric, ylabel,
                          ['Fixed', 'Nearest', 'GAT', 'Trained', 'Warm'])
        axes[0].set_ylim(0.45, 1.0)
        axes[1].set_ylim(bottom=0)
        axes[0].set_title('(a) Energy')
        axes[1].set_title('(b) Interdiction quality')
        fig.savefig(figures / f'transfer_depth_{suffix}.pdf')
        fig.savefig(figures / f'transfer_depth_{suffix}.png', dpi=220)
        plt.close(fig)

    # Slide layout: depths in columns, metrics in rows, identical axis scales.
    with plt.rc_context({'font.family': 'sans-serif', 'font.size': 13}):
        fig, axes = plt.subplots(2, 2, figsize=(13.333, 7.5),
                                 sharex=True, sharey='row')
        for column, reps in enumerate((1, 2)):
            for row, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
                plot_transfer(axes[row, column], transfer, reps, metric,
                              ylabel if column == 0 else '',
                              ['Fixed', 'Nearest', 'GAT', 'Trained', 'RWS'])
            axes[0, column].set_title(f'Depth $p={reps}$', fontsize=18)
            axes[0, column].set_ylim(0.45, 1.0)
            axes[1, column].set_ylim(0, 0.30)
        axes[1, 0].set_ylabel('Optimal-interdiction\nprobability')
        fig.suptitle('Five QAOA methods on the same six test networks', fontsize=20, y=0.98)
        fig.text(0.5, 0.925, 'Statevector simulation | 18\u201321 qubits | Budget B = 1',
                 ha='center', fontsize=12)
        fig.text(0.5, 0.035,
                 'Colored lines: individual networks (same colors in all panels). '
                 'Black diamonds: means. Higher is better.', ha='center', fontsize=11)
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.17, top=0.84,
                            hspace=0.13, wspace=0.08)
        fig.savefig(figures / 'transfer_depth_comparison.pdf')
        fig.savefig(figures / 'transfer_depth_comparison.png', dpi=300)
        fig.savefig(figures / 'transfer_depth_comparison.svg')
        plt.close(fig)
    sources = ['results/parameter_transfer.csv', 'results/parameter_transfer.json',
               'results/qaoa_mid.csv']
    manifest = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    (generated / 'data_sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Generated two tables, depth-one/two and comparison figures, and input checksums.')


if __name__ == '__main__':
    main()
