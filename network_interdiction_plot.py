"""
Plot the benchmark results CSV produced by run_benchmark.py (issue #6).

Draws two figures per run:

  <stem>_flow_reduction.pdf  flow reduction achieved by each annealing
                             solver, and by the classical baselines when the
                             CSV has them, relative to the exact optimum
  <stem>_qubo_energy.pdf     QUBO energy reached by each annealing solver,
                             relative to the exact QUBO ground state

where <stem> is the input CSV's file name without its extension, so
`--csv result_dwave_planar.csv` still writes the result_dwave_planar_*.pdf
names this script has always written.

This used to read a hard-coded `result_dwave_planar.csv` that was not checked
in and had no generating script, so the script could not be run at all. It now
defaults to the small checked-in sample and takes any CSV via --csv. A solver
whose columns are empty (typically QA, which needs a D-Wave QPU) is skipped
instead of drawing empty bars. The Greedy / Random_Greedy baseline columns
(issue #15) are optional, so CSVs written before they existed still plot.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas

DEFAULT_CSV = os.path.join('results', 'sample_results_sa.csv')

# Columns every row needs, whichever solvers were run.
REQUIRED_COLUMNS = ['V', 'E', 'B', 'Opt_NIP', 'Opt_QUBO', 'maxflow']

# label -> (objective column, greedy-repaired objective column, energy column)
SOLVERS = {
    'SA': ('SA_NIP', 'SA_NIP_Greedy', 'SA_QUBO'),
    'QA': ('QA_NIP(advantage1)', 'QA_NIP(advantage1)_Greedy',
           'QA_QUBO(advantage1)'),
}

# label -> column, for the classical baselines run_benchmark.py writes. These
# have one number per row (the surviving flow), not the solver triple.
BASELINES = {
    'Greedy': 'Greedy',
    'Random-Greedy': 'Random_Greedy',
    'LP-Round-Repair': 'LP_Round_Repair',
    'Lagrangian-Feasible': 'Lagrangian_Feasible',
    'Benders': 'Benders',
}

BASELINES.update({f'{rule}-{mode}': f'Decomp_{rule}_{mode}'
                  for rule in ('mincut', 'local', 'marginal', 'multiplier')
                  for mode in ('sequential', 'iterated')})

# Plotting order, and the colour each series has always been drawn in.
# Greedy takes the dark grey the original palette defined but never used;
# Random-Greedy is amber, since the palette's other spare (#ff2d55) is too
# close to SA-Greedy's red to tell apart.
SERIES_COLORS = {
    'QA': '#99aab5',
    'SA': '#007aff',
    'QA-Greedy': '#5ac8fa',
    'SA-Greedy': '#ff3b30',
    'Greedy': '#2c2f33',
    'Random-Greedy': '#ffcc00',
    'LP-Round-Repair': '#8e44ad',
    'Lagrangian-Feasible': '#16a085',
    'Benders': '#a65d2a',
}

SERIES_COLORS.update(dict(zip(
    (f'{rule}-{mode}' for rule in ('mincut', 'local', 'marginal', 'multiplier')
     for mode in ('sequential', 'iterated')),
    ['#1f77b4', '#aec7e8', '#2ca02c', '#98df8a', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94'])))

BAR_WIDTH = 0.25
GROUP_STEP = 1.5


def load_results(csv_path):
    """Read the results CSV, failing loudly on a CSV the plot cannot use."""
    frame = pandas.read_csv(csv_path, comment='#')
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise SystemExit(
            f"{csv_path} is missing required column(s): {', '.join(missing)}. "
            f"Generate it with run_benchmark.py (see README).")
    if frame.empty:
        raise SystemExit(f"{csv_path} has no data rows.")
    if (frame['maxflow'] == frame['Opt_NIP']).any():
        raise SystemExit(
            f"{csv_path} has a row where the optimum equals maxflow, i.e. the "
            f"budget buys no flow reduction at all; every ratio plotted here "
            f"is normalised by that reduction, so there is nothing to plot.")
    return frame


def available_solvers(frame):
    """
    Solvers that this CSV actually carries data for.

    A solver's columns are all present-but-empty when that pass was not run
    (run_benchmark.py leaves them blank rather than writing a sentinel).
    """
    present = []
    for label, columns in SOLVERS.items():
        if not all(c in frame.columns for c in columns):
            continue
        if frame[list(columns)].isna().all(axis=None):
            continue
        present.append(label)
    return present


def available_baselines(frame):
    """Baseline labels whose column is present and not entirely empty."""
    return [label for label, column in BASELINES.items()
            if column in frame.columns and not frame[column].isna().all()]


def compute_series(frame, solvers, baselines=()):
    """
    Turn the raw columns into the two ratios the figures show.

    flow_reduction is each solver's -- and each requested baseline's -- flow
    reduction as a fraction of the optimal flow reduction (1.0 = matched the
    exact optimum); energy_ratio is each solver's QUBO energy as a fraction
    of the exact ground-state energy. Baselines have no energy.
    """
    # Column-wise, not row-wise: iterrows() would upcast these integer counts
    # to float via the row's mixed dtype and label the axis "16.0 (1.0)".
    num_vars = frame['qubits'] if 'qubits' in frame else frame['V'] + 3 * frame['E']
    x_labels = [f'{v} ({b})' for v, b in zip(num_vars, frame['B'])]

    optimal_reduction = (frame['maxflow'] - frame['Opt_NIP']) / frame['maxflow']

    flow_reduction = {}
    energy_ratio = {}
    for label in solvers:
        nip_col, greedy_col, qubo_col = SOLVERS[label]

        # A 0 in the raw objective column means "this solver found nothing
        # usable for this row" -- the original script's convention, kept so
        # existing CSVs plot the same way. run_benchmark.py writes an empty
        # cell rather than a 0 for a missing value, precisely because 0 is
        # taken here to mean something else.
        reduction = (frame['maxflow'] - frame[nip_col]) / frame['maxflow']
        # Explicit modern diagnostics distinguish a genuine zero-flow sample
        # from the historical zero sentinel. Missing values remain missing.
        feasible = f'{label}_raw_feasible_count'
        legacy_zero = frame[nip_col] == 0
        if feasible in frame:
            legacy_zero &= frame[feasible].isna() | (frame[feasible] == 0)
        flow_reduction[label] = np.where(legacy_zero, 0.0, reduction / optimal_reduction)

        greedy_reduction = ((frame['maxflow'] - frame[greedy_col])
                            / frame['maxflow'])
        flow_reduction[f'{label}-Greedy'] = (greedy_reduction
                                             / optimal_reduction).to_numpy()

        energy_ratio[label] = (frame[qubo_col] / frame['Opt_QUBO']).to_numpy()

    for label in baselines:
        # No zero sentinel here: run_benchmark.py always writes a baseline's
        # real surviving flow, and 0 is a legitimate value for it.
        column = BASELINES[label]
        reduction = (frame['maxflow'] - frame[column]) / frame['maxflow']
        flow_reduction[label] = (reduction / optimal_reduction).to_numpy()

    return x_labels, flow_reduction, energy_ratio


def grouped_bar_positions(num_groups, num_series):
    """Bar centres and tick positions, with room for every optional series."""
    group_step = max(GROUP_STEP, BAR_WIDTH * (num_series + 1))
    base = np.arange(num_groups) * group_step
    positions = [base + i * BAR_WIDTH for i in range(num_series)]
    centres = base + BAR_WIDTH * (num_series - 1) / 2
    return positions, centres


def draw(series, x_labels, ylabel, ylim, out_path, reference_line=None):
    plt.figure(figsize=(12, 8))
    plt.tight_layout()
    plt.subplot(1, 1, 1)
    plt.tick_params(labelsize=14)

    positions, centres = grouped_bar_positions(len(x_labels), len(series))
    for (label, values), left in zip(series.items(), positions):
        plt.bar(left, values, width=BAR_WIDTH, edgecolor='white', label=label,
                color=SERIES_COLORS.get(label))

    if reference_line is not None:
        plt.axhline(y=reference_line, color='r', linestyle='dotted')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('#Variables (#Interdicted Edges)', fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xticks(centres, x_labels, rotation=45)
    plt.legend(prop={'size': 12}, loc='upper left', ncol=min(4, len(series)),
               fancybox=False, shadow=True)
    plt.tight_layout()
    plt.ylim(ylim)
    plt.savefig(out_path)
    plt.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--csv', default=DEFAULT_CSV,
        help=f'results CSV from run_benchmark.py (default: {DEFAULT_CSV})')
    parser.add_argument(
        '--outdir', default='.',
        help='directory to write the PDFs into (default: the current one)')
    args = parser.parse_args(argv)

    frame = load_results(args.csv)
    solvers = available_solvers(frame)
    baselines = available_baselines(frame)
    if not solvers and not baselines:
        raise SystemExit(
            f"{args.csv} has no results to plot: every SA_*/QA_* and baseline "
            f"column is empty. Re-run run_benchmark.py with --solvers sa and/or "
            f"qa, or with --baselines.")

    x_labels, flow_reduction, energy_ratio = compute_series(
        frame, solvers, baselines)

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.csv))[0]

    # Keep the original drawing order (QA before SA, plain before greedy) for
    # whichever solvers are present.
    flow_series = {label: flow_reduction[label]
                   for label in SERIES_COLORS if label in flow_reduction}
    energy_series = {label: energy_ratio[label]
                     for label in ('QA', 'SA') if label in energy_ratio}

    flow_path = os.path.join(args.outdir, f'{stem}_flow_reduction.pdf')
    draw(flow_series, x_labels, 'Flow Reduction from Maxflow against OPT',
         (0.0, 1.1), flow_path, reference_line=1)

    written = [flow_path]
    # Baselines have no QUBO energy, so a baselines-only CSV has no second
    # figure to draw.
    if energy_series:
        energy_path = os.path.join(args.outdir, f'{stem}_qubo_energy.pdf')
        draw(energy_series, x_labels, 'QUBO Energy Ratio against OPT',
             (0.9, 1.02), energy_path)
        written.append(energy_path)

    print(f'Plotted {len(frame)} row(s) from {args.csv} '
          f'({", ".join(solvers + baselines)}):')
    for path in written:
        print(f'  {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
