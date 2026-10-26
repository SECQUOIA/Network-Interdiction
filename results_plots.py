"""
Figures for notebooks/analysis.ipynb, the results notebook.

Each plot_* function reads committed results (results/*.csv) through the
load_* helpers and returns one matplotlib figure; the notebook explains it
and saves it to figures/. Figures carry no titles: the notebook text states
each takeaway, and tests/test_analysis_notebook.py checks every number it
quotes against the data.

Solution quality is measured on the original objective: the max flow f
that survives an interdiction within budget, against the exact optimum f*,

    gap = f / f* - 1        (0 = optimal; 0.25 = 25% more flow survives),

the "approximation ratio minus one" of the quantum-preconditioning
notebook. Every optimum here is positive. QAOA states are scored by where
their probability goes over interdiction plans (results/qaoa_outcomes.csv,
from qaoa_outcomes.py), not by energy: the penalized energy rewards the
empty plan, which interdicts nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import network_interdiction as ni
import utils
from utils import (C_ALERT, C_GREEDY, C_GUROBI, C_RANDOM, C_REFERENCE, C_SA,
                   FAMILY_MARKERS, FAMILY_ORDER, GRID_GREY, QAOA_DEPTH_COLORS,
                   QAOA_INSTANCE_STYLE, QAOA_BUDGET_LINESTYLE, framed_legend,
                   style_axes)

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "results"
FAMILY_SHORT = {"planar": "planar", "Erdős–Rényi": "ER", "k-NN": "k-NN"}
_FAMILY_FROM_NAME = {"planar": "planar", "erdos": "Erdős–Rényi", "knn": "k-NN"}

# Classical methods on the large instances:
# label -> (flow column, time columns, colour, marker). Every time is when the
# method's answer was ready. For Gurobi (time columns None) that is when its
# MIPSOL callback first reported the optimal incumbent (Gurobi_trajectory);
# Gurobi_time_sec runs on until optimality is proved.
GUROBI = "Gurobi (exact MILP)"
CLASSICAL = {
    GUROBI: ("Opt_NIP", None, C_GUROBI, "D"),
    "Greedy from scratch": ("Greedy", ["Greedy_time_sec"], C_GREEDY, "s"),
    "SA + greedy repair": ("SA_NIP_Greedy", ["SA_time_sec", "SA_repair_time_sec"],
                           C_SA, "o"),
    "Random start + repair": ("Random_Greedy", ["Random_Greedy_time_sec"],
                              C_RANDOM, "^"),
}
HEURISTICS = [label for label in CLASSICAL if label != GUROBI]

# Where a QAOA state's probability goes (qaoa_outcomes.OUTCOMES order).
OUTCOME_STYLE = {
    "optimal": ("Optimal plan", "#009E73"),
    "suboptimal": ("Within budget, not optimal", "#56B4E9"),
    "empty": ("Empty plan (nothing cut)", "#CC79A7"),
    "over_budget": ("Over budget", "#D55E00"),
}
TRANSFER_METHODS = {
    "fixed": "Fixed\ndonor",
    "ibm": "Nearest\ndonor (IBM)",
    "gat": "GAT",
    "optimized": "Target-\ntrained",
    "rws": "RWS\nwarm start",
}

# Gap bins of the classical heat map, in percent.
GAP_BINS = [-1e-9, 1e-9, 5, 20, 50, 100, np.inf]
GAP_BIN_LABELS = ["optimal", "< 5%", "5–20%", "20–50%", "50–100%", "> 100%"]
GAP_BIN_COLOURS = ["#E3F1E6", "#FFF3B0", "#FDCB6E", "#F39C4F", "#D9542C", "#8E1B10"]


# ── data ────────────────────────────────────────────────────────────────────
def qubo_qubits(V, E):
    """Variables of nip_to_qubo(): x, gamma and w per arc, pi per inner node."""
    return V - 2 + 3 * E


def short_name(instance: str) -> str:
    """'testcases/50_0.15.erdosflow' -> 'ER 50'."""
    stem = Path(instance).name
    family = utils._FAMILY_FROM_EXTENSION[stem.rsplit(".", 1)[1]]
    return f"{FAMILY_SHORT[family]} {stem.split('.')[0].split('_')[0]}"


def load_large() -> pd.DataFrame:
    """
    The large instances: the nine networks of testcases/ (50-125 nodes) at
    27 budgets, from results/slide_instances_sa.csv, with family, qubits and
    a case label.
    """
    df = utils.load_results(RESULTS / "slide_instances_sa.csv")
    df["qubits"] = qubo_qubits(df["V"], df["E"])
    df["name"] = df["instance"].map(short_name)
    df["case"] = df["name"] + " B=" + df["B"].astype(str)
    return df


def load_mid() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "mid_instances.csv", comment="#")
    df["family"] = df["family"].map(_FAMILY_FROM_NAME)
    df["optimal_cut"] = df["optimal_cut"].map(json.loads)
    return df


def load_outcomes() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "qaoa_outcomes.csv", comment="#")


def load_sample_quality() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "qaoa_sample_quality.csv", comment="#")


def load_qaoa_mid() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "qaoa_mid.csv", comment="#")
    df["family"] = df["family"].map(_FAMILY_FROM_NAME)
    return df


def load_decomposition() -> pd.DataFrame:
    """results/decomposition_large.csv with the gap of each decomposed cut."""
    df = pd.read_csv(RESULTS / "decomposition_large.csv", comment="#")
    df["gap"] = df["flow"] / df["optimum"] - 1
    df["family"] = df["instance"].str.rsplit(".", n=1).str[1].map(
        utils._FAMILY_FROM_EXTENSION)
    return df


def classical_long(large: pd.DataFrame) -> pd.DataFrame:
    """One row per (case, method) a method ran on: flow, gap and wall time."""
    rows = []
    for label, (column, times, _colour, _marker) in CLASSICAL.items():
        ran = large[column].notna()
        sub = large[ran]
        rows.append(pd.DataFrame({
            "case": sub["case"], "family": sub["family"], "method": label,
            "flow": sub[column], "gap": sub[column] / sub["Opt_NIP"] - 1,
            "time_sec": (sub.apply(utils.gurobi_time_to_optimum, axis=1)
                         if times is None else sub[times].sum(axis=1)),
        }))
    return pd.concat(rows, ignore_index=True)


def classical_summary(large: pd.DataFrame) -> pd.DataFrame:
    """Per method: cases run, optimal count, median and worst gap, median time."""
    long = classical_long(large)
    table = long.groupby("method", sort=False).agg(
        cases=("gap", "size"),
        optimal=("gap", lambda gap: int((gap.abs() < 1e-12).sum())),
        median_gap=("gap", "median"), worst_gap=("gap", "max"),
        median_time_sec=("time_sec", "median"))
    return table


# ── 1. the problem ──────────────────────────────────────────────────────────
EXAMPLE_CASES = [("testcases/mid/planar_8_s0.flow", 2),
                 ("testcases/mid/erdos_8_0.45_s0.flow", 2),
                 ("testcases/mid/knn_11_2_s0.flow", 2)]


def plot_example_instances(mid: pd.DataFrame, cases=EXAMPLE_CASES) -> plt.Figure:
    """
    One mid-size network per family with its optimal interdiction: arc width
    grows with capacity, cut arcs are red.
    """
    fig, axes = plt.subplots(1, len(cases), figsize=(5.4 * len(cases), 5.6))
    for ax, (instance, B) in zip(np.atleast_1d(axes), cases):
        row = mid[(mid["instance"] == instance) & (mid["B"] == B)].iloc[0]
        di_g, src, tgt = ni.read_flow_file(REPO / instance)
        position = nx.kamada_kawai_layout(di_g.to_undirected())
        cut = {tuple(arc) for arc in row["optimal_cut"]}
        capacity = np.array([di_g[u][v]["weight"] for u, v in di_g.edges])
        width = 0.6 + 3.4 * capacity / capacity.max()
        kept = [(arc, w) for arc, w in zip(di_g.edges, width) if arc not in cut]
        nx.draw_networkx_edges(di_g, position, ax=ax,
                               edgelist=[arc for arc, _ in kept],
                               width=[w for _, w in kept], edge_color="#9AA3AD",
                               arrowsize=10, connectionstyle="arc3,rad=0.08",
                               node_size=260)
        nx.draw_networkx_edges(di_g, position, ax=ax, edgelist=sorted(cut),
                               width=4.2, edge_color=C_ALERT, arrowsize=14,
                               connectionstyle="arc3,rad=0.08", node_size=260)
        inner = [node for node in di_g.nodes if node not in (src, tgt)]
        nx.draw_networkx_nodes(di_g, position, nodelist=inner, ax=ax,
                               node_size=110, node_color="white",
                               edgecolors="#4A4A4A", linewidths=1.0)
        nx.draw_networkx_nodes(di_g, position, nodelist=[src, tgt], ax=ax,
                               node_size=420, node_color=[C_RANDOM, C_GUROBI])
        nx.draw_networkx_labels(di_g, position, {src: "s", tgt: "t"}, ax=ax,
                                font_color="white", font_weight="bold")
        ax.text(0.5, -0.02, f"{row['family']}: {row['V']} nodes, {row['E']} arcs\n"
                f"max flow {row['maxflow']} → {row['Opt_NIP']} after cutting {B} arcs",
                transform=ax.transAxes, ha="center", va="top", fontsize=12)
        ax.set_axis_off()
    fig.tight_layout()
    return fig


def plot_problem_sizes(large: pd.DataFrame, mid: pd.DataFrame,
                       outcomes: pd.DataFrame,
                       decomposition: pd.DataFrame) -> plt.Figure:
    """
    QUBO qubits against arcs for every instance set, with the limits that
    decide where each can run, and the largest piece of each large instance
    after the 156-qubit decomposition.
    """
    tiny = outcomes[outcomes["source"] == "tiny"].drop_duplicates("instance")
    tiny_arcs = [ni.read_flow_file(REPO / path)[0].number_of_edges()
                 for path in tiny["instance"]]
    mid_sizes = mid.drop_duplicates("instance")
    large_sizes = large.drop_duplicates("instance")
    groups = [
        ("Tiny (exact statevector QAOA)", tiny_arcs, tiny["qubits"],
         ["tiny"] * len(tiny), "#B5648F"),
        ("Mid-size (MPS QAOA)", mid_sizes["E"], mid_sizes["qubits"],
         mid_sizes["family"], utils.C_ACCENT_DARK),
        ("Large (Gurobi, SA, decomposition)", large_sizes["E"],
         large_sizes["qubits"], large_sizes["family"], C_GUROBI),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for label, arcs, qubits, families, colour in groups:
        for family in [*FAMILY_ORDER, "tiny"]:
            mask = np.asarray(families) == family
            if mask.any():
                ax.scatter(np.asarray(arcs)[mask], np.asarray(qubits)[mask],
                           s=70, color=colour,
                           marker=FAMILY_MARKERS.get(family, "P"),
                           edgecolor="white", linewidth=0.8, zorder=3)
    fitted = (decomposition[(decomposition["setting"] == "fit156")
                            & (decomposition["rule"] != "no fit")]
              .groupby("instance")["max_piece_qubits"].max())
    for _, row in large_sizes.iterrows():
        if row["instance"] in fitted.index:
            piece = fitted[row["instance"]]
            ax.plot([row["E"], row["E"]], [piece, row["qubits"]], ":",
                    color=C_REFERENCE, linewidth=1.1, zorder=1)
            ax.scatter(row["E"], piece, s=70, facecolor="white",
                       edgecolor=C_GUROBI, marker=FAMILY_MARKERS[row["family"]],
                       linewidth=1.3, zorder=3)
    # The large networks crowd each other, so their names sit in a column to
    # the left, in the same top-to-bottom order as the points, with leaders.
    ordered = large_sizes.sort_values("qubits", ascending=False)
    label_y = np.geomspace(ordered["qubits"].max() * 1.45,
                           ordered["qubits"].min() * 0.62, len(ordered))
    for y, (_, row) in zip(label_y, ordered.iterrows()):
        fits = row["instance"] in fitted.index
        ax.annotate(row["name"] if fits else
                    f"{row['name']}: no split fits 155 qubits",
                    (row["E"], row["qubits"]),
                    xytext=(ordered["E"].min() / 1.7, y), ha="right", va="center",
                    fontsize=9.5, color=C_GUROBI if fits else C_ALERT,
                    arrowprops={"arrowstyle": "-", "color": "#A0A0A0",
                                "linewidth": 0.8, "shrinkA": 3, "shrinkB": 4})
    for level, text, colour in ((24, "exact statevector limit here (24 qubits)",
                                 C_REFERENCE),
                                (156, "IBM Heron device (156 qubits)", C_ALERT)):
        ax.axhline(level, color=colour, linestyle="--", linewidth=1.2)
        ax.text(0.01, level * 1.08, text, fontsize=10, color=colour,
                transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(None, ordered["qubits"].max() * 1.9)
    ax.set_xlabel("arcs in the network")
    ax.set_ylabel("QUBO variables = qubits")
    style_axes(ax)
    handles = [Line2D([], [], linestyle="",
                      marker="P" if np.asarray(families)[0] == "tiny" else "o",
                      color=colour, markersize=8, label=label)
               for label, _arcs, _qubits, families, colour in groups]
    handles.append(Line2D([], [], linestyle="", marker="o", markerfacecolor="white",
                          markeredgecolor=C_GUROBI, markersize=8,
                          label="largest piece, 156-qubit decomposition"))
    handles += [Line2D([], [], linestyle="", marker=FAMILY_MARKERS[f],
                       color=C_REFERENCE, markersize=8, label=f)
                for f in FAMILY_ORDER]
    framed_legend(fig, handles, ncol=3, y=0.0)
    fig.tight_layout()
    return fig


# ── 2. classical methods ────────────────────────────────────────────────────
def plot_classical_gaps(large: pd.DataFrame) -> plt.Figure:
    """
    Heat map of each heuristic's gap on each large case (Gurobi's row would
    be all optimal). Columns are grouped by instance, instances by family.
    """
    matrix = np.array([100 * (large[CLASSICAL[label][0]] / large["Opt_NIP"] - 1)
                       for label in HEURISTICS])
    cmap = ListedColormap(GAP_BIN_COLOURS).with_extremes(bad="#EDEDED")
    norm = BoundaryNorm(GAP_BINS[:-1] + [1e9], cmap.N)
    fig, ax = plt.subplots(figsize=(15.5, 4.6))
    ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect="auto")
    for (i, j), value in np.ndenumerate(matrix):
        if np.isnan(value):
            text, colour = "–", C_REFERENCE
        elif abs(value) < 1e-9:
            text, colour = "✓", "#2E7D32"
        else:
            text = f"{value:.0f}" if value >= 1 else f"{value:.1f}"
            colour = "white" if value >= 50 else "#222222"
        ax.text(j, i, text, ha="center", va="center", fontsize=9, color=colour)
    ax.set_yticks(range(len(HEURISTICS)), HEURISTICS)
    ax.set_xticks(range(len(large)), [f"B={b}" for b in large["B"]],
                  rotation=90, fontsize=9)
    ax.tick_params(length=0)
    ax.grid(False)
    # Instance labels under their columns; lines between instances/families.
    for name, group in large.reset_index(drop=True).groupby("name", sort=False):
        start, stop = group.index.min(), group.index.max()
        ax.text((start + stop) / 2, -0.6, name, ha="center", va="bottom",
                fontsize=10.5)
        if start > 0:
            new_family = large["family"].iloc[start] != large["family"].iloc[start - 1]
            ax.axvline(start - 0.5, color="black" if new_family else "white",
                       linewidth=2.2 if new_family else 3)
    handles = [Patch(facecolor=colour, edgecolor=GRID_GREY, label=label)
               for label, colour in zip(GAP_BIN_LABELS, GAP_BIN_COLOURS)]
    handles.append(Patch(facecolor="#EDEDED", edgecolor=GRID_GREY,
                         label="not run (no valid λ)"))
    framed_legend(fig, handles, ncol=7, y=0.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def plot_time_vs_gap(large: pd.DataFrame) -> plt.Figure:
    """
    Time until each method's answer is found against its gap, every case and
    method (Gurobi: its first optimal incumbent); large markers are medians.
    """
    long = classical_long(large)
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for label, (_column, _times, colour, marker) in CLASSICAL.items():
        sub = long[long["method"] == label]
        ax.scatter(sub["time_sec"], 100 * sub["gap"], s=34, color=colour,
                   marker=marker, alpha=0.55, edgecolor="none", label=label)
        ax.scatter(sub["time_sec"].median(), 100 * sub["gap"].median(), s=190,
                   color=colour, marker=marker, edgecolor="black", linewidth=1.4,
                   zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(-0.5, None)
    ax.set_yticks([0, 1, 10, 100, 1000], ["0", "1", "10", "100", "1000"])
    ax.set_xlabel("wall time until the answer is found (s)")
    ax.set_ylabel("gap to optimum, %")
    style_axes(ax)
    handles = [Line2D([], [], linestyle="", marker=m, color=c, markersize=8, label=l)
               for l, (_col, _t, c, m) in CLASSICAL.items()]
    handles.append(Line2D([], [], linestyle="", marker="o", color="white",
                          markeredgecolor="black", markersize=11, label="median"))
    framed_legend(fig, handles, ncol=5, y=0.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


# ── 3. QAOA on small instances ──────────────────────────────────────────────
def _stack(ax, x, shares, width=None):
    """Stacked outcome bars (width given) or areas (width None) at positions x."""
    bottom = np.zeros(len(x))
    for outcome, (label, colour) in OUTCOME_STYLE.items():
        values = np.asarray(shares[f"p_{outcome}"], dtype=float)
        if width is None:
            ax.fill_between(x, bottom, bottom + values, color=colour, alpha=0.9,
                            linewidth=0, label=label)
        else:
            ax.bar(x, values, width, bottom=bottom, color=colour,
                   edgecolor="white", linewidth=0.6, label=label)
        bottom = bottom + values


def plot_qaoa_depth(outcomes: pd.DataFrame) -> plt.Figure:
    """
    The six tiny cases at depths 0 (the uniform start) to 3: P(optimal plan)
    per case, and where the probability goes on average.
    """
    tiny = outcomes[outcomes["source"] == "tiny"].copy()
    tiny["name"] = tiny["case"].str.split(" ").str[0]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for case, sub in tiny.groupby("case", sort=False):
        sub = sub.sort_values("reps")
        colour, marker = QAOA_INSTANCE_STYLE[sub["name"].iloc[0]]
        left.plot(sub["reps"], sub["p_optimal"], marker=marker, color=colour,
                  linestyle=QAOA_BUDGET_LINESTYLE[int(sub["B"].iloc[0])],
                  linewidth=2, markersize=7, label=f"{case} ({sub['qubits'].iloc[0]} q)")
    left.set_xticks([0, 1, 2, 3], ["0\n(uniform)", "1", "2", "3"])
    left.set_xlabel("QAOA depth p")
    left.set_ylabel("Probability of optimal plan")
    left.set_ylim(0, None)
    left.legend(fontsize=9, ncol=2, loc="upper left")
    style_axes(left)
    mean = tiny.groupby("reps")[[f"p_{o}" for o in OUTCOME_STYLE]].mean()
    _stack(right, mean.index.to_numpy(), mean)
    right.set_xticks([0, 1, 2, 3], ["0\n(uniform)", "1", "2", "3"])
    right.set_xlim(0, 3)
    right.set_ylim(0, 1)
    right.set_xlabel("QAOA depth p")
    right.set_ylabel("Probability")
    style_axes(right, ygrid=False)
    handles = [Patch(color=colour, label=label)
               for label, colour in OUTCOME_STYLE.values()]
    framed_legend(fig, handles, ncol=4, y=0.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def plot_qaoa_sample_quality(quality: pd.DataFrame) -> plt.Figure:
    """Mean expected best gap with SEM across cases; one line per tiny depth."""
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for reps, sub in quality.groupby("reps", sort=True):
        summary = sub.groupby("measurements")["expected_best_gap"].agg(["mean", "sem"])
        ax.errorbar(
            summary.index.to_numpy(), 100 * summary["mean"].to_numpy(),
            yerr=100 * summary["sem"].to_numpy(), capsize=3, elinewidth=1,
            color=C_REFERENCE if reps == 0 else QAOA_DEPTH_COLORS[reps],
            linestyle="--" if reps == 0 else "-", linewidth=2,
            marker="o", markersize=4,
            label="Uniform random guessing" if reps == 0 else f"QAOA p = {reps}")
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 1000, 4096], ["1", "10", "100", "1,000", "4,096"])
    ax.set_xlim(1, quality["measurements"].max())
    ax.set_ylim(0, None)
    ax.set_xlabel("Number of independent measurements")
    ax.set_ylabel("Expected best gap to optimum, %")
    style_axes(ax)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_transfer_outcomes(outcomes: pd.DataFrame) -> plt.Figure:
    """
    The six held-out transfer graphs (21 qubits, B=1): mean outcome shares
    per method at p = 1 and 2, the initial states for reference, and each
    graph's P(optimal) as a dot.
    """
    transfer = outcomes[outcomes["source"] == "transfer"]
    columns = [f"p_{o}" for o in OUTCOME_STYLE]
    bars = [("uniform", 0, "Uniform\nstart")] + \
        [(method, None, label) for method, label in TRANSFER_METHODS.items()]
    bars.insert(-1, ("rws", 0, "RWS start\n(p = 0)"))
    depths = sorted(d for d in transfer["reps"].unique() if d > 0)
    fig, axes = plt.subplots(1, len(depths), figsize=(15.5, 5.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, depth in zip(axes, depths):
        means, dots = [], []
        for method, reps, _label in bars:
            sub = transfer[(transfer["method"] == method)
                           & (transfer["reps"] == (depth if reps is None else reps))]
            means.append(sub[columns].mean())
            dots.append(sub["p_optimal"].to_numpy())
        x = np.arange(len(bars))
        _stack(ax, x, pd.DataFrame(means), width=0.7)
        for xi, values in zip(x, dots):
            ax.scatter(xi + rng.uniform(-0.16, 0.16, len(values)), values, s=16,
                       color="black", zorder=4)
        ax.set_xticks(x, [label for *_rest, label in bars], fontsize=10)
        ax.set_xlabel(f"QAOA depth p = {depth}")
        ax.set_ylim(0, 1)
        style_axes(ax, ygrid=False)
        for xi, (method, reps, _label) in zip(x, bars):
            if reps == 0:
                ax.axvspan(xi - 0.42, xi + 0.42, color="#F2F2F2", zorder=0)
    axes[0].set_ylabel("Probability")
    handles = [Patch(color=colour, label=label)
               for label, colour in OUTCOME_STYLE.values()]
    handles.append(Line2D([], [], linestyle="", marker="o", color="black",
                          markersize=5, label="P(optimal), one graph"))
    framed_legend(fig, handles, ncol=5, y=0.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


# ── 4. QAOA at mid size ─────────────────────────────────────────────────────
def plot_mid_qaoa(qaoa_mid: pd.DataFrame) -> plt.Figure:
    """
    MPS-simulated QAOA on the 23 mid-size cases: the share of shots within
    budget and of optimal shots, against qubits, by depth and angle source.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    markers = {"donor": "o", "train": "s"}
    for ax, column, label in ((axes[0], "p_feasible", "shots cutting ≤ B arcs"),
                              (axes[1], "shot_p_optimal", "shots that are optimal plans")):
        for (angles, reps), sub in qaoa_mid.groupby(["angles", "reps"]):
            ax.scatter(sub["qubits"], sub[column], s=48, marker=markers[angles],
                       facecolor=QAOA_DEPTH_COLORS[reps] if angles == "train" else "white",
                       edgecolor=QAOA_DEPTH_COLORS[reps], linewidth=1.6, zorder=3)
        ax.set_xlabel("qubits")
        ax.set_ylabel(f"fraction of {label}")
        ax.set_ylim(-0.02, None)
        style_axes(ax)
    handles = [Line2D([], [], linestyle="", marker="s", color=QAOA_DEPTH_COLORS[p],
                      markersize=8, label=f"p = {p}") for p in sorted(QAOA_DEPTH_COLORS)]
    handles += [Line2D([], [], linestyle="", marker="o", markerfacecolor="white",
                       markeredgecolor=C_REFERENCE, markersize=8,
                       label="donor angles (transferred)"),
                Line2D([], [], linestyle="", marker="s", color=C_REFERENCE,
                       markersize=8, label="trained on the instance")]
    framed_legend(fig, handles, ncol=5, y=0.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


# ── 5. decomposition ────────────────────────────────────────────────────────
DECOMPOSITION_RULES = {"mincut": "min-cut\nsplit", "local": "local\nsplit",
                       "marginal": "marginal\nsplit (DP)",
                       "marginal+refine": "marginal\n+ refine"}


def plot_decomposition(decomposition: pd.DataFrame,
                       large: pd.DataFrame) -> plt.Figure:
    """
    Gap of the decomposed solution per budget-split rule (iterated mode),
    for four pieces and for pieces of at most 155 qubits, beside greedy on
    the whole graph.
    """
    iterated = decomposition[(decomposition["mode"] == "iterated")
                             & decomposition["flow"].notna()]
    greedy_gap = large["Greedy"] / large["Opt_NIP"] - 1
    settings = [("pieces4", "4 pieces"),
                ("fit156", "pieces of ≤ 155 qubits")]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), sharey=True,
                             gridspec_kw={"width_ratios": [5, 4]})
    rng = np.random.default_rng(0)
    for ax, (setting, title) in zip(axes, settings):
        sub = iterated[iterated["setting"] == setting]
        rules = [r for r in DECOMPOSITION_RULES if r in set(sub["rule"])]
        columns = [(DECOMPOSITION_RULES[r], sub[sub["rule"] == r]) for r in rules]
        columns.append(("greedy,\nwhole graph", large.assign(gap=greedy_gap)))
        for x, (_label, data) in enumerate(columns):
            for family in FAMILY_ORDER:
                values = 100 * data.loc[data["family"] == family, "gap"]
                ax.scatter(x + rng.uniform(-0.18, 0.18, len(values)), values,
                           s=30, marker=FAMILY_MARKERS[family],
                           color=C_GREEDY if x == len(columns) - 1 else "#7A2E57",
                           alpha=0.7, edgecolor="none")
            ax.plot([x - 0.3, x + 0.3], [100 * data["gap"].median()] * 2,
                    color="black", linewidth=2.6)
        ax.set_xticks(range(len(columns)), [label for label, _ in columns])
        ax.set_xlabel(f"budget-split rule: {title}")
        ax.set_xlim(-0.6, len(columns) - 0.4)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_yticks([0, 10, 100, 1000], ["0", "10", "100", "1000"])
        style_axes(ax)
    axes[0].set_ylabel("gap to optimum, %")
    handles = [Line2D([], [], linestyle="", marker=FAMILY_MARKERS[f],
                      color=C_REFERENCE, markersize=8, label=f) for f in FAMILY_ORDER]
    handles.append(Line2D([], [], color="black", linewidth=2.6, label="median"))
    framed_legend(fig, handles, ncol=4, y=0.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig
