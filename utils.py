"""
Loading and plotting helpers for notebooks/analysis.ipynb.

The notebook analyses the benchmark CSVs written by run_benchmark.py --
principally results/slide_instances_sa.csv, the classical rerun of the
D-Wave slide instances (issue #18). Everything that computes or draws lives
here so the notebook stays a sequence of short, explained calls.

Every method is scored the way the slides score it: the flow reduction it
achieves as a fraction of the optimal flow reduction,

    ratio = (maxflow - flow left by the method) / (maxflow - Opt_NIP),

so 1.0 is optimal. Derived columns are named ``ratio_<method>`` and never
reuse a CSV column name: an earlier draft of the notebook overwrote the raw
``Greedy`` flows with their ratios and silently corrupted every count that
compared flows.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── palette ─────────────────────────────────────────────────────────────────
# Okabe-Ito colours (distinguishable under the common colour-vision
# deficiencies), named by role so a palette change touches only this block.
C_GUROBI = "#1A1A1A"        # Gurobi's incumbent trajectory
C_GREEDY = "#0072B2"        # greedy from scratch
C_SA = "#E69F00"            # SA + greedy repair
C_SA_RAW = "#F5C462"        # raw SA sample (lighter shade of C_SA)
C_RANDOM = "#009E73"        # random start + repair
C_ALERT = "#D55E00"         # infeasible rows, empty windows, the budget
C_ACCENT = "#56B4E9"        # windows, bars and bands that are not a method
C_ACCENT_DARK = "#2B6F99"
C_ENERGY = "#CC79A7"        # the energy-gap figure
C_ENERGY_DARK = "#8E4A73"
C_REFERENCE = "#6E6E6E"     # the optimum, diagonals, envelopes
C_MEAN = "#1A1A1A"          # mean markers and chosen-λ dots
GRID_GREY = "#CCCCCC"

FAMILY_ORDER = ["planar", "Erdős–Rényi", "k-NN"]
_FAMILY_FROM_EXTENSION = {
    "planarflow": "planar",
    "erdosflow": "Erdős–Rényi",
    "2dnnflow": "k-NN",
}
FAMILY_MARKERS = {"planar": "o", "Erdős–Rényi": "s", "k-NN": "^"}

# method key -> (CSV column with the surviving flow, label, colour, marker, linestyle)
METHODS: Dict[str, tuple] = {
    "sa_raw": ("SA_NIP", "SA (raw sample)", C_SA_RAW, "o", "--"),
    "sa_repair": ("SA_NIP_Greedy", "SA + greedy repair", C_SA, "o", "-"),
    "greedy": ("Greedy", "Greedy from scratch", C_GREEDY, "s", "-"),
    "random": ("Random_Greedy", "Random start + repair", C_RANDOM, "^", ":"),
}


def apply_plot_style() -> None:
    """Set consistent plotting defaults for notebook figures."""
    plt.rcParams.update(
        {
            "figure.figsize": (9.2, 5.8),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "font.size": 12,
        }
    )


def style_axes(ax: plt.Axes, ygrid: bool = True) -> None:
    """Boxed axes with inward ticks on all four sides and a faint y-grid."""
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_alpha(0.85)
        ax.spines[side].set_linewidth(1.1)
    ax.tick_params(axis="both", which="major", direction="in", top=True,
                   right=True, length=7, width=1.1, labelsize=12)
    ax.tick_params(axis="both", which="minor", direction="in", top=True,
                   right=True, length=3.5, width=0.9)
    ax.grid(axis="x", visible=False)
    if ygrid:
        ax.grid(axis="y", which="major", linestyle="--", linewidth=0.8,
                alpha=0.45)
    else:
        ax.grid(axis="y", visible=False)


def framed_legend(fig: plt.Figure, handles, ncol: int, y: float = -0.02):
    """The framed, multi-column legend below the panels."""
    return fig.legend(handles=handles, loc="upper center",
                      bbox_to_anchor=(0.5, y), ncol=ncol, frameon=True,
                      framealpha=0.88, edgecolor=GRID_GREY, fontsize=11,
                      handlelength=2.4, handletextpad=0.6)


# ── data ────────────────────────────────────────────────────────────────────
def read_provenance(csv_path: Path) -> list[str]:
    """The ``#`` header lines run_benchmark.py writes, minus the long command."""
    lines = []
    for line in Path(csv_path).read_text().splitlines():
        if not line.startswith("#"):
            break
        if not line.startswith("# command"):
            lines.append(line[2:])
    return lines


def load_results(csv_path: Path) -> pd.DataFrame:
    """
    Read a run_benchmark.py CSV and add the derived columns the plots use:
    ``family``, ``qubo_vars`` (V + 3E, the slides' x-axis), ``ratio_<method>``
    for every method in METHODS, and ``lambda_width``. Rows keep the CSV
    order, which run_benchmark.py writes per instance by increasing budget.
    """
    df = pd.read_csv(csv_path, comment="#")
    df["family"] = df["instance"].str.rsplit(".", n=1).str[1].map(
        _FAMILY_FROM_EXTENSION)
    df["qubo_vars"] = df["V"] + 3 * df["E"]
    optimal_reduction = df["maxflow"] - df["Opt_NIP"]
    for key, (column, *_rest) in METHODS.items():
        df[f"ratio_{key}"] = (df["maxflow"] - df[column]) / optimal_reduction
    df["lambda_width"] = df["lambda_hi"] - df["lambda_lo"]
    return df


def head_to_head_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    SA + repair against greedy from scratch, per family and overall, on the
    rows that have a QUBO (an empty lambda window leaves none). "Better"
    means leaving less flow. One plain dict per row keeps the counts integer.
    """
    rows = []
    for name, sub in [(f, df[df.family == f]) for f in FAMILY_ORDER] + [("all", df)]:
        sub = sub.dropna(subset=["SA_NIP_Greedy"])
        rows.append({
            "rows": len(sub),
            "SA+repair better": int((sub["SA_NIP_Greedy"] < sub["Greedy"]).sum()),
            "tie": int((sub["SA_NIP_Greedy"] == sub["Greedy"]).sum()),
            "Greedy better": int((sub["SA_NIP_Greedy"] > sub["Greedy"]).sum()),
            "Greedy optimal": int((sub["Greedy"] == sub["Opt_NIP"]).sum()),
            "SA+repair optimal": int((sub["SA_NIP_Greedy"] == sub["Opt_NIP"]).sum()),
            "raw SA feasible": int(sub["SA_NIP"].notna().sum()),
            "mean ratio SA+repair": sub["ratio_sa_repair"].mean(),
            "mean ratio Greedy": sub["ratio_greedy"].mean(),
        })
    return pd.DataFrame(rows, index=FAMILY_ORDER + ["all"])


def slide_values() -> pd.DataFrame:
    """
    Values read off slides 8-10 of S-T_NIP.pptx by eye (about ±0.01). The
    original result files were never stored, so this is pinned evidence, not
    data. NaN is a bar the slide drew at zero (no feasible sample). The
    slides' QA-Greedy and SA-Greedy bars were identical on every row, so
    ``slide_repaired`` covers both.
    """
    instances = (["testcases/50.planarflow"] * 4 + ["testcases/75.planarflow"] * 3
                 + ["testcases/125.planarflow"] * 2
                 + ["testcases/50_0.15.erdosflow"] * 3 + ["testcases/70_0.1.erdosflow"] * 3
                 + ["testcases/90_0.1.erdosflow"] * 3
                 + ["testcases/50_5.2dnnflow"] * 3 + ["testcases/75_6.2dnnflow"] * 3
                 + ["testcases/100_6.2dnnflow"] * 3)
    nan = np.nan
    return pd.DataFrame({
        "instance": instances,
        "B": [5, 6, 8, 10, 5, 8, 10, 6, 10,
              5, 6, 8, 7, 8, 10, 8, 10, 12,
              4, 5, 6, 6, 7, 8, 6, 7, 8],
        "slide_QA": [0.91, 0.91, 0.87, 0.92, 0.60, 0.72, 0.73, nan, nan,
                     0.89, 0.90, 0.95, 0.97, 0.83, 0.78, 0.86, 0.80, 0.70,
                     0.37, nan, 0.47, nan, nan, 0.86, 0.52, 0.90, 0.84],
        "slide_SA": [0.91, 0.91, 0.93, 0.85, nan, nan, 0.82, 0.16, nan,
                     0.91, 0.93, 0.93, 1.00, 1.00, 0.89, 0.90, 0.95, 0.86,
                     nan, nan, nan, 0.81, 0.79, 0.86, 0.66, 0.90, 0.90],
        "slide_repaired": [1.0] * 9
                          + [1.0, 1.0, 1.0, 1.0, 1.0, 0.99, 1.0, 1.0, 0.97]
                          + [1.0, 1.0, 1.0, 0.95, 0.94, 0.93, 0.90, 0.90, 0.94],
    })


def compare_with_slides(df: pd.DataFrame) -> pd.DataFrame:
    """Join the rerun with the pinned slide values; adds the absolute gaps."""
    cmp = df.merge(slide_values(), on=["instance", "B"], validate="one_to_one")
    cmp["gap_vs_greedy"] = (cmp["slide_repaired"] - cmp["ratio_greedy"]).abs()
    cmp["gap_vs_sa_repair"] = (cmp["slide_repaired"] - cmp["ratio_sa_repair"]).abs()
    cmp["gap_raw_sa"] = (cmp["slide_SA"] - cmp["ratio_sa_raw"]).abs()
    return cmp


# ── plots ───────────────────────────────────────────────────────────────────
def _instance_layout(sub: pd.DataFrame, gap: float = 0.8):
    """x positions grouping each instance's budgets, with a gap between instances."""
    positions, groups, x = [], [], 0.0
    for _, inst in sub.groupby("instance", sort=False):
        xs = [x + i for i in range(len(inst))]
        positions.extend(xs)
        groups.append((np.mean(xs), int(inst["qubo_vars"].iloc[0]), inst.index))
        x = xs[-1] + 1 + gap
    return np.array(positions), groups


def plot_ratio_by_family(df: pd.DataFrame) -> plt.Figure:
    """
    Flow-reduction ratio of every method, one panel per graph family. Lines
    join the budgets of one instance; instances are separated by a gap and
    labelled by QUBO size. Vermillion crosses mark rows where raw SA had no
    budget-feasible sample; "no valid λ" marks the empty-window row.
    """
    # Small sideways offset per method so tied results (e.g. SA + repair and
    # greedy both at 1.0) sit side by side instead of hiding each other.
    dodge = dict(zip(METHODS, [-0.15, -0.05, 0.05, 0.15]))
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), sharey=True)
    for ax, family in zip(axes, FAMILY_ORDER):
        sub = df[df.family == family]
        xs, groups = _instance_layout(sub)
        sub = sub.assign(_x=xs)
        for _, inst_idx in [(g[1], g[2]) for g in groups]:
            inst = sub.loc[inst_idx]
            for key, (_col, label, color, marker, ls) in METHODS.items():
                ax.plot(inst["_x"] + dodge[key], inst[f"ratio_{key}"], linestyle=ls,
                        marker=marker, color=color, linewidth=2.2,
                        markersize=7, markeredgecolor=color, label=label)
        infeasible = sub[sub["SA_NIP"].isna() & sub["SA_NIP_Greedy"].notna()]
        ax.scatter(infeasible["_x"] + dodge["sa_raw"], np.full(len(infeasible), 0.03), marker="x",
                   s=60, color=C_ALERT, linewidths=2, zorder=5)
        for _, row in sub[sub["lambda"].isna()].iterrows():
            ax.text(row["_x"], 0.10, "no valid λ", color=C_ALERT, ha="center",
                    fontsize=10, rotation=90, va="bottom")
        ax.axhline(1.0, color=C_REFERENCE, linestyle="--", linewidth=1.3, alpha=0.7)
        ax.set_xticks(xs, [f"{b}" for b in sub["B"]])
        for center, qubo_vars, _ in groups:
            ax.text(center, -0.105, f"{qubo_vars} vars", ha="center", va="top",
                    fontsize=10.5, color="#666666",
                    transform=ax.get_xaxis_transform())
        ax.set_title(family)
        ax.set_xlabel("budget B, grouped by instance", labelpad=30)
        style_axes(ax)
    axes[0].set_ylabel("flow reduction / optimal")
    axes[0].set_ylim(0.0, 1.06)
    handles = [Line2D([], [], color=c, marker=m, linestyle=ls, linewidth=2.2,
                      markersize=7, label=label)
               for _col, label, c, m, ls in METHODS.values()]
    handles += [Line2D([], [], color=C_ALERT, marker="x", linestyle="",
                       markersize=8, markeredgewidth=2,
                       label="raw SA: no feasible sample"),
                Line2D([], [], color=C_REFERENCE, linestyle="--", alpha=0.7,
                       label="optimal (Gurobi)")]
    fig.tight_layout()
    framed_legend(fig, handles, ncol=3)
    return fig


def plot_performance_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Final solution quality per family and method, in the style of the
    quantum-preconditioning performance-distribution figure. Performance is
    100 × ratio. The top strip holds instances solved exactly (with the
    fraction above each column); the main panel shows the rest on a log
    scale of the gap to 100%, with a black bar at the mean.
    """
    from matplotlib.gridspec import GridSpec

    methods = ["greedy", "sa_repair", "random"]
    offsets = dict(zip(methods, [-0.25, 0.0, 0.25]))
    fig = plt.figure(figsize=(10.5, 7.2))
    grid = GridSpec(2, 1, height_ratios=[1, 3.2], hspace=0.06, figure=fig)
    top = fig.add_subplot(grid[0])
    main = fig.add_subplot(grid[1], sharex=top)
    rng = np.random.default_rng(0)

    for fx, family in enumerate(FAMILY_ORDER):
        sub = df[df.family == family]
        for row_offset, key in zip([0.3, 0.0, -0.3], methods):
            color = METHODS[key][2]
            perf = 100.0 * sub[f"ratio_{key}"].dropna()
            x0 = fx + offsets[key]
            optimal = perf[np.isclose(perf, 100.0)]
            top.scatter(x0 + rng.uniform(-0.07, 0.07, len(optimal)),
                        np.full(len(optimal), row_offset), color=color,
                        s=38, edgecolor="white", linewidth=0.5, zorder=3)
            top.text(x0, 0.82, f"{len(optimal)}/{len(perf)}", ha="center",
                     va="bottom", fontsize=10, color=color,
                     transform=top.get_xaxis_transform())
            rest = perf[~np.isclose(perf, 100.0)]
            main.scatter(x0 + rng.uniform(-0.07, 0.07, len(rest)), 100.0 - rest,
                         color=color, s=42, edgecolor="white", linewidth=0.5,
                         zorder=3)
            mean = perf.mean()
            if mean < 100.0 - 1e-9:
                main.plot([x0 - 0.09, x0 + 0.09], [100.0 - mean] * 2,
                          color=C_MEAN, linewidth=3.2, zorder=4)

    top.set_ylim(-0.6, 0.6)
    top.set_yticks([0.0], ["100%"])
    top.axhline(0.0, color=C_REFERENCE, linestyle="--", linewidth=1.2, alpha=0.8)
    top.text(0.5, 1.22, "Fraction solved optimally", ha="center",
             transform=top.transAxes, fontsize=11, color="#777777")
    top.tick_params(axis="x", labelbottom=False, bottom=False)
    for side in ["top", "right", "bottom"]:
        top.spines[side].set_visible(False)
    top.grid(False)

    main.set_yscale("log")
    main.set_ylim(100.0, 0.5)  # inverted: better (smaller gap) at the top
    gaps = [1, 5, 10, 25, 50, 100]
    main.set_yticks(gaps, [f"{100 - g:g}%" for g in gaps])
    main.minorticks_off()
    main.set_xticks(range(len(FAMILY_ORDER)), FAMILY_ORDER)
    main.set_xlim(-0.6, len(FAMILY_ORDER) - 0.4)
    main.set_ylabel("performance (% of optimal reduction)")
    style_axes(main)
    main.spines["top"].set_visible(False)
    # broken-axis marks between the strip and the main panel
    kwargs = dict(transform=main.transAxes, color="k", clip_on=False, linewidth=1.1)
    main.plot((-0.012, 0.012), (0.985, 1.015), **kwargs)
    main.plot((-0.012, 0.012), (1.005, 1.035), **kwargs)

    handles = [Patch(color=METHODS[k][2], label=METHODS[k][1]) for k in methods]
    handles.append(Line2D([], [], color=C_MEAN, linewidth=3.2, label="Mean"))
    framed_legend(fig, handles, ncol=4, y=0.045)
    fig.subplots_adjust(bottom=0.13)
    return fig


def plot_raw_sa_feasibility(df: pd.DataFrame) -> plt.Figure:
    """
    Raw SA quality against problem size. Each point is a row where at least
    one of SA's 20 lowest-energy samples respected the budget; the vermillion
    number above each size counts rows where none did.
    """
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    with_qubo = df.dropna(subset=["SA_NIP_Greedy"])
    for family in FAMILY_ORDER:
        sub = with_qubo[(with_qubo.family == family) & with_qubo["SA_NIP"].notna()]
        ax.scatter(sub["qubo_vars"], sub["ratio_sa_raw"],
                   marker=FAMILY_MARKERS[family], s=70, color=C_SA_RAW,
                   edgecolor=C_SA, linewidth=1.1, label=family)
    for size, group in with_qubo.groupby("qubo_vars"):
        missing = int(group["SA_NIP"].isna().sum())
        if missing:
            ax.text(size, 1.03, str(missing), ha="center", color=C_ALERT,
                    fontsize=12, fontweight="bold")
    ax.axhline(1.0, color=C_REFERENCE, linestyle="--", linewidth=1.3, alpha=0.7)
    ax.set_ylim(0.0, 1.1)
    ax.set_xlabel("QUBO variables (V + 3E)")
    ax.set_ylabel("raw SA flow reduction / optimal")
    style_axes(ax)
    handles = [Line2D([], [], marker=FAMILY_MARKERS[f], color=C_SA_RAW,
                      markeredgecolor=C_SA, linestyle="", markersize=8,
                      label=f) for f in FAMILY_ORDER]
    handles.append(Patch(color=C_ALERT, label="rows with no feasible sample"))
    fig.tight_layout()
    framed_legend(fig, handles, ncol=4)
    return fig


def plot_slide_check(cmp: pd.DataFrame, precision: float = 0.01) -> plt.Figure:
    """
    The slides' "repaired" values against this rerun's greedy (left) and
    SA + repair (right). The shaded band is the ±precision the slide figures
    can be read to.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    panels = [("ratio_greedy", "Greedy from scratch", "gap_vs_greedy", C_GREEDY),
              ("ratio_sa_repair", "SA + greedy repair", "gap_vs_sa_repair", C_SA)]
    line = np.linspace(0.0, 1.05, 2)
    for ax, (column, label, gap_col, color) in zip(axes, panels):
        ax.fill_between(line, line - precision, line + precision,
                        color=C_ACCENT, alpha=0.2, linewidth=0)
        ax.plot(line, line, color=C_REFERENCE, linestyle="--", linewidth=1.3, alpha=0.7)
        for family in FAMILY_ORDER:
            sub = cmp[cmp.family == family]
            ax.scatter(sub[column], sub["slide_repaired"],
                       marker=FAMILY_MARKERS[family], s=70, color=color,
                       alpha=0.85, edgecolor="white", linewidth=0.6)
        ax.set_xlim(0.0, 1.05)
        ax.set_ylim(0.85, 1.02)
        ax.set_xlabel(f"this rerun: {label}")
        ax.text(0.97, 0.05, f"largest gap {cmp[gap_col].max():.3f}",
                transform=ax.transAxes, ha="right", fontsize=14)
        style_axes(ax)
    axes[0].set_ylabel('slides: "QA/SA-Greedy" value')
    handles = [Line2D([], [], marker=FAMILY_MARKERS[f], color="#666666",
                      linestyle="", markersize=8, label=f) for f in FAMILY_ORDER]
    handles.append(Patch(color=C_ACCENT, alpha=0.35,
                         label=f"±{precision:g} slide reading precision"))
    fig.tight_layout()
    framed_legend(fig, handles, ncol=4)
    return fig


def plot_lambda_windows(df: pd.DataFrame) -> plt.Figure:
    """
    The exact valid λ window per row (bar) and the midpoint used (dot). An
    empty window -- lower end above upper end -- is drawn in vermillion.
    """
    fig, ax = plt.subplots(figsize=(10.5, 9.5))
    positions, labels, y = [], [], 0.0
    for family in FAMILY_ORDER:
        for _, row in df[df.family == family].iterrows():
            empty = row["lambda_lo"] >= row["lambda_hi"]
            color = C_ALERT if empty else C_ACCENT
            ax.plot([row["lambda_lo"], row["lambda_hi"]], [y, y], color=color,
                    linewidth=7, solid_capstyle="butt", alpha=0.9)
            if empty:
                ax.text(row["lambda_lo"] + 1.0, y, "empty: no valid λ",
                        va="center", color=C_ALERT, fontsize=11)
            else:
                ax.plot(row["lambda"], y, "o", color=C_MEAN, markersize=6)
            positions.append(y)
            labels.append(f"{row['qubo_vars']} vars, B={row['B']}")
            y += 1
        first = positions[-int((df.family == family).sum())]
        ax.text(1.01, (first + y - 1) / 2, family, rotation=-90, va="center",
                ha="left", fontsize=13, transform=ax.get_yaxis_transform())
        ax.axhline(y - 0.2, color="#999999", linewidth=0.8, alpha=0.6)
        y += 0.6  # gap between families
    ax.set_yticks(positions, labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("cardinality penalty λ")
    style_axes(ax, ygrid=False)
    ax.grid(axis="x", visible=True, linestyle="--", linewidth=0.8, alpha=0.45)
    handles = [Line2D([], [], color=C_ACCENT, linewidth=7, label="valid λ window"),
               Line2D([], [], color=C_MEAN, marker="o", linestyle="",
                      label="λ used (midpoint)"),
               Line2D([], [], color=C_ALERT, linewidth=7, label="empty window")]
    fig.tight_layout()
    framed_legend(fig, handles, ncol=3)
    return fig


def plot_flow_curve(curve: list, B: int, title: Optional[str] = None,
                    window: int = 4) -> plt.Figure:
    """
    Why a λ window is empty, in two panels. Left: the optimal surviving flow
    f(k) around the budget, with its lower convex envelope -- f(B) above the
    envelope means no per-cut price stops the QUBO at exactly B cuts. Right:
    how much each extra cut saves, f(k-1) - f(k); for a convex f these
    savings never increase, so a later cut saving more than the B-th is the
    non-convexity, made visible.
    """
    ks = np.arange(len(curve))
    f = np.asarray(curve, dtype=float)
    hull = [0]
    for k in range(1, len(f)):  # lower convex hull (monotone chain)
        while len(hull) >= 2:
            k1, k2 = hull[-2], hull[-1]
            if (f[k2] - f[k1]) * (k - k1) >= (f[k] - f[k1]) * (k2 - k1):
                hull.pop()
            else:
                break
        hull.append(k)

    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.6))
    lo, hi = max(0, B - window), min(len(f) - 1, B + window)
    zoom = (ks >= lo) & (ks <= hi)
    hull_k = np.array(hull)
    hull_zoom = hull_k[(hull_k >= lo - 1) & (hull_k <= hi + 1)]
    left.plot(hull_zoom, f[hull_zoom], color=C_REFERENCE, linestyle="--",
              linewidth=2.2, label="lower convex envelope")
    left.plot(ks[zoom], f[zoom], "o-", color=C_ACCENT_DARK, linewidth=2.2,
              markersize=8, label="optimal flow f(k)")
    left.plot([B], [f[B]], "o", color=C_ALERT, markersize=13, zorder=5,
              label=f"budget B={B}")
    left.set_xlim(lo - 0.3, hi + 0.3)
    left.set_ylim(f[hi] - 0.05 * (f[lo] - f[hi]), f[lo] + 0.05 * (f[lo] - f[hi]))
    left.set_xlabel("number of cuts k")
    left.set_ylabel("best surviving max flow f(k)")
    style_axes(left)
    left.legend(frameon=True, framealpha=0.88, edgecolor=GRID_GREY)

    savings = f[:-1] - f[1:]  # savings[k-1] = f(k-1) - f(k)
    cuts = ks[1:]
    colors = [C_ALERT if k == B else C_ACCENT for k in cuts]
    right.bar(cuts, savings, color=colors, edgecolor="white", width=0.75)
    right.axhline(savings[B - 1], color=C_ALERT, linestyle="--", linewidth=1.3,
                  alpha=0.8)
    for k in cuts[cuts > B]:
        if savings[k - 1] > savings[B - 1]:
            right.annotate(f"{savings[k - 1]:g} > {savings[B - 1]:g}",
                           (k, savings[k - 1]), xytext=(0, 6),
                           textcoords="offset points", ha="center",
                           color=C_ALERT, fontsize=11, fontweight="bold")
    right.set_xlabel("k-th cut")
    right.set_ylabel("flow saved by the k-th cut")
    style_axes(right)
    if title:
        fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    return fig


def plot_energy_gap(df: pd.DataFrame) -> plt.Figure:
    """
    Relative gap between SA's lowest QUBO energy and the exact ground state,
    1 - E_SA / E_opt (energies are negative, so 0 means SA reached the ground
    state). Rows where SA was exact are counted in vermillion at the bottom.
    """
    with_qubo = df.dropna(subset=["SA_QUBO"])
    gap = 1.0 - with_qubo["SA_QUBO"] / with_qubo["Opt_QUBO"]
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    floor = 1e-5
    for family in FAMILY_ORDER:
        mask = (with_qubo.family == family) & (gap > 0)
        ax.scatter(with_qubo.loc[mask, "qubo_vars"], gap[mask],
                   marker=FAMILY_MARKERS[family], s=70, color=C_ENERGY,
                   edgecolor=C_ENERGY_DARK, linewidth=1.1)
    exact = with_qubo[gap <= 0]
    for size, group in exact.groupby("qubo_vars"):
        ax.text(size, floor * 1.4, f"{len(group)} exact", ha="center",
                color=C_ALERT, fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(floor, None)
    ax.set_xlabel("QUBO variables (V + 3E)")
    ax.set_ylabel("1 − E(SA) / E(ground state)")
    style_axes(ax)
    handles = [Line2D([], [], marker=FAMILY_MARKERS[f], color=C_ENERGY,
                      markeredgecolor=C_ENERGY_DARK, linestyle="", markersize=8,
                      label=f) for f in FAMILY_ORDER]
    fig.tight_layout()
    framed_legend(fig, handles, ncol=3)
    return fig


# ── time: Gurobi's incumbent trajectory against the other methods ───────────
# method key -> (time column(s) summed for that method's result, label)
METHOD_TIMES = {
    "sa_raw": (["SA_time_sec"], "SA (raw sample)"),
    "sa_repair": (["SA_time_sec", "SA_repair_time_sec"], "SA + greedy repair"),
    "greedy": (["Greedy_time_sec"], "Greedy from scratch"),
    "random": (["Random_Greedy_time_sec"], "Random start + repair"),
}


def gurobi_best_so_far(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    (times, ratio of the best incumbent so far) from a row's
    ``Gurobi_trajectory``, scored like every other method.
    """
    trajectory = json.loads(row["Gurobi_trajectory"])
    times = np.array([t for t, _ in trajectory])
    best_flow = np.minimum.accumulate([flow for _, flow in trajectory])
    ratio = (row["maxflow"] - best_flow) / (row["maxflow"] - row["Opt_NIP"])
    return times, ratio


def gurobi_time_to_optimum(row: pd.Series) -> float:
    """Seconds until Gurobi's first incumbent that attains the optimum."""
    times, ratio = gurobi_best_so_far(row)
    return float(times[np.isclose(ratio, 1.0)][0])


def method_time(row: pd.Series, key: str) -> float:
    """Wall time of one non-Gurobi method on a row (NaN when not run)."""
    columns, _label = METHOD_TIMES[key]
    return float(sum(row[c] for c in columns))


def time_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Median wall time and median ratio per method, over the rows each method
    was run on, with Gurobi's time to its optimal incumbent and to proving
    it optimal.
    """
    gurobi_found = df.apply(gurobi_time_to_optimum, axis=1)
    rows = {
        "Gurobi (optimal incumbent)": {"median time (s)": gurobi_found.median(),
                                       "median ratio": 1.0, "rows": len(df)},
        "Gurobi (optimality proved)": {"median time (s)": df["Gurobi_time_sec"].median(),
                                       "median ratio": 1.0, "rows": len(df)},
    }
    for key, (_columns, label) in METHOD_TIMES.items():
        times = df.apply(lambda r: method_time(r, key), axis=1)
        ratio = df[f"ratio_{key}"]
        ran = times.notna()
        rows[label] = {"median time (s)": times[ran].median(),
                       "median ratio": ratio[ran].median(),
                       "rows": int(ran.sum())}
    return pd.DataFrame(rows).T


def plot_incumbent_trajectory(df: pd.DataFrame, instance: str, B: int) -> plt.Figure:
    """
    One row's race against the clock: Gurobi's best incumbent so far (black
    steps, one dot per incumbent, dotted line where Gurobi proves
    optimality) against where each other method finishes, all scored as
    flow reduction / optimal. Log time axis.
    """
    row = df[df["instance"].str.endswith(instance) & (df["B"] == B)].iloc[0]
    times, ratio = gurobi_best_so_far(row)
    runtime = float(row["Gurobi_time_sec"])

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.step(np.append(times, runtime), np.append(ratio, ratio[-1]), where="post",
            color=C_GUROBI, linewidth=2.4, label="Gurobi: best incumbent so far")
    ax.plot(times, ratio, "o", color=C_GUROBI, markersize=6)
    ax.axvline(runtime, color=C_GUROBI, linestyle=":", linewidth=1.4)
    ax.text(runtime, 0.04, "  optimality\n  proved", color=C_GUROBI, fontsize=11,
            va="bottom", ha="left")
    for key, (_columns, label) in METHOD_TIMES.items():
        t = method_time(row, key)
        if np.isnan(t) or np.isnan(row[f"ratio_{key}"]):
            continue
        _col, _label, color, marker, _ls = METHODS[key]
        if key == "sa_raw":  # hollow ring, so it stays visible under coincident markers
            ax.plot(t, row[f"ratio_{key}"], marker=marker, linestyle="",
                    markerfacecolor="none", markeredgecolor=C_SA, markersize=17,
                    markeredgewidth=2.2, label=label, zorder=6)
            continue
        ax.plot(t, row[f"ratio_{key}"], marker=marker, linestyle="", color=color,
                markersize=12, markeredgecolor="white", markeredgewidth=1.0,
                label=label, zorder=5)
    ax.axhline(1.0, color=C_REFERENCE, linestyle="--", linewidth=1.3, alpha=0.7)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("wall time (s, log scale)")
    ax.set_ylabel("flow reduction / optimal")
    ax.set_title(f"{row['family']}: {row['qubo_vars']} QUBO variables, B={B}")
    style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    framed_legend(fig, handles, ncol=3)
    return fig


def plot_time_vs_quality(df: pd.DataFrame) -> plt.Figure:
    """
    Every row, every method: wall time against flow reduction / optimal.
    Gurobi is plotted at the time of its first optimal incumbent. Marker
    colour is the method, marker shape the graph family.
    """
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for family in FAMILY_ORDER:
        sub = df[df.family == family]
        marker = FAMILY_MARKERS[family]
        ax.scatter(sub.apply(gurobi_time_to_optimum, axis=1), np.ones(len(sub)),
                   marker=marker, s=70, color=C_GUROBI, edgecolor="white",
                   linewidth=0.6, zorder=4)
        for key in METHOD_TIMES:
            color = METHODS[key][2]
            times = sub.apply(lambda r: method_time(r, key), axis=1)
            ax.scatter(times, sub[f"ratio_{key}"], marker=marker, s=60,
                       color=color, edgecolor="white", linewidth=0.6, alpha=0.9)
    ax.axhline(1.0, color=C_REFERENCE, linestyle="--", linewidth=1.3, alpha=0.7)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("wall time (s, log scale)")
    ax.set_ylabel("flow reduction / optimal")
    style_axes(ax)
    handles = [Patch(color=C_GUROBI, label="Gurobi (first optimal incumbent)")]
    handles += [Patch(color=METHODS[k][2], label=label)
                for k, (_c, label) in METHOD_TIMES.items()]
    handles += [Line2D([], [], marker=FAMILY_MARKERS[f], color="#777777",
                       linestyle="", markersize=8, label=f) for f in FAMILY_ORDER]
    fig.tight_layout()
    framed_legend(fig, handles, ncol=4)
    return fig
