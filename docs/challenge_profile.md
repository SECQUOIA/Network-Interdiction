# A bounded difficulty screen for interdiction

`run_challenge_profile.py` profiles matched graph/budget pairs with the
monolithic Gurobi formulation and max-flow-cut Benders. The [input description](../testcases/challenge/README.md)
provides provenance, licenses, units, allowed arcs, terminal selection and the
synthetic construction. No hardware is contacted.

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
# A usable Gurobi license is required.
ulimit -v 8388608
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout 900 .venv/bin/python run_challenge_profile.py \
  --time-limit 20 --out results/challenge_profile-new.json
```

The output must not exist. Each invocation records its own attempt, pins five
executable source hashes plus the input manifest and flow-file hashes, and saves
after every solver. An interrupted process leaves partial evidence. `--names`
selects a separate pilot; it does not resume an output. The retained artifact uses
one repetition, 20 seconds per solver, eight Gurobi threads, seed zero, and zero
requested relative/absolute MIP gaps. This driver permits at most 24 cases, 1000
vertices, 4000 arcs, 60 seconds per solver, and 1000 Benders masters per case.

The monolithic path records branch-and-bound nodes, best-bound/solver-objective
trajectory at most four times per second plus a final point, and every incumbent
interdiction witness. It independently recomputes those witnesses' surviving flows
after optimization: the MILP cut objective can exceed the true flow for the same
interdictions. Audit time is separate, outside the allowance; add it for end-to-end
comparisons. Benders reports cumulative master nodes, cuts, iterations, bounds and
feasible flows. Its max-flow subproblems are inside its wall allowance. Construction
and NetworkX calls are cooperative and can overrun deadlines; the external process
limit is the final safeguard. Actual elapsed times are retained, not capped labels.

Bounds and statuses retain Gurobi's numerical qualifications. A time limit never
becomes an optimum. An empty cut is a feasible fallback. QUBO dimensions count the
**actual compiled linear encoding**, with coefficients 1/2 used only as a size
probe, not valid solution penalties. The optional squared-budget formulation adds
couplings and must be measured separately. Source/sink degree, undirected planarity,
budget and allowed-arc count accompany each row.

## Predetermined classification

The threshold is **five seconds for each solver**, with a 20-second per-solver cap.
This is a short local screen, not intrinsic complexity or a universal size limit.

1. Any source/sink-incident arc in either solver's proven-optimal witness excludes
   the case. Both `any` and `all` terminal-incidence flags describe the saved cut,
   not every alternative optimum. Empty cuts have both flags false.
2. An interior optimum must be established by a completed optimal witness, or
   structurally because **every allowed arc is interior**. Otherwise optimal
   terminal location remains unknown and the case is not called hard.
3. Only when both solver runtimes reach five seconds and step 2 holds is the case
   `hard_for_both_measured_solvers`. An incomplete solve can qualify only with the
   structural location certificate; its optimum remains unknown.
4. Keep every other case and its exclusion reason.

Restricted covering cases prevent terminal interdiction by construction; every
comparator must match their allowed sets. Real networks can be easy, legacy random
families can have terminal optima, and planarity never establishes intrinsic
hardness. The cohort retains these controls. The exploratory two-second pilot
used unrestricted maximum-distance terminals (power bus 8 was degree one), so the
retained campaign uses the documented minimum-degree selection rule. Pilot results
are excluded; the five-second hardness criterion was unchanged.

Matched greedy, LP, SA, decomposition and simulator scaling use these pinned inputs
in follow-up work. This screen supplies bounded references and explicit unknowns
instead of forcing an exact solve before each experiment.

## Retained screen (one repetition)

The [raw JSON](../results/challenge_profile.json) retains all 32 solver calls,
including incomplete Benders results. Source revision
`10f50c6dc5933d381ca682a3cac4fd968cb25107`; Intel Xeon w5-2565X,
Python 3.12.12, Gurobi 13.0.3, NetworkX 3.7, PyQUBO 1.5.0. One successful
campaign attempt; the earlier exploratory pilot is not included. Source hashes
match that revision. The following values come directly from the retained rows.

| Network | B | Linear QUBO variables | Optimum | MILP seconds / nodes | Benders flow [bound] | Benders seconds | Screen |
|---|---:|---:|---:|---:|---:|---:|---|
| ema | 1 | 846 | 6102 | 0.032 / 1 | 6102 [6102] | 0.008 | terminal optimum witness |
| ema | 3 | 846 | 0 | 0.023 / 1 | 0 [0] | 0.082 | not both slow |
| school4a | 1 | 1273 | 3200 | 0.048 / 1 | 3200 [3200] | 0.009 | terminal optimum witness |
| school4a | 3 | 1273 | 1640 | 0.030 / 1 | 1640 [1640] | 1.811 | terminal optimum witness |
| ieee14 | 1 | 132 | 170 | 0.007 / 1 | 170 [170] | 0.003 | not both slow |
| ieee14 | 2 | 132 | 53 | 0.006 / 1 | 53 [53] | 0.018 | not both slow |
| cover_n12_s0 | 20 | 142 | 4 | 0.021 / 1 | 4 [4] | 3.294 | not both slow |
| cover_n18_s0 | 55 | 343 | 6 | 0.088 / 387 | 9 [4] | 20.003 | not both slow |
| cover_n24_s0 | 113 | 664 | 6 | 0.461 / 3483 | 12 [3] | 20.012 | not both slow |
| cover_n30_s0 | 192 | 1095 | 8 | 12.157 / 39271 | 14 [3] | 20.004 | both slow; interior |
| 50.planarflow | 3 | 912 | 428 | 0.019 / 1 | 428 [428] | 0.176 | terminal optimum witness |
| 50.planarflow | 6 | 912 | 205 | 0.032 / 1 | 205 [205] | 0.062 | terminal optimum witness |
| 50_0.15.erdosflow | 3 | 1122 | 503 | 0.042 / 1 | 503 [503] | 0.022 | terminal optimum witness |
| 50_0.15.erdosflow | 6 | 1122 | 294 | 0.032 / 1 | 294 [294] | 0.092 | terminal optimum witness |
| 50_5.2dnnflow | 3 | 1068 | 220 | 0.026 / 1 | 220 [220] | 0.025 | terminal optimum witness |
| 50_5.2dnnflow | 6 | 1068 | 75 | 0.031 / 1 | 75 [75] | 0.060 | terminal optimum witness |

**The retained hard set contains `cover_n30_s0`, B=192, only.** Its monolithic
optimum is 8, proven after 39,271 branch-and-bound nodes and 12.16 seconds.
Benders stops at 20.00 seconds with flow 14 and lower bound 3. Every allowed
interdiction is internal. Its 1,095-variable linear encoding exceeds the
24-qubit statevector limit used by this project; a matched simulation must report
that refusal or use a separately bounded approximation/decomposition.

Smaller covering cases expose Benders weakness without qualifying as hard for
both methods. All six legacy cases are excluded by terminal-incident optimal
witnesses. The application networks are useful domain controls but were easy in
this screen. These observations do not establish quantum advantage, a general
size threshold, or representative runtime distributions. The largest completed
monolithic case here has 1,273 QUBO variables (school4a); size alone is a poor
proxy for difficulty. All original-optimum claims use the MILP certificate plus
an independently recomputed integer max flow, subject to solver tolerances.

`tests/test_challenge_results.py` checks every saved solver and incumbent cut,
allowed-arc membership, flow, cross-method bound, terminal flag and covering
endpoint-union identity without repeating the campaign. The contact-derived
rows retain the source's CC BY-NC-SA 3.0 terms; see the input attribution document.


A second bounded run covers the remaining eight large `.flow` inputs at B=3/6,
using the same 20-second limits and unchanged five-second classification:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout 900 .venv/bin/python run_challenge_profile.py \
  --manifest testcases/challenge/legacy_manifest.json \
  --time-limit 20 --out results/challenge_legacy_profile-new.json
```

Together the two manifests cover all 11 existing large `.flow` files (50–150
vertices), at two budgets each. Raw `.u` files are generator inputs, not additional
converted flow cases. Tiny and `mid/` cases already support the exact/unit and
simulator comparisons; they are not proposed as Gurobi-hard candidates here.

The [second raw JSON](../results/challenge_legacy_profile.json) records 32 completed
solver calls from source `b05c943589900c133d7e3383026c6233b31ebe40`, one successful attempt.

| Network | B | Linear QUBO variables | Optimum | MILP seconds / nodes | Benders seconds | Screen |
|---|---:|---:|---:|---:|---:|---|
| 75.planarflow | 3 | 1387 | 399 | 0.292 / 1 | 0.113 | terminal optimum witness |
| 75.planarflow | 6 | 1387 | 237 | 0.042 / 1 | 0.100 | terminal optimum witness |
| 125.planarflow | 3 | 2337 | 547 | 0.053 / 1 | 0.203 | terminal optimum witness |
| 125.planarflow | 6 | 2337 | 328 | 0.052 / 1 | 0.417 | terminal optimum witness |
| 150.planarflow | 3 | 2812 | 909 | 0.080 / 1 | 0.084 | terminal optimum witness |
| 150.planarflow | 6 | 2812 | 691 | 0.071 / 1 | 0.514 | terminal optimum witness |
| 70_0.1.erdosflow | 3 | 1436 | 404 | 0.034 / 1 | 0.028 | terminal optimum witness |
| 70_0.1.erdosflow | 6 | 1436 | 188 | 0.043 / 1 | 0.111 | terminal optimum witness |
| 90_0.1.erdosflow | 3 | 2368 | 555 | 0.047 / 1 | 0.036 | terminal optimum witness |
| 90_0.1.erdosflow | 6 | 2368 | 341 | 0.061 / 1 | 0.127 | terminal optimum witness |
| 75_6.2dnnflow | 3 | 1843 | 312 | 0.043 / 1 | 0.026 | not both slow |
| 75_6.2dnnflow | 6 | 1843 | 145 | 0.047 / 1 | 1.537 | terminal optimum witness |
| 100_6.2dnnflow | 3 | 2396 | 342 | 0.062 / 1 | 0.084 | terminal optimum witness |
| 100_6.2dnnflow | 6 | 2396 | 126 | 0.058 / 1 | 0.045 | terminal optimum witness |
| 100_7.2dnnflow | 3 | 2783 | 479 | 0.085 / 1 | 0.288 | terminal optimum witness |
| 100_7.2dnnflow | 6 | 2783 | 257 | 0.085 / 1 | 0.458 | terminal optimum witness |

None of these additional legacy cases qualifies as hard for both methods.
Across both runs, 24 of 32 cases are excluded by terminal-incident optimal witnesses.
The largest legacy encoding measured is 2812 variables; all legacy solves completed within their limits.


The literature and standard-source survey used covering reductions and the
repository's existing planar/ER/k-NN generators, the Transportation Networks
repository, and the standard IEEE test system. Generic DIMACS max-flow instances
alone do not certify interdiction difficulty; no new DIMACS generator is claimed
here. The retained cases are an initial bounded hard-set screen, not a survey of
all instances or budgets in those external collections.
