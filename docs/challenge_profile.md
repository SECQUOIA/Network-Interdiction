# A bounded difficulty screen for interdiction

`run_challenge_profile.py` profiles the same 16 graph/budget pairs with the
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
