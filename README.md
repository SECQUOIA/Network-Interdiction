# Network-Interdiction
Algorithms for solving maximum flow network interdiction problem with qubo-based solvers.

## Setup

```
pip install -r requirements.txt
```

You'll also need:

- **A Gurobi license** for `solve_nip_gurobi`, `solve_nip_gurobi_relaxation`, and
  `gurobi_verify_parameters` (in [network_interdiction.py](network_interdiction.py)).
  `pip install gurobipy` alone ships a size-limited trial license good for
  tiny models; anything past a few hundred variables/constraints needs a
  real license (academic licenses are free — see
  [Gurobi's site](https://www.gurobi.com/academia/academic-program-and-licenses/)).
- **A D-Wave Leap API token** for the `qa` (real QPU) solver path in
  [solve_qubo_via_dwave.py](solve_qubo_via_dwave.py) — run `dwave setup` from
  the `dwave-ocean-sdk`, or set the `DWAVE_API_TOKEN` env var. The `sa`
  (simulated annealing, `dwave-neal`) path needs no credentials and is what
  `main()` uses by default.
- **Julia is not required.** `solve_nip_qubo` and
  `solve_nip_qubo_repair_once_end` call out to a `nip_cubic.jl` script that
  isn't included in this repo, so those two functions are currently
  unsupported (see [issue #5](../../issues/5)) — they raise
  `NotImplementedError` if called. Use `nip_to_qubo()` +
  `qubo_verify_parameters()` instead (the path `main()` actually runs), which
  builds the QUBO directly via PyQUBO with no external process.

## Running

```
python network_interdiction.py
```

`main()` picks a test instance from [testcases/](testcases/), solves it
exactly via Gurobi, then via the QUBO formulation + simulated annealing, and
prints a comparison.

## Benchmark results and plots

[network_interdiction_plot.py](network_interdiction_plot.py) draws the two
comparison figures (flow reduction and QUBO energy, each relative to the exact
optimum) from a results CSV. That CSV is produced by
[run_benchmark.py](run_benchmark.py), which runs the pipeline over a list of
`instance:budget:lambda` cases and writes one row per case.

A small sample is checked in, so the plotting script runs with no arguments and
no license or token:

```
python network_interdiction_plot.py
```

It reads [results/sample_results_sa.csv](results/sample_results_sa.csv) and
writes `sample_results_sa_flow_reduction.pdf` and
`sample_results_sa_qubo_energy.pdf` into the current directory (`--csv` and
`--outdir` change both). The sample covers the deliberately tiny
`testcases/tiny_*.flow` instances and only the simulated-annealing pass; its
`QA_*` columns are empty, and a solver whose columns are empty is skipped
rather than drawn as empty bars.

[results/slide_instances_sa.csv](results/slide_instances_sa.csv) is the
same comparison on the nine instances behind the D-Wave slides (the original
D-Wave results were not stored), with Gurobi's optimum, simulated annealing,
and both baselines; the D-Wave column is empty. Its header records the exact
command that produced it. On those instances annealing followed by greedy
repair never beats plain greedy, and plain greedy reproduces the repaired
values reported on the slides, which were produced by a version of the repair
that ignored the annealer's sample (#12).
[notebooks/analysis.ipynb](notebooks/analysis.ipynb)
walks through that comparison family by family, including the check against the
slide values and the penalty windows; the plotting helpers it calls live in
[analysis_utils.py](analysis_utils.py) and its figures are saved to
[figures/](figures/). It reads only the committed CSV, so it needs pandas and
matplotlib but no Gurobi license or D-Wave token (one optional cell uses Gurobi).

### Regenerating the results CSV

The sample is reproducible on any machine, with no Gurobi license and no
D-Wave token:

```
python run_benchmark.py --exact brute --solvers sa \
    --case testcases/tiny_a.flow:1:auto \
    --case testcases/tiny_b.flow:1:auto \
    --case testcases/tiny_b.flow:2:auto \
    --case testcases/tiny_c.flow:1:auto \
    --case testcases/tiny_d.flow:1:auto \
    --case testcases/tiny_d.flow:2:auto \
    --out results/sample_results_sa.csv
```

`--exact brute` gets the `Opt_NIP`/`Opt_QUBO` columns by enumerating every edge
subset within the budget and by brute-forcing the QUBO's ground state with
`dimod.ExactSolver`. That costs `2**(3E + V - 2)` and is capped at 24 QUBO
variables, which is why the sample instances are as small as they are.

Every run also fills two classical baseline columns by default (issue #15):
`Greedy`, the surviving flow after cutting the single highest-impact arc `B`
times starting from nothing, and `Random_Greedy`, the best over 20 random
interdictions (each arc cut with probability `B/E`) after the same greedy
repair the annealer samples get. They hold the repair fixed and remove the
annealer's information, so an SA or QA result that does not beat them on an
instance contributed nothing measurable there. Both are free to compute;
`--baselines none` turns them off and `--seed` fixes the random one. On the
checked-in sample both baselines already reach the optimum on every row, so
the sample shows the format rather than separating SA from greedy.

A real run needs the licensed tools:

```
python run_benchmark.py --exact gurobi --solvers sa,qa \
    --case testcases/50.planarflow:10:auto \
    --out result_dwave_planar.csv
```

`--exact gurobi` takes the exact columns from `solve_nip_gurobi()` and
`gurobi_verify_parameters()` (Gurobi license required), and `--solvers sa,qa`
adds the QPU pass (D-Wave Leap token required). Both are described under
[Setup](#setup) above.

The third field of each `--case` is the cardinality penalty coefficient, passed
to `nip_to_qubo()` as `(lambda, lambda+1)`. It is not a free parameter: only a
window of coefficients makes the QUBO ground state the optimal interdiction,
and outside it every ratio in the row would be measured against the wrong
target. Write `auto` to have `run_benchmark.py` compute that window exactly
from the optimal flow at neighbouring budgets and use its midpoint (issue #2;
the derivation is in the `run_benchmark.py` docstring). The chosen value and
the window are recorded in the `lambda`, `lambda_lo` and `lambda_hi` columns.
The window is often only about one unit wide on the benchmark instances, and it
is empty when the optimal flow is not convex in the budget: then no lambda
works, and the row keeps the exact optimum and the baselines but no QUBO or
solver columns. With a hand-picked number instead, `run_benchmark.py` still
checks that the exact QUBO optimum decodes to an optimal interdiction and
refuses to write the row otherwise.
