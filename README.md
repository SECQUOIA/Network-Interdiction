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
  All production Gurobi models explicitly use eight threads by default,
  including exact references, penalty checks, Lagrangian solves, and pieces.
  The timed trajectory solver also defaults to eight; its
  `--gurobi-threads` option can override that dedicated solve. Thread count
  is recorded in benchmark headers. Earlier saved timing results retain
  their original settings and must not be mixed with an eight-thread rerun.
  NetworkX partitioning, contraction, and max-flow calculations are separate
  from Gurobi and remain serial.
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
- **Qiskit and CUDA-Q, only for the QAOA modules** (see
  [QAOA](#qaoa-on-simulators-qiskit-and-cuda-q) below):
  `pip install -r requirements.txt -r requirements-qaoa.txt`. No GPU or
  account is needed; both run on CPU simulators.

## Running

```
python network_interdiction.py
```

`main()` picks a test instance from [testcases/](testcases/), solves it
exactly via Gurobi, then via the QUBO formulation + simulated annealing, and
prints a comparison.

## Decomposition into smaller QUBOs

[decomposition.py](decomposition.py) splits an instance into pieces whose
QUBOs are smaller than the whole graph's `V + 3E - 2` variables (issue #14).
It partitions the graph into contiguous pieces, gives each arc to the piece
its tail lies in, and splits the budget between the pieces either in
proportion to the global minimum-cut arcs each owns (`mincut`) or to each
piece's local max flow (`local`). Each piece is then solved on its own
network: only its arcs may be cut, so they alone carry `x`/`gamma`/`w`
variables (`nip_to_qubo(..., interdictable=...)`), and the rest of the graph,
which has no budget there, is contracted exactly. To compare the sizes:

```
python decomposition.py testcases/50.planarflow --pieces 4 --budget 12
```

Each piece's subproblem is exact: cutting any set of its arcs leaves the
reduced network with the same max flow as the whole graph with those arcs
cut. The decomposition as a whole is a heuristic, since the budget split is
fixed before any piece is solved. Two limits to know:

- A piece's QUBO needs its own cardinality coefficient: the lambda window
  (issue #2) of the piece, not the whole graph's.
- The contraction solves `2**t` minimum cuts for a zone with `t` boundary
  terminals, so it is skipped above 12 (`--max-terminals`). Pieces of the
  Erdős–Rényi testcases have more than that and stay uncontracted; their
  QUBOs are smaller only by the three variables of each arc they do not own.

`solve_decomposed()` solves the pieces sequentially (each seeing what earlier
pieces cut), independently, or `iterated` (sequentially, then re-solving
every piece against all the others' cuts until nothing improves), with
`brute_force_piece_solver` for small pieces, `gurobi_piece_solver` for the
exact optimum of larger ones, or `qubo_piece_solver(lambda)` for the
annealer.

A third budget rule, `marginal`, solves each piece for 0 to B cuts first and
gives the budget to the pieces whose cuts save the most flow, solving a
separable allocation approximation exactly by dynamic programming.
`refine_split()` then moves one unit of budget at a time between pieces while
that lowers the combined flow. `--fit-qubits 156` picks the fewest pieces
whose QUBOs use at most 156 logical qubits. Device connectivity, embedding
and transpiled circuit depth require separate checks:

```
python decomposition.py testcases/50.planarflow --fit-qubits 156 --budget 6 --split marginal
```

The `multiplier` rule implements #14's alternative: one shared allocation
price, tuned by projected subgradient ascent. It measures the same curves
`f_i(k)` as `marginal`, then each piece chooses a count minimizing
`f_i(k) + lambda*k`. The allocation dual is
`L(lambda) = sum_i min_k(f_i(k) + lambda*k) - lambda*B`, with `lambda >= 0`.
The update uses count violation `sum_i k_i - B`, step
`M / (G*sqrt(iteration))`, and projection onto `[0, M]`; `M` is the largest
marginal flow drop and `G` bounds that count violation. Tied local minima
choose the smaller count. A deterministic repair removes the least costly
budget units or adds the most valuable ones until the available budget is
assigned. The best repaired allocation and best dual bound are retained.

```bash
python decomposition.py testcases/tiny_a.flow --pieces 2 --budget 1 \
    --split multiplier --multiplier-iterations 100
```

`common_multiplier_split()` returns budgets, measured curves, price history,
the repaired surrogate value and its lower bound. Arithmetic for curve
values and prices uses exact fractions of the supplied integer/binary-float
inputs; exported bounds round outward. `optimal_surrogate` means that
allocation gap closed exactly; `iteration_limit` preserves the incumbent
and bound after the allowed updates. The history includes the exact price
as a rational string for audits. Curves must be finite, nonnegative and
nonincreasing. With a heuristic piece solver, bounds describe its measured
curves only; the benchmark uses exact Gurobi piece solves.
The piece oracle requests zero MIP gap and requires `OPTIMAL`, subject to
solver numerical tolerances; an incomplete solve raises an error instead
of supplying an allegedly exact curve.

These are **allocation-surrogate bounds, not global NIP bounds**. Curves
measure each piece in isolation, so their savings can overlap. Every combined
cut is scored on the original graph. Nonconvex curves may also leave a
positive duality gap within the surrogate: for `f=[10,9,0]`, `B=1`, the
integer value is 9 but the best priced lower bound is 5. The method does
not replace its result with the dynamic-programming answer or report that
gap as closed. Changing the common allocation price also does not choose
each piece QUBO's cardinality penalty.

[run_decomposition_benchmark.py](run_decomposition_benchmark.py) measures
every rule and mode against Gurobi's optimum on the slide instances and
budgets, with each piece solved exactly, so the only error left is the
decomposition's own. `--rules` and `--modes` select a bounded subset;
`--max-terminals` caps exact contraction work (0 to 12), and
`--multiplier-iterations` caps price updates. The recorded allocation time
includes measuring every curve, and partition time is separate. The CSV
retains actual interdicted arcs, surrogate status/bounds/curves and the price
history. Each completed variant is flushed before another solve starts.

## QAOA on simulators (Qiskit and CUDA-Q)

The same QUBO the annealer samples can be solved with a penalty-QAOA, written
twice: once in Qiskit ([qaoa_qiskit.py](qaoa_qiskit.py), Aer simulators) and
once in CUDA-Q ([qaoa_cudaq.py](qaoa_cudaq.py)). Issues #3 and #16 start
here. Both build the Ising form of `nip_to_qubo()`'s QUBO through
[qaoa_common.py](qaoa_common.py), use the standard X mixer, train the angles
with COBYLA against the exact expectation value, and score the trained state
the same way:

```
python qaoa_qiskit.py testcases/tiny_a.flow --budget 1 --reps 2
python qaoa_cudaq.py  testcases/tiny_a.flow --budget 1 --reps 2
```

Each prints the expected QUBO energy, the probability of measuring the
ground state and of measuring an optimal interdiction (exact, from the
statevector), and the best surviving flow among the shots before and after
the same greedy repair the annealer passes use. The uniform superposition's
scores and the greedy baseline are printed alongside. `--json` saves the
result. Lambda is `auto` by default: the midpoint of its exact window, as
in `run_benchmark.py`. A budget with an empty window is refused, because no
lambda makes the ground state the answer.

[run_qaoa_benchmark.py](run_qaoa_benchmark.py) runs depths p = 1, 2, 3 over
the six rows of `results/sample_results_sa.csv` and writes
[results/qaoa_tiny.csv](results/qaoa_tiny.csv); its docstring has the exact
command. It also scores the angles trained on the first row, reused unchanged
on every other row, which is the baseline for parameter transfer (#16). The
QAOA section of [notebooks/analysis.ipynb](notebooks/analysis.ipynb) puts
these results next to simulated annealing and greedy. A rerun is close to the
committed file but not bit-identical: Aer's multithreaded simulation can sum
in a different order, so COBYLA can end a few 1e-5 away in the angles. In two
runs, the exact scores agreed to 1e-5 and the shot estimates of P(optimal) to
0.02.

The two backends prepare the same state for the same angles, and the tests
check both against an independent numpy simulation. The angles are ordered
`[beta..., gamma...]`, the order of Qiskit's `qaoa_ansatz()` and of
`qaoa_training_pipeline`. The Hamiltonian is scaled to coefficients of at
most 1, so angles are comparable across instances with different capacities
and lambdas, which parameter transfer (#16) needs.

Limits:

- Statevector simulation of the full `V + 3E - 2`-qubit QUBO caps the exact
  scores at 24 qubits, which means the `tiny_*` instances (14 to 22 qubits).
  Larger problems use matrix product states (next section).
- CUDA-Q's CPU simulator (`qpp-cpu`) is much slower than Aer: 0.3 to 1
  second per evaluation at 14 qubits, against milliseconds for Aer. It is
  faster with `OMP_NUM_THREADS=1` at these sizes. On a GPU, pass
  `--target nvidia`, or `--target tensornet-mps` for more qubits.

## Parameter transfer and regularized warm starts

[qaoa_transfer.py](qaoa_transfer.py) transfers schedules at a fixed depth
through IBM's `TransferTrainer`; [qaoa_gat.py](qaoa_gat.py) adapts the
[MIS GAT training path](https://github.com/jxsoortha/QAOA-Parameter-Transfer-via-GAT)
to NIP. Both return `[beta..., gamma...]` for the normalized Hamiltonian
above, without optimizing a circuit on the target. The IBM dependency is
pinned to the `provide_params()` API tested here.

The GAT reads the directed flow network: source/sink flags, in/out degrees
and capacity sums at nodes; capacity, direction and ownership on arcs;
size, budget fraction, lambda and capacity scale at graph level. Fixed
boundary arcs remain visible but cannot be interdicted. Three GATv2 layers
and mean pooling classify a graph into a donor schedule. The training-only
transfer-energy profiles are clustered, with a medoid schedule per cluster.
IBM uses nearest graph-level features standardized on the donor bank.
Neither method averages periodic angles.

Install the optional dependencies with Python 3.11 or newer and run the
controlled comparison:

```bash
pip install -r requirements.txt -r requirements-qaoa.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-transfer.txt
OMP_NUM_THREADS=1 python run_parameter_transfer.py --out results/parameter_transfer.csv
```

The default experiment trains on 12 weighted two-route networks with four
vertices, then freezes all models and evaluates six unseen five-vertex
networks at depths 1 and 2. Cases have budget 1 and are conditioned on a
nonempty penalty window; greedy is therefore exact on this cohort. Raw
energy and probability scores distinguish the quantum methods even when
repair returns the optimum for all of them. It compares per-instance COBYLA, the first fixed
donor, IBM transfer, GAT transfer, and a regularized warm start with a pooled
training schedule. The CSV reports exact QUBO approximation ratios, energy
gap closed, optimal-interdiction probability, raw feasibility and repaired
flow. Training times, every graph and donor angle, and GAT checkpoints are
saved alongside it. Output files are never overwritten. These small graphs
test the transfer workflow; they do not establish performance on hard NIP
instances. Split by graph before adding other budgets to a training bank.

The completed [60-row comparison](results/parameter_transfer.csv), its
[graph and donor record](results/parameter_transfer.json), and the two GAT
checkpoints are saved in `results/`. At depth 2, mean energy-gap closure
over the same six targets is 0.6528 for the fixed donor, 0.6511 for IBM,
0.6621 for GAT, 0.6695 for target-trained QAOA, and 0.8610 for the warm
start. The warm start has lower mean optimal-interdiction probability than
target-trained QAOA (0.1183 versus 0.1377), despite its better energy.
All 60 rows reach the exact flow after repair, as does greedy from scratch.
Transfer uses no target quantum-angle optimization evaluations, but model
training, exact penalty selection and warm-start classical setup still cost
work. The results come from one split and seed, not a generalization study.

[qaoa_warm_start.py](qaoa_warm_start.py) adapts the regularized product state
and tilted mixer of [RWS-QAOA](https://arxiv.org/abs/2603.10191) to the NIP
minimization Hamiltonian. The regularization strength is in normalized
Hamiltonian units and is separate from the budget penalty. Its fixed
schedule is trained on the small NIP graphs, not copied from Max-Cut.
Only the classical product-state optimization is repeated on a target.

`transferred_piece_solver(transfer)` supplies the callback expected by
`decomposition.solve_decomposed()`, using MPS and returning arcs in the
original graph. For example, with an `IBMTransfer` or `GATTransfer` model:

```python
from qaoa_transfer import transferred_piece_solver
from decomposition import solve_decomposed

solve_piece = transferred_piece_solver(transfer, max_qubits=156)
cuts, flow = solve_decomposed(graph, src, tgt, pieces, budgets, solve_piece)
```

Reload a saved training bank with
`load_donors('results/parameter_transfer.json', reps=1)` from `qaoa_transfer`,
then construct `IBMTransfer(donors)` or call
`GATTransfer.load('results/parameter_transfer.p1.gat.json', donors)`.

Each piece gets its own validated penalty window. An empty window is an
error, not a fallback to a different Hamiltonian. Window selection still
uses exact classical solves (Gurobi by default), so this integration does
not yet remove that scaling cost. The GAT and IBM dependencies are optional
and excluded from the core installation. Run their tests with
`REQUIRE_TRANSFER_BACKENDS=1 python -m pytest tests/test_parameter_transfer.py`.

## Mid-size instances and matrix-product-state QAOA

Between `tiny_c` (22 qubits) and `50.planarflow` (912) there were no
instances at all. [instances.py](instances.py) makes nine, in
[testcases/mid/](testcases/mid/): planar, Erdős–Rényi and k-NN networks
whose QUBOs have 39 to 149 variables, so each fits a 156-qubit IBM device.
They are seeded and reproducible, and
[results/mid_instances.csv](results/mid_instances.csv) lists their exact
optima for the 23 budgets worth benchmarking (λ window with an interior, and
flow left). On every one of those rows greedy is already optimal, and on 22
the optimal cut touches the source or sink: like the random `testcases/`
families, they exercise the machinery but are easy classically (#1).

Past 24 qubits, `qaoa_qiskit.py --method mps` simulates the circuit as a
matrix product state, with `--bond-dimension` capping it (approximate, but
polynomial). Qubits are then ordered by reverse Cuthill–McKee, which keeps
coupled variables close. Without that order, one 78-qubit evaluation had not
finished after ten minutes; with it a 40-qubit one takes 0.1 s. The exact
optimum and ground energy come from Gurobi (`--exact-backend gurobi`), and
the run is scored from shots and by `gap_closed`: the share of the energy
gap between the uniform state (0) and the ground state (1). That measure
replaces the approximation ratio, on which random guessing already scores
0.56–0.58.

```
python qaoa_qiskit.py testcases/mid/knn_11_2_s0.flow --budget 2 --reps 1 \
    --method mps --bond-dimension 32 --exact-backend gurobi --max-qubits 200
```

[run_qaoa_mid.py](run_qaoa_mid.py) runs every manifest row at depths 1 to 3
with the angles trained on `tiny_a` reused unchanged (parameter transfer,
#16). For comparison, it also trains the angles directly, at every depth, on
the rows of up to 81 qubits (`--train-max-qubits`).

[results/qaoa_mid.csv](results/qaoa_mid.csv) contains the completed run:
69 transferred-angle rows (all 23 cases at depths 1, 2 and 3), plus 30
directly trained rows (the ten cases of at most 81 qubits at those depths).
These use MPS bond dimension 32, 4096 shots and seed 0; the header identifies
the clean source commit and command. The simulations are approximate and
the instances are already solved optimally by greedy, so these rows show
simulator reach and angle-transfer behavior rather than a quantum advantage.

The saved `qaoa_mid.csv` predates the deterministic RCM-order fix and did
not record its variable permutation. Its original approximate energies and
timings are historical measurements; they are not exactly reproducible from
the seed alone, since Python hash randomization affected the old ordering.
New runs use stable integer labels for RCM and record the actual `variables`
order and framework versions. Regenerate into a separate file when comparing
new results; the historical CSV is retained unchanged.

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
[utils.py](utils.py) and its figures are saved to
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

Two additional baselines are opt-in:

```bash
python run_benchmark.py --exact brute --solvers sa --baselines greedy,random,lp \
    --case testcases/tiny_a.flow:1:auto --out /tmp/lp_comparison.csv
```

`lp` solves a **continuous LP relaxation** with SciPy/HiGHS, rounds each
interdiction variable at 0.5, and applies the same greedy budget repair used
for annealer samples. `LP_Round_Repair` is the surviving flow of that feasible
cut; `LP_lower_bound` is a lower bound, not a feasible solution. Both are in
flow-capacity units. `LP_time_sec` includes LP construction, solve and bound
verification; `LP_repair_time_sec` includes rounding, repair and final scoring.
The plot draws the feasible result alongside the other baselines. Historical
CSVs without these optional columns remain supported.

The LP minimizes `sum(c_e*z_e)` subject to `pi_u-pi_v <= x_e+z_e`,
`sum(x_e) <= B`, `pi_s=1`, `pi_t=0`, with every variable in `[0,1]`.
This is the projection of the McCormick linearization of the binary
min-cut/interdiction model. Relaxing integrality gives a lower bound; its
integrality gap can be large, as shown by
[Altner, Ergun and Uhan (2010)](https://doi.org/10.1016/j.orl.2009.09.013).
No approximation guarantee is claimed for rounding and repair.

The bound is independently reconstructed from HiGHS' inequality marginals.
For `A*v <= b`, box bounds `l <= v <= u`, and any `y <= 0`,
`b*y + min_{l<=v<=u} (c-A.T*y)*v` is a lower bound. We project the returned
duals to that sign, evaluate the expression using exact rational arithmetic
on the input numbers, and round the final float downward. Thus a numerical
dual-stationarity residual cannot silently turn an LP objective into an
invalid bound. `LP_objective` separately records the solver's primal value.

`--baselines lp,lagrangian` also runs the existing **binary Lagrangian**
search, requiring Gurobi. This is different from the LP relaxation: it
removes the budget constraint and adds `lambda*(sum(x)-B)` to the objective.
The search returns each iteration's multiplier, global solver lower bound,
objective, cut count and surviving flow in `Lagrangian_iterations` (JSON).
`Lagrangian_lower_bound` retains the best bound, subject to Gurobi's numerical
tolerances; `Lagrangian_Feasible` retains only budget-feasible incumbents and
is empty if none was found. The search uses eight threads and at most 100
iterations; midpoint selection need not maximize the Lagrangian bound.
`Lagrangian_time_sec` includes model construction and all iterations.

When either bound pass runs, `Lower_bound` is the largest available lower
bound. Each available feasible SA, QA or baseline result `F` gets a `_gap`
column `(F-Lower_bound)/F`, an upper bound on relative suboptimality for this
minimization problem. Zero flow has gap zero; unrun or infeasible results
remain empty. These are gaps in **surviving flow**, not the flow-reduction
ratios plotted elsewhere.

`--baselines greedy,random,lp,lagrangian,benders` adds a **Benders baseline**
with a binary interdiction master solved by Gurobi and max-flow subproblems
solved by NetworkX. It accepts the nonnegative integer capacities used in
`.flow` files. `Benders` is the best feasible surviving flow,
`Benders_lower_bound` is the master lower bound, and `Benders_status` is
`optimal`, `time_limit`, `iteration_limit` or `interrupted`. A limited run
still returns its feasible incumbent and available lower bound; it does
not become an exact reference. The bound also contributes to `Lower_bound`
and the `_gap` columns. Unrun Benders columns remain empty.

```bash
python run_benchmark.py --exact brute --solvers sa \
    --baselines greedy,lp,benders --benders-time-limit 60 \
    --benders-max-iterations 1000 --case testcases/tiny_a.flow:1:auto \
    --out /tmp/benders_comparison.csv
```

The master minimizes `eta` over binary `x`, with `sum(x) <= B` and
`eta >= 0`. For any feasible original-network flow `f` of value `v`, add
`eta >= v - sum(f_e*x_e)`. To see validity, take any source–sink cut: its
surviving capacity is at least its surviving forward flow, which is at
least `v - sum(f_e*x_e)` because backward flow is nonnegative and the sum
subtracts removed flow over *all* arcs. The min-cut theorem gives the bound
on surviving max flow. A maximum flow after the current interdiction has
zero flow on removed arcs, making its inequality tight at that interdiction.
Thus exact master solves and complete cut generation converge over the
finite binary set. This argument is specific to maximum-flow interdiction;
[Israeli and Wood (2002)](https://doi.org/10.1002/net.10039) treats
shortest-path interdiction, and this implementation does not reproduce its
algorithm or supervalid inequalities.

`Benders_iterations` counts master solves; `Benders_history` records each
completed candidate's bounds, feasible flows, elapsed time and master status.
`Benders_interdicted_edges` records the final feasible cut as JSON so its
surviving flow can be independently recomputed.
`Benders_time_sec` includes model construction, all master solves and all
max-flow subproblems. Masters use eight threads and zero requested MIP gap;
their bounds and stopping comparisons remain subject to Gurobi's numerical
tolerances. The default limits are 60 seconds and 1000 master solves per
pass. The wall limit is cooperative: a NetworkX max-flow call in progress
may overrun it, after which no further solve starts. Exact reference and
other baseline passes have their own timing and are outside that allowance.
For timing comparisons, `Gurobi_total_time_sec` includes the monolithic
trajectory model's construction, solve and final flow scoring;
`SA_total_time_sec` includes QUBO construction, sampling and postprocessing.
The existing `Gurobi_time_sec` remains solver-only for compatibility.
`Reference_time_sec` separately records exact reference/QUBO verification
and any auto-lambda calibration; do not treat this setup work as free when
reporting the complete QUBO pipeline cost. These columns exclude graph-file
loading, and the total solve times include their usual diagnostic output.
The complete same-instance per-family comparison remains work under #15.

The saved [three-family sample](results/classical_baselines_sample.csv)
runs these methods on the same three existing slide cases, using the
command and environment recorded in its
[manifest](results/classical_baselines_sample.json). Surviving flow is
minimized; all three Benders runs closed their bound gaps.

| Instance (budget) | Exact | SA + repair | Greedy | LP + repair | Benders [lower, upper] | Master solves |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `50.planarflow` (5) | 276 | 311 | 276 | 276 | [276, 276] | 9 |
| `50_0.15.erdosflow` (5) | 358 | 393 | 358 | 358 | [358, 358] | 6 |
| `50_5.2dnnflow` (4) | 165 | 326 | 165 | 165 | [165, 165] | 9 |

One-run elapsed seconds, rounded here only for readability:

| Instance | Gurobi total | SA total | Greedy | LP + repair | Benders total | Reference/calibration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `50.planarflow` | 0.022 | 24.338 | 2.482 | 0.016 | 0.111 | 0.312 |
| `50_0.15.erdosflow` | 0.028 | 28.116 | 3.296 | 0.015 | 0.049 | 0.376 |
| `50_5.2dnnflow` | 0.027 | 31.562 | 2.281 | 0.017 | 0.075 | 0.311 |

The sample uses eight Gurobi threads, one thread for the listed numerical
libraries, and SA seed 0 with 2500 reads and the installed sampler's 1000
sweeps default. There is one run per case and no dedicated warmup; references
run first in the same process. These easy cases also have exact LP bounds
and optimal greedy solutions. They validate execution and saved-cut scoring,
not hardness, statistical timing differences, scaling or a speedup.
The CSV leaves hardware and other unrun methods empty. Decomposition and
the remaining same-instance baselines are still needed for #15's full table.

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
The window is often only about one unit wide on the benchmark instances. It
has no strict interior when `lambda_lo > lambda_hi` (the budget lies above
the lower convex envelope of optimal flow) or `lambda_lo == lambda_hi`
(other cut counts tie at the only candidate lambda, even for a convex flow
curve). Neither case guarantees exactly B cuts, so the row keeps the exact
optimum and the baselines but no QUBO or solver columns. With a hand-picked
number instead, `run_benchmark.py` still checks that the exact QUBO optimum
decodes to an optimal interdiction and refuses to write the row otherwise.
