# Penalties for the network-interdiction QUBO

The original model uses a **linear** budget term. Its exact window is useful
when nonempty and keeps the graph sparse. The optional squared model provides
closed-form sufficient coefficients when a guaranteed exact-budget encoding
is needed, at the cost of a clique on the interdiction variables. This does
not change the default model or claim that larger penalties improve sampling.

## Model and assumptions

Minimize surviving maximum s–t flow after interdicting at most B allowed arcs.
Capacities are nonnegative, all interdiction costs are one, and
0 ≤ B ≤ number of allowed arcs. The implementation's certified entry point
requires integer capacities and a simple directed graph with node labels
0,…,n−1. Source and sink are distinct. A smaller interdiction set can be padded
to size B without increasing flow, so the exact-B and at-most-B optimum values
coincide; their sets of optimizers need not coincide.

For an allowed arc write a=π_u, b=π_v, c=γ_uv and w for its auxiliary bit.
The polynomial in `nip_to_qubo` simplifies to

```
D(a,b,c,w) = w(a+b+c−1) + bc − b − c.
```

The normalized term D+1 is a nonnegative integer, dominates a(1−b)(1−c),
and attains that cubic when w=1 exactly for a=b=c=0 (this is one valid
minimizing choice, including ties). These are exhaustive Boolean identities
proved by `quadratization_lower`, `quadratization_attained`,
`dualTerm_nonnegative`, and `dualTerm_positive_gap`.

## Linear model: exchange and strict window

For a violated cut a=1,b=c=0, D=0 for either w. Replacing this arc's variables
by x=c=1,w=0 changes its local contribution from λx to λ−β. Thus β>λ≥0
gives a strict improvement, regardless of capacity. This is `dual_exchange`;
the rest of the objective is arbitrary and unchanged. It rules out violated
cuts in a ground state; it does **not** separate every infeasible assignment
from every feasible assignment.

After minimizing the auxiliaries and cut variables, define f(k) as the best
surviving flow with exactly k interdictions. The cardinality part minimizes
f(k)+λk (up to the common constants −λB−β|A_I|). Exactly B is the unique
optimal cardinality iff

```
max_{k>B} (f(B)−f(k))/(k−B) < λ < min_{k<B} (f(k)−f(B))/(B−k).
```

The Lean theorem `strictSupport_iff_strictWindow` proves this equivalence
for an arbitrary finite real-valued curve. Its universal-quantifier form
handles empty endpoint sets. `window_exists_iff_exposed` proves existence
iff a line through (B,f(B)) lies strictly below every other sampled point:
the exposed lower-support characterization, with slope −λ. These statements
allow any real λ; a nonnegative-penalty application must additionally
intersect the window with λ≥0. The formal definition uses the supporting
line directly, rather than importing a convex-hull geometry library.

A straight segment f=(10,5,0), B=1 has window (5,5): increasing λ cannot
fix it. The comparison includes two equal-capacity disjoint paths realizing
this example. Do not interpret β>λ as a sufficient bound for the squared
model: the same x repair can then raise the budget term by more than λ.

## Squared model: simultaneous separation

Let U=Σu_e over all arcs, m=|A_I|, k=Σx, and Q=Σ(D+1) over allowed arcs.
Fixed arcs contribute their ordinary quadratic cut capacity. For the squared
model, the normalized energy is

```
E + βm = C + λ(k−B)² + βQ,       0 ≤ C ≤ U.
```

Choose **λ=β=U+1**. Every encoded feasible assignment (k=B,Q=0) has
normalized energy C≤U. If either condition fails, its integer penalty gap
is at least one, and the other penalty is nonnegative. Consequently every
encoded infeasible assignment has normalized energy >U, including joint
violations. `squared_budget_gap` and `simultaneous_penalty_separation` prove
the cardinality gap and the real-arithmetic separation. A feasible original
cut has an auxiliary completion with Q=0; nonminimizing auxiliary choices
are not included in the phrase "encoded feasible."

The coefficients have the same units as flow. They are sufficient, generally
conservative, and require no optimization or bisection. The Python helper
rejects noninteger capacities and a conservative absolute coefficient-mass
bound above 2^52; this keeps integer expansion/evaluation inside binary64's
exact integer range. The Lean statements use exact real arithmetic and do
not formalize PyQUBO compilation, NetworkX, Gurobi, or floating-point code.
Exhaustive Python truth tables independently check the actual compiled
polynomial, simultaneous separation, and min-cut optimum on small graphs.

Usage:

```python
from certified_penalties import certified_squared_qubo
bqm, certificate = certified_squared_qubo(B, graph, source, sink)
# bqm ground energy + certificate.normalized_offset is the optimum flow.
```

## Relation to Gurobi_QP

The repository historically named Gurobi_QP is now
[Quantum_Preconditioning_Constrained_Optimization](https://github.com/SECQUOIA/Quantum_Preconditioning_Constrained_Optimization).
Its [critical-penalty enumeration at commit 9f42d138](https://github.com/SECQUOIA/Quantum_Preconditioning_Constrained_Optimization/blob/9f42d1384394d3ca8c49a8f268e03bd13d63ede7/scripts/recalculate_rho_star.py)
computes max(0,max_infeasible (H*−H(z))/(Σz)²) for a balanced-spin problem.
That ratio concerns comparison with the **best** feasible value. Here the
stronger every-feasible/every-infeasible statement uses the full objective
range U and a unit gap in either of two nonnegative penalties. The spin
imbalance and NIP budget gaps have different scales. The other repository's
main preconditioning benchmark retains a hard balance constraint; it is not
evidence that its production QP was replaced by an unconstrained QUBO.

## Reproduce the proof

Install Lean from the pinned `lean-toolchain`, then from this directory:

```sh
MATHLIB_NO_CACHE_ON_UPDATE=1 lake update
lake exe cache get Mathlib.Basic.Real.Basic Mathlib.Tactic.Linarith Mathlib.Tactic.NormNum
lake --wfail build
```

Lean 4.35.0 and mathlib commit 56a005855ea17e0b0958f73ec801ae140b0a1679
are pinned, including transitive dependencies in `lake-manifest.json`.
All theorem axiom reports are guarded: only `propext`, `Classical.choice`,
and `Quot.sound` are accepted; proof holes or extra axioms fail the build.
The dedicated GitHub job performs the same build. On a memory-constrained
machine, `LEAN_NUM_THREADS=2 LEAN_IMPORT_WORKERS=2 LEAN_STACK_SIZE_KB=16384`
reduces Lean's default stack reservations.

## Retained software comparison

[`results/certified_penalties.json`](../results/certified_penalties.json) records
six original tiny rows, the equal-path empty-window case, and one planar,
Erdős–Rényi, and k-NN mid-size case. All ten squared-QUBO ground states attain
the original MILP optimum after decoding. Full binary witnesses, objective
bounds, input/source hashes, and dependency versions are retained. Nonunique
optimal sets are compared by feasibility and value, not by edge-list equality.

| Instance | B | Variables | Linear couplings | Squared couplings | Optimum flow |
|---|---:|---:|---:|---:|---:|
| tiny_a | 1 | 14 | 14 | 20 | 5 |
| tiny_b | 1 / 2 | 21 | 21 | 36 | 13 / 5 |
| tiny_c | 1 | 22 | 24 | 39 | 4 |
| tiny_d | 1 / 2 | 21 | 21 | 36 | 13 / 6 |
| equal_paths | 1 | 14 | 14 | 20 | 5 |
| planar_8_s0 | 3 | 90 | 112 | 490 | 43 |
| erdos_8_0.45_s0 | 3 | 78 | 97 | 373 | 36 |
| knn_11_2_s0 | 3 | 81 | 99 | 375 | 15 |

Coupling counts compare the two budget expressions at identical coefficients;
the linear model at these conservative coefficients is **not** asserted to
encode B. Every row adds exactly m(m−1)/2 interdiction couplings. No extra
binary variables are needed. Timings are single software runs, not an
advantage claim: the original MILP was faster than the squared MIQP on every
row. The larger coefficient range can also hurt approximate samplers; none
were run for this comparison, and no hardware result is inferred.

The completed campaign used Gurobi 13.0.3, eight threads, MIPGap=0, a
20-second limit per solve, and a 300-second outer limit per attempt under an
8 GiB process limit. The driver caps cases at 12, arcs at 50, and compiled
variables at 160 before solver dispatch. It requires both solves to finish
optimally and independently checks the decoded max flow. CI audits the saved
witnesses without needing a license.

The [attempt ledger](../results/certified_penalties_attempts.json) retains an
earlier export failure after the first two solves and the successful 20-solve
attempt: 22 completed case solves total. The failed export produced no result
file, so its timings are not silently reused. Source hashes bind the saved
data to producer commit `1ac55133ec7dcd14d70b79d94a824221efd33bed`. The later
isolated-vertex audit fix does not affect these connected cases; their saved
witnesses are also checked against the final builder.

Reproduce from the repository root, writing a fresh output path:

```sh
python run_penalty_comparison.py \
  --case testcases/tiny_a.flow:1 \
  --case testcases/tiny_b.flow:1 --case testcases/tiny_b.flow:2 \
  --case testcases/tiny_c.flow:1 \
  --case testcases/tiny_d.flow:1 --case testcases/tiny_d.flow:2 \
  --case testcases/equal_paths.flow:1 \
  --case testcases/mid/planar_8_s0.flow:3 \
  --case testcases/mid/erdos_8_0.45_s0.flow:3 \
  --case testcases/mid/knn_11_2_s0.flow:3 --out /tmp/penalty-comparison.json
```
