# Budget-preserving QAOA

The optional `FixedWeightMixer` in `qaoa_xy.py` keeps the interdiction
register at exactly B ones. This replaces the linear cardinality penalty
for that variant. With unit interdiction costs, nonnegative capacities,
and B no larger than the allowed arc count, an optimum using fewer arcs
can be padded without increasing surviving flow, so the original at-most-B
problem has the same optimum value.

Only the x register is constrained by the mixer. The cut and auxiliary
variables π, γ, w retain X mixing and the existing cut-link penalty.
Their nonlinear relationships are not simple cardinality constraints;
forcing them into this mixer would require a different transition design
and connectivity proof. This implementation makes no claim that every
measured state satisfies those cut-link constraints.

## Circuit and conventions

The cost layer is unchanged: RZ(2γ h/scale) and RZZ(2γ J/scale), with
the original BQM offset restored when reporting energies. The parameter
order remains `[beta layers, gamma layers]`. Register indices are explicit
and can be noncontiguous in the full variable order.

Each XY pair applies RXX(2β) followed by RYY(2β), equal to
exp(−iβ(XX+YY)). It fixes |00⟩ and |11⟩ and mixes |01⟩ with |10⟩, so
it preserves the pair's Hamming weight. The pair order is even adjacent
pairs, odd adjacent pairs, then the ring-closing pair; a two-qubit register
uses its pair once. The product preserves total register weight. It is a
partitioned alternating-operator mixer, **not generally** exact evolution
under the sum of all ring terms, which need not commute.

This follows the constraint-preserving framework of
[Hadfield et al. (2019)](https://arxiv.org/abs/1709.03489) and the distinction
between simultaneous and partitioned XY mixers in
[Wang, Rubin, Dominy and Rieffel (2020), §III.B.2–3](https://arxiv.org/pdf/1904.09314).
[Qiskit's RXX convention](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.RXXGate)
uses half the supplied rotation angle, hence the factor of two above.
The cited graph-coloring experiments are not replications or evidence of
performance on network interdiction.

Two feasible starts are available:

- `basis`: set the first B variables in the supplied register to one.
  This uses no exact solution or greedy warm start; its quality can depend
  strongly on arc order and is reported separately in the comparison.
- `dicke`: uniformly superpose all weight-B register strings. This uses
  Qiskit's generic `StatePreparation`, with a hard cap of 12 register qubits
  before allocating the dense amplitude vector. It is a bounded simulator
  implementation, not the efficient polynomial Dicke-preparation algorithm.
  No assumption that it is a ground state of the partitioned ring mixer
  is needed or claimed.

Other qubits start in |+⟩ and use RX(2β). A product-state warm start cannot
be combined with `FixedWeightMixer`: it would generally leave the required
sector. Circuit resource counts include state preparation, cost, and mixer
gates after decomposition to the same logical basis used by penalty QAOA;
they assume unconstrained connectivity and exclude measurement/routing.

## The remaining dual penalty

Set the budget coefficient to zero for the XY variant and choose β greater
than the largest arc capacity. For a violated cut constraint a=π_u=1,
b=π_v=0,c=γ=0, the old dual polynomial D is zero. Setting c=1,w=0 while
holding x fixed changes its contribution by u(1−x)−β<0. This repair preserves
the register budget. Nonminimizing auxiliary choices can also be repaired
without changing x. Thus a constrained ground state is a valid min cut.

The old β>λ exchange changes x and cannot justify the new fixed-budget
variant. A one-arc B=0 regression demonstrates that β>λ=0 alone permits
cut-link violations, while β>capacity restores the correct flow. This is a
ground-state guarantee, not a statement that arbitrary samples are feasible.

The bounded comparison uses a common β=max(max capacity, linear λ)+1 for
all methods. Penalty QAOA and SA use λ inside the exact strict window;
XY uses λ=0 and fixes the budget through its circuit. If the window is
empty, the two linear-penalty methods are explicitly marked inapplicable;
the XY variants still run. The squared alternative in PR #34 is a separate
encoding and is not silently substituted in this comparison.

## API and validation

```python
from solve_qubo_via_qaoa import solve_bqm_qaoa
samples = solve_bqm_qaoa(
    bqm_without_budget_penalty, reps=1, shots=1024, maxiter=20, seed=0,
    fixed_weight_variables=[f'x_{i}' for i in range(number_of_arcs)],
    hamming_weight=B, initial_state='dicke')
```

The sampler does not alter the supplied BQM. The caller must supply the
appropriate remaining penalties. Invalid registers, duplicate indices,
nonintegral/out-of-range weights and oversized Dicke states are refused.
The sampler's existing 24-qubit cap still applies before circuit building.
The direct `QiskitBackend(..., fixed_weight=...)` API also supports the
existing MPS method, but the comparison below uses exact statevectors.
Capped MPS introduces approximation error, so the mathematical circuit
invariant is not a promise about every truncated numerical state.

Tests compare actual Aer probabilities and expectations with independent
binary-basis evolution using original BQM energies. They cover odd/even
rings, the two-qubit case, noncontiguous register order, zero/full and
multi-excitation weights, shot decoding, and resource/failure guards.
The dedicated QAOA CI lane requires the actual backend dependencies.

## Bounded comparison

`run_xy_comparison.py` uses the existing `qaoa_common.score_samples`
semantics for every method: all shots contribute to raw feasibility and
optimal-interdiction probability; the lowest-energy 20 distinct records
receive classical greedy repair. Full counts and variable order are saved.
Original MILP optima are cross-checked against exhaustive arc-subset
max-flow values. An optimal x set can occur in a sample with inconsistent
cut auxiliaries; `shot_p_optimal` refers to the interdiction, not QUBO
ground-state probability.

The driver refuses more than seven arcs or 24 compiled qubits before
enumerating references or dispatching solvers. It caps four cases, p≤3,
10,000 shots and 100 objective evaluations; each original MILP has a
20-second limit and eight threads. Every completed method is checkpointed
before the next begins. Use an external wall-clock/memory limit for the
whole simulation campaign, and write a new output path for each attempt.

Raw energies across the two Hamiltonians are not comparable quality scores.
Compare surviving flow, shot optimality, repair contribution, circuit depth,
and total work instead. Timing includes building/training/shots, with stages
separately retained; the first quantum method also pays lazy backend import
costs. A single seeded run is a feasibility check, not statistical evidence
of a speed or quality advantage. Hard-instance and larger-scale comparisons
remain a separate part of the software series; no hardware is run here.
