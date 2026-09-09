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
