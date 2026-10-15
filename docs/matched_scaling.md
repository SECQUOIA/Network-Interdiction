# Bounded software comparisons and simulator scaling

This protocol extends the [challenge cohort](challenge_profile.md) with comparable
software baselines and concrete resource limits. It uses the original at-most-B
network-interdiction objective, including each manifest's allowed arcs. The
SocioPatterns-derived rows retain the source's CC BY-NC-SA 3.0 terms; see the
[input notices](../testcases/challenge/README.md). No device jobs are submitted.

## Protocol fixed before the retained runs

`run_matched_benchmark.py` selects nine cases: EMA B=3, school class 4A B=3,
IEEE14 B=2, the three 50-node families B=6, covering n=12 B=20, covering
n=30 B=192, and planar n=150 B=6. The two archived challenge profiles supply
MILP/Benders references and their **original** timings; hashes bind input files,
allowed sets, budgets, and reference artifacts. These are not remeasured timings
from the new producing revision. There is one repetition, seed 0, BLAS/OpenMP
threads 1 and Gurobi threads 8. No statistical speedup claim is supported.

Each greedy, LP, SA or decomposition method receives a cooperative 20-second
allowance, including its setup and score/repair work. Up to 5,000 actual original
max-flow calls are permitted; repeated cuts are cached but still check the wall
limit. A 120-second process limit and 8 GiB address-space limit contain native
operations. Timeouts and failures have their own rows. The initial empty cut is
a feasible fallback; every improvement is independently scored on the original
network before it can replace the incumbent. Reported time includes auditing.

- Greedy uses the existing add/remove repair policy starting with no cuts.
- LP uses the same relaxation and half-threshold repair, plus a deterministic
  top-B fractional candidate. This is an explicitly named variant; it preserves
  a feasible improvement if removal repair is interrupted. Its rational bound
  certificate is separate from the feasible heuristic result.
- SA compiles one [certified squared-budget BQM](../certified_penalties.py), avoiding
  lambda bisection. It requests 200 reads, 1,000 sweeps, seed 0, with the seed
  passed to the actual sampler. It interrupts between reads at half the remaining
  wall allowance, leaving time for scoring and the shared repair. A read already
  running may overrun. The lowest-energy 20 distinct records retain bitstrings,
  energies, multiplicities, raw cut/flow and budget feasibility. These selected
  records are not the full shot distribution. Completed read counts are retained.
- Decomposition uses four seeded pieces, at most four terminals per contraction,
  mincut/local/marginal/common-multiplier splits, and sequential/one-round iterated
  reconstruction. Each child has at most five seconds within the shared allowance;
  up to 128 calls are reserved before dispatch. Curves require optimal children;
  oversize curve plans are refused before child dispatch. The multiplier dual
  bound concerns the allocation surrogate only. All scored feasible candidates,
  including curve-only and partial-prefix cuts, compete for the retained best
  flow. The final reconstructed cut/flow is reported separately when reached.
  Fixed splits may finish with audited inexact children; their status says so.

The fit pass compiles the certified squared BQM for the whole graph and all four
restricted pieces. It uses proportional allowed-arc budget splitting, which is a
size probe rather than a claim to reproduce every adaptive reconstruction model.
It checks 24-qubit statevector/256-qubit capped-MPS guards, constructs the actual
Dirac-3 request schema offline (474 binary variables / 949 optical levels), and
attempts a verified embedding on ideal Pegasus P16 with seed 0, one try, three
seconds. Every piece must embed before `all_pieces_embedded` is true. A failed
heuristic embedding does not establish infeasibility; an offline schema pass does
not establish device availability, precision, performance, or hardware success.

`run_simulator_scaling.py` tests 14, 21, 40, 78, 116, 149 and 1,095 logical
variables at B=1 except the hard covering case B=192. It compares squared-penalty
QAOA with XY basis initialization at the **same dual penalty**, plus SA and an
original-problem MILP reference. The hard case reuses its archived optimum.
Both QAOA modes use p=1, eight optimizer evaluations, 256 shots, seed 0, RCM
variable ordering, one start and no connectivity constraints. Statevector caps
at 24 qubits; approximate MPS caps at 256 qubits with bond dimension 8 and
truncation threshold 1e-16. Each call has 60 seconds and 8 GiB, including scoring
and repair. Full observed shot counts are retained. In particular, MPS need not
preserve the XY fixed-weight subspace exactly: raw feasibility is measured.

“Meaningful” is fixed before execution as a completed sampler with at least one
selected raw budget-feasible cut improving on empty-cut flow. This is a modest
usability criterion, not a quality/advantage claim. A largest passing case is
specific to this ladder, bond cap, circuit, settings and machine. The XY basis
initial cut is reported separately and excluded from the sampler incumbent.

## Reproduction

Use the repository environment plus QAOA/IBM dependencies and a Gurobi license.
Install Python dependencies using `uv` into a local `.venv`. Outputs must be new
paths: exclusive creation prevents overwriting an earlier attempt. The parent
checkpoints before and after every method, including explicit refusals/failures.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python run_matched_benchmark.py --out results/matched_scaling.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python run_simulator_scaling.py --out results/simulator_scaling.json
```

Results and their independent artifact audits are added after execution. This
protocol does not imply that a challenging full instance becomes simulatable or
embeddable merely because it has been partitioned.

## Execution notes

The first matched campaign completed all 108 dispatches and retains the largest
planar fit pass as `outer_time_limit`: minorminer's internal timeout did not
contain all native setup work. A supplementary fit run adds a 12-second process
limit to each embedding attempt and retains all five whole/piece size records.
The first attempt is not overwritten or relabeled as successful.

The [source-edit note](../results/matched_scaling_execution.json) preserves a
CSV/plot-only patch temporarily present during part of the first campaign.
The recorded hashes are the committed source at campaign start; no solver,
model, scoring, reference or resource-control function changed. In particular,
the imported `run_benchmark.parse_case` was unchanged. The patch was restored
before the last two cases and applied permanently only after the campaign.
This is not claimed to be an immutable whole-tree execution. All reported
results retain the original producing snapshot and this exception.
