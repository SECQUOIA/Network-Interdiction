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

The retained results below include independent artifact audits. This
protocol does not imply that a challenging full instance becomes simulatable or
embeddable merely because it has been partitioned.

## Execution notes

The first matched campaign completed all 108 dispatches and retains the largest
planar fit pass as `outer_time_limit`: minorminer's internal timeout did not
contain all native setup work. A supplementary fit run adds a 12-second process
limit to each embedding attempt and retains all five whole/piece size records.
The first attempt is not overwritten or relabeled as successful. The saved
supplement used the POSIX fork worker at its pinned producer. The final driver
uses a fresh spawned interpreter to avoid inheriting native threads from callers;
its containment and successful mapping are checked independently. Timings from
the archived fork run are not relabeled as measurements of the spawn path.

The [source-edit note](../results/matched_scaling_execution.json) preserves a
CSV/plot-only patch temporarily present during part of the first campaign.
The recorded hashes are the committed source at campaign start; no solver,
model, scoring, reference or resource-control function changed. In particular,
the imported `run_benchmark.parse_case` was unchanged. The patch was restored
before the last two cases and applied permanently only after the campaign.
This is not claimed to be an immutable whole-tree execution. All reported
results retain the original producing snapshot and this exception.

## Retained matched results

The [raw campaign](../results/matched_scaling.json) has 108 method dispatches;
the [supplement](../results/matched_fit_supplement.json) adds one fit dispatch.
The independent exporter recomputes all retained original-network witnesses,
raw sample energies/flows, multiplier surrogate bounds and successful embedding
certificates. It produced seven family CSVs: [road](../results/matched/road.csv),
[contact](../results/matched/contact.csv), [power](../results/matched/power.csv),
[planar](../results/matched/planar.csv), [Erdős–Rényi](../results/matched/erdos.csv),
[k-NN](../results/matched/knn.csv), and [covering](../results/matched/covering.csv).
They use the shared benchmark columns, add explicit statuses and compiled qubit
counts, and leave hardware columns empty. The plot understands actual zero-flow
raw solutions when feasibility counts are supplied, preserving the older zero
sentinel for historical CSVs without those diagnostics.

Flows below are minimized, in each input's own units. Decomposition columns
show the best audited feasible candidate after one iterated round, including
curve work and partial prefixes. Sequential results and reconstructed-cut
results are separately retained in JSON/CSV; no surrogate bound is a global
NIP bound. A dash means the curve plan was refused before any child solve.

| Case (B) | Optimum | Benders | Greedy | LP variant | SA + repair | Mincut | Local | Marginal | Multiplier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ema (3) | 0 | 0 | 0 | 0 | 6102 | 0 | 10599 | 0 | 0 |
| school4a (3) | 1640 | 1640 | 1640 | 1640 | 6100 | 1640 | 1640 | 1640 | 1640 |
| ieee14 (2) | 53 | 53 | 53 | 53 | 193 | 53 | 53 | 76 | 76 |
| 50.planarflow (6) | 205 | 205 | 205 | 205 | 573 | 243 | 476 | 205 | 205 |
| 50_0.15.erdosflow (6) | 294 | 294 | 294 | 294 | 636 | 319 | 394 | 294 | 294 |
| 50_5.2dnnflow (6) | 75 | 75 | 75 | 75 | 361 | 86 | 220 | 75 | 75 |
| cover_n12_s0 (20) | 4 | 4 | 5 | 5 | 7 | 9 | 7 | 9 | 6 |
| cover_n30_s0 (192) | 8 | 14 | 30† | 16† | 20 | 28 | 16 | — | — |
| 150.planarflow (6) | 691 | 691 | 774† | 691 | 1130 | 691 | 995 | 691 | 691 |

† Cooperative time limit reached; the feasible incumbent is retained. Benders
on the hard covering case reached its 20-second limit at flow 14 with lower
bound 3. The archived MILP proved flow 8 in 12.16 seconds. The nine-case
comparison therefore supplies no advantage claim: even its best completed
decomposition result is 16 on this hard case. Greedy and the LP variant match
the optimum on six easy cases; LP also matches the largest planar optimum.

SA completed 200 reads on eight cases and 61 on the largest planar case before
its sampling allowance stopped it. All 20 retained records per case satisfy
the budget. This shows that certified penalties can remove parameter search
and yield feasible samples without good objective values. The shared greedy
policy only adds/removes arcs to meet B; it does not improve an already exact-B
cut by swaps. These SA results must not be attributed to a local-swap search.

## Whole-model and four-piece size limits

| Case | Whole variables | Largest piece | All pieces pass QCI schema | All pieces embedded on ideal P16 |
| --- | ---: | ---: | --- | --- |
| ema | 846 | 387 | yes | no |
| school4a | 1273 | 361 | yes | no |
| ieee14 | 132 | 33 | yes | yes |
| 50.planarflow | 912 | 261 | yes | no |
| 50_0.15.erdosflow | 1122 | 315 | yes | no |
| 50_5.2dnnflow | 1068 | 375 | yes | no |
| cover_n12_s0 | 142 | 97 | yes | yes |
| cover_n30_s0 | 1095 | 696 | no | no |
| 150.planarflow | 2812 | 1150 | no | no |

The largest whole model with a successful squared-BQM P16 embedding here is
142 variables (covering n=12), whose four pieces top out at 97 variables. The
largest whole model whose four pieces all pass the QCI schema guard is the
1,273-variable contact network (largest piece 361). That is an offline size
observation, not a hardware result. The hard 1,095-variable covering case still
needs pieces as large as 696; the 2,812-variable planar case needs 1,150.
Neither is reached by the four-piece QCI or capped-MPS configuration. The three
new 12-second embedding-worker timeouts in the supplementary run remain explicit.

These squared-budget embeddings include a clique among allowed interdiction
variables. Earlier [linear-BQM embedding measurements](../results/embedding_resources.json)
used a different interaction graph; their timings/fit rates are not a matched
penalty-only attribution experiment. Device execution, live defects/calibration,
and actual QPU times remain unmeasured.

## Simulator results

The [simulator artifact](../results/simulator_scaling.json) retains 42 method
dispatches, with full observed counts for each completed quantum sampler.
Statevector completed the 14- and 21-variable cases and refused larger inputs
at its configured 24-variable guard. Approximate MPS completed at 149 variables,
with bond dimension 8; this is the largest tested meaningful case, **not** a
universal maximum or a guarantee for smaller inputs.

| Variables (B) | Exact flow | SA + repair | Penalty MPS raw / repaired | XY MPS raw / repaired | XY exact-B shots / 256 |
| --- | ---: | ---: | --- | --- | --- |
| 14 (1) | 5 | 5 | 5 / 5 | 5 / 5 | 256 |
| 21 (1) | 13 | 13 | 13 / 13 | 13 / 13 | 256 |
| 40 (1) | 40 | 40 | 40 / 40 | 40 / 40 | 252 |
| 78 (1) | 144 | 144 | 144 / 144 | 144 / 144 | 253 |
| 116 (1) | 89 | 89 | 145 / 89 | native crash | — |
| 149 (1) | 209 | 223 | 209 / 209 | 303 / 303 | 253 |
| 1095 (192) | 8 | 20 | refused | refused | — |

The 116-variable XY MPS worker terminated with signal 11 after 5.03 seconds,
without a backend diagnostic; its row is retained and no repair or solution is
invented. The same fixed configuration completed the 149-variable XY case, so
successful size is not monotonic. Reproduce the isolated failed case with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout 60 .venv/bin/python run_simulator_scaling.py \
  --worker testcases/mid/planar_13_s0.flow 1 xy_mps --out /tmp/xy116-new.json
```

The exact statevector XY runs retained all 256 shots in the fixed-budget
sector. MPS XY retained 252/256 at 40 variables and 253/256 at 78 and 149;
its exact-unitary guarantee cannot be applied unchanged after truncation.
Raw surviving flows are audited only for the selected <=20 records; full-shot
budget counts include all observations. The 116-variable penalty MPS sampler
completed but failed the predeclared meaningfulness criterion: all its selected
feasible raw records left flow unchanged, and repair supplied the improvement.

At 149 variables, penalty/XY logical depths were 104/34 and two-qubit gate
counts were 1,223/280; method wall times were 15.18/4.05 seconds. These use
different cost Hamiltonians, unconstrained connectivity and one seeded run.
The shallower XY circuit returned repaired flow 303 versus the optimum 209;
no runtime or solution-quality advantage follows from the gate count alone.

The hard case has 1,095 variables and was refused by both simulator guards
before backend construction. Its exact reference is 8 and bounded SA returns
20. Its largest four-way piece is also above both guards. Thus a direct
QAOA comparison on the measured hard set remains beyond this validated local
configuration; this study records that gap instead of calling a refusal a run.
Actual device execution and QPU time remain excluded from this software series.

To regenerate family plots, run `network_interdiction_plot.py --csv` on each
family CSV. The committed flow-reduction PDFs are under
[`results/matched/figures`](../results/matched/figures). The existing energy-ratio
plot can also be generated, but its offset-dependent ratios should not be read
as independent solution-quality evidence. The benchmark's exact squared-BQM
energy column follows from the certified reduction: `Opt_NIP - beta*|A_I|`.

## Separate hard-case MPS probe

The initial ladder's refusal reflects its configured 256-qubit cap. A separate
`run_hard_mps_probe.py` experiment tests the actual 1,095-variable hard case
with stronger controls on the other resource dimensions: p=1, at most four
optimizer evaluations, 256 shots, bond dimension 8, at most 50,000 compiled
quadratic interactions, 8 GiB and 600 seconds per method. This is an explicit
`extended_mps=True` opt-in with a 1,200-qubit guard; normal adapter limits stay
unchanged. Statevector is never extended. Both methods use the same certified
dual coefficient and seed 0; XY starts from the first B allowed arcs.

The case and exact reference are the same archived hard covering instance.
Atomic checkpoints preserve model metadata before sampling, complete observed
counts before scoring, and raw scores before repair. Scoring/repair receives a
separate 20-second allowance inside the enclosing 600-second method cap. These
four-evaluation probe results are separate from the eight-evaluation ladder and
are not pooled with it. Only these two bounded attempts are planned.
