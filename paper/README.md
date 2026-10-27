# Network interdiction manuscript

Main document: `Main.tex`. Authors: Anurag Ramesh and David E. Bernal Neira.
This is a working research draft, with data frozen on 14 October 2026.

## Build and Overleaf

Run `bash paper/build.sh` from the repository root. It runs pdfLaTeX, BibTeX,
and two more pdfLaTeX passes. The result is `paper/Main.pdf`; auxiliary files
stay in a printed temporary directory. Upload this directory as an Overleaf
project, choose `Main.tex` as the main document, and use pdfLaTeX.
The generated tables and figure are included, so Overleaf needs no Python.

To refresh them from the saved experiment data, run
`python paper/make_results.py` in the repository environment with pandas,
NumPy, and Matplotlib. This does not rerun the experiments.
`generated/data_sha256.json` records the exact input files.

For presentations, `figures/transfer_depth_comparison.png` places depth one
and depth two side by side, with shared scales for each metric. PDF and SVG
versions are also included. Colors identify the same six test networks in
every panel, and black diamonds show their means. Separate figures are in
`figures/transfer_depth_one.*` and `figures/transfer_depth_two.*`; the paper
uses the depth-two PDF.

The paper uses the unmodified `siamart171218.cls` and `siamplain.bst` from
the neighboring `Quantum-Preconditioning-For-Constrained-Optimization-Problems`
Overleaf project. `Shared.tex` follows its title, author, and running-header
layout. Its manuscript text and author comments were not copied.
The SIAM companion distribution is retained under `siam-distribution/`.
`siam_compat.tex` restores the standard label/counter hooks before hyperref
loads, so the 2017 class also works with current LaTeX packages.

## Content and limits

The draft contains the restricted-arc QUBO, proofs of the cut penalty and
the strict budget-multiplier interval, exact contraction of fixed regions,
heuristic budget allocation, QAOA normalization, nearest-donor/GAT transfer,
and the regularized warm start. It reports the completed 60-row statevector
comparison and the paired subset of the 99-row MPS experiment.

The decomposition benchmark is still incomplete at the data cutoff, so the
paper includes its protocol and labels the results as pending. It does not
combine the archived partial run with the fresh eight-thread Gurobi run.
Partitioning and contraction remain serial NetworkX operations. It does not
infer hardware performance from simulator results. The statevector test set
has budget one, where greedy is exact. The MPS cohort is also solved by
greedy, and its bond cap has not received a convergence study.

Before a submission, finish the decomposition comparison and add harder
instances, multiple splits/seeds, warm-product-state ablations, and complete
runtime accounting. The bibliography is a starting set of primary sources,
not a completed systematic literature review.

## Writing style

The prose follows the requested approximate ASD-STE100 style: short sentences,
active verbs, one main point per paragraph, and consistent technical terms.
Mathematical terminology and necessary qualifications are retained. This is
an adaptation for a research article, not a claim of formal STE compliance.
The [official STE overview](https://www.asd-ste100.org/about.html) explains
the distinction between general vocabulary and project-specific technical terms.

## Reproducing the experiments

Use a fresh output path because the transfer runner rejects existing files:

```sh
OMP_NUM_THREADS=1 python run_parameter_transfer.py --out /tmp/transfer-rerun.csv
python run_qaoa_mid.py --reps 1,2,3 --angles donor,train --out /tmp/mid-rerun.csv
```

The transfer run needs `requirements-transfer.txt` as well as the core and
QAOA requirements. Full experiment parameters and dependency versions are
in `results/parameter_transfer.json`. The mid-size source settings are in
the header of `results/qaoa_mid.csv`. Runtime comparisons require a controlled
machine; the recorded component times came from a shared machine.
