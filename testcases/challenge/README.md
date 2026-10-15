# Application networks and covering stressors

These are inputs to a **single-commodity directed maximum-flow proxy**, minimizing
remaining flow after removing at most B allowed directed arcs. They are not traffic
assignment, epidemic simulations, or AC/DC power-flow models. Opposite directions
are independent interdictions; a physical road, contact, or power line is not
removed as a coupled pair. All original IDs, selected terminals, allowed arcs,
budgets, and SHA-256 hashes are in `manifest.json`.

`python challenge_instances.py --out /tmp/rebuilt-cohort` regenerates the seven new
`.flow` files and manifest offline. Original synthetic/legacy files have the
repository's code/data terms. **Third-party sources and their derived flow files
retain the terms below; they are not relicensed by the repository's MIT license.**

## Road: Eastern Massachusetts highway subnetwork

`ema.flow`: 74 vertices, 258 directed arcs. Source/sink are original nodes **1/34**.
Original directed capacities are rounded to the nearest integer, half up; maximum
absolute rounding error is 0.5 original capacity units per arc. We do not infer a
physical capacity unit absent from the source's units table. The original TNTP
capacity scale is retained; travel times, lengths, OD demand, and costs are unused.

- Source: [Transportation Networks for Research Core Team, Eastern Massachusetts](https://github.com/bstabler/TransportationNetworks/tree/d1639b4ef218c17928ba573e806ddf8ba5e7ae6d/Eastern-Massachusetts),
  revision `d1639b4ef218c17928ba573e806ddf8ba5e7ae6d`, retrieved 2026-10-15.
- `sources/EMA_net.tntp` is the unchanged network input. Its README describes a
  highway subnetwork, derived from April 2012 PM traffic data via
  [InverseVIsTraffic](https://github.com/jingzbu/InverseVIsTraffic), and credits
  Jing Zhang, Sepideh Pourazarm, Christos G. Cassandras, and Ioannis Ch. Paschalidis.
- [Upstream terms](https://github.com/bstabler/TransportationNetworks/blob/d1639b4ef218c17928ba573e806ddf8ba5e7ae6d/README.md#license):
  donated data, **academic research purposes only**, cite the dataset source in
  publications, users responsible for results and conclusions, no responsibility
  accepted by the maintainers or contributing agencies for content/correctness.
  These terms also accompany the converted `ema.flow`.
- Related primary study: [Data-driven Estimation of Origin-Destination Demand and
  User Cost Functions for the Optimization of Transportation Networks](https://arxiv.org/abs/1610.09580).

Sioux Falls was considered but not selected: its own upstream README says it is
not a realistic network. This cohort uses a measured-data-derived highway network.

## Contact: SocioPatterns primary school, class 4A

`school4a.flow`: 21 anonymous participants, 418 directed arcs, original source/sink
**1503/1519**. Select every contact with both class fields `4A`, across both recorded
days. This is the smallest named class in the supplied data, selected for bounded
computation, not based on optimization outcomes. Other classes, teachers, and
inter-class contacts are excluded. Each recorded 20-second interval adds 20 seconds
to that unordered pair; emit both directions with the resulting total duration.
The static aggregation discards temporal ordering; flow has contact-second proxy
units and is not transmission probability, prevalence, or predicted infections.

- [SocioPatterns, Primary school temporal network data](https://sociopatterns.org/datasets/primary-school-temporal-network-data/),
  retrieved 2026-10-15; acknowledge the [SocioPatterns collaboration](https://sociopatterns.org/).
- Original [primaryschool.csv.gz](https://sociopatterns.org/assets/data/primaryschool.csv.gz):
  125773 rows, SHA-256 `5c93d9f5a61ad44b3c90fb5146d42345ae86dd44d77bc5da2461989d3d547fc9`.
  The retained `sources/primaryschool_4A.tsv.gz` is the 7265 selected original rows,
  gzip-compressed with timestamp zero. Decompressed SHA-256:
  `31d4c367d9ce7cf43f290c944e05568108081a9f458097adee731472d096d31c`.
- Source subset, derived `school4a.flow`, and contact-derived benchmark data are
  distributed under [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/):
  attribution, noncommercial use, share alike; changes are the selection,
  aggregation, bidirection expansion, and relabeling described here.
- Required citations: [Gemmetto et al., Mitigation of infectious disease at school:
  targeted class closure vs school closure (2014)](https://doi.org/10.1186/s12879-014-0695-9),
  and [Stehlé et al., High-Resolution Measurements of Face-to-Face Contact Patterns
  in a Primary School (2011)](https://doi.org/10.1371/journal.pone.0023176).

## Power: PGLib-OPF IEEE 14-bus test system

`ieee14.flow`: 14 buses, 40 directed arcs (20 physical branches), original
source/sink **2/13**. Active branches supply `rateA` as a positive integer MVA
capacity in each direction. These are **modeled thermal ratings**, not measured
line limits: the source records a statistical/current capacity model and missing
rating data. We ignore impedances, voltage, angle, generation and demand. A bus-pair
max-flow proxy does not model an actual grid operating condition or load shedding.

- [IEEE PES Power Grid Library, v23.07, case14](https://github.com/power-grid-lib/pglib-opf/blob/dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3/pglib_opf_case14_ieee.m),
  revision `dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3`, retrieved 2026-10-15.
- `sources/pglib_opf_case14_ieee.m` retains the complete original notices and
  transformation log. Copyright (c) 1999 Richard D. Christie, University of
  Washington; [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Those terms
  also apply to `ieee14.flow`; changes are branch extraction, direction expansion,
  terminal selection, and relabeling.
- Cite [The Power Grid Library for Benchmarking AC Optimal Power Flow Algorithms](https://arxiv.org/abs/1908.02788),
  plus the original case source above. No endorsement by the dataset creators.

For each real network, choose the lexicographically first ordered pair at maximum
unweighted hop distance among sources with at least three outgoing arcs and sinks
with at least three incoming arcs. This protocol was fixed before the retained
campaign; an exploratory pilot with unrestricted endpoints chose the power
network's degree-one bus 8 and was excluded. The minimum-degree rule avoids that
single-arc disconnection artifact. It does not guarantee a difficult optimum. Relabel sorted original IDs to
0..n-1. The criterion is deterministic, not a claim that these terminals model a
specific intervention. No case is assumed hard from its application label.

## Synthetic covering family

A fixed seeded undirected graph supplies an edge-to-endpoint incidence network.
For each edge `{u,v}`, add `s -> a -> b -> u,v` with capacity 2 on all four arcs;
for each original vertex, add capacity 1 to `t`. Only `a -> b` is removable.
Consequently **all removable arcs are internal by construction**, and the
`interdictable` manifest field is part of the instance, not optional metadata.
Loading the `.flow` alone and allowing all arcs solves a different problem.

Keeping r original edges leaves flow equal to the number of their distinct
endpoints: each covered vertex can receive one unit and each edge supplies at most
two units. Thus the interdiction problem selects r edges with minimum endpoint
union. A clique of order k retains `k*(k-1)/2` edges on k vertices. The implementation
is checked over all 64 interdiction states of a four-vertex complete graph.
This provides a transparent covering stressor; it does not claim replication of
Wood's experiments or hardness of each random draw.

The four retained draws use n=12/18/24/30, edge probability 0.5, Python `Random(0)`,
and retain 6/10/15/21 edges respectively. Increasing incidence density, symmetric
small capacities, and a budget that leaves a dense small union are the tested
structural knobs. The original graph edges are recorded, so they do not depend on
future random-generator behavior. Covering-based interdiction and weak LP bounds
are motivated by [Altner, Ergun and Uhan (2010)](https://doi.org/10.1016/j.orl.2009.09.013);
our flow identity above is the mechanism actually implemented and tested.

The manifest also profiles the existing 50-node planar, ER and k-NN networks at
budgets 3 and 6. These controls are retained even if a terminal-only optimum makes
them unsuitable as hard cases. Planarity is never treated as inherent hardness;
[Zenklusen (2010)](https://doi.org/10.1016/j.dam.2010.04.008) describes the
pseudo-polynomial planar-network setting and its limitations.
