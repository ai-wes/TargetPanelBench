Public Methods Integrated
=========================

This repository integrates public tools alongside local baselines.

Open Targets
------------
- Source: https://platform.opentargets.org/
- API: GraphQL v4 (`https://api.platform.opentargets.org/api/v4/graphql`)
- Adapter (API): `public_methods/opentargets.py`
- Adapter (Local Parquet): `public_methods/opentargets_local.py`
- Protocol (API): For a given disease EFO ID, fetch top-N associated targets and rank
  them by the overall association score. Construct a self-contained positive set
  as the top-M targets (see protocol note below).
- Protocol (Local): For a given Disease Ontology ID (`DOID_*`), load associations from
  `associations_direct_overall` and map `targetId` (ENSG) to `approvedSymbol` using
  `targets_core_by_annotation`. Rank by `score`.
- Caveat: Both paths can produce perfect metrics if the positive set is taken as
  top-M from the same source. Use an independent gold standard for publication‑grade analyses.

Baselines
---------
- Simple Score & Rank: `baselines/simple_score_rank.py` sums normalised feature
  columns and ranks by the aggregate.
- Random-Search Weighted Sum (CMA-ES style): `baselines/cma_es.py` samples random
  weight vectors for linear combinations of features and maximises a chosen
  objective metric (nDCG, precision@k or MRR).

Planned Adapters
----------------
- DISEASES (Jensen Lab) API for disease–gene associations.
- STRING for PPI edges to compute panel diversity on public networks.
- g:Profiler / Enrichr for functional context (for analysis, not ranking).

Usage
-----
- Run synthetic baselines: `python benchmark.py run-baselines --data-dir data`.
- Run Open Targets adapter: `python benchmark.py run-opentargets --efo-id EFO_0003767`.
- Summarise results: `python benchmark.py summarise --results-dir results`.
