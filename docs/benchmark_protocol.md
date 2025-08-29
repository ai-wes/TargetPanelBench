Benchmark Protocol
==================

This document describes the evaluation protocol used in TargetPanelBench, and
how public tools are integrated for reproducible comparisons.

Tasks
-----
- Target prioritisation (ranking): Given a candidate set of genes for a disease,
  produce an ordered list that ranks true disease genes near the top.
- Panel design (selection): Select a fixed-size set of targets that maximises
  both recall of known positives and biological diversity.

Metrics
-------
- Precision@k: Fraction of top-k ranked genes that are in the ground truth.
- Mean Reciprocal Rank (MRR): Reciprocal rank of the first relevant item.
- nDCG@k: Normalised DCG at k for binary relevance.
- Panel Recall: Fraction of positives covered by the selected panel.
- Panel Diversity: Average shortest-path distance among all panel pairs on a PPI graph.

Datasets
--------
Two dataset modes are supported:

1) Synthetic (default)
- Generated via `download_data.py` with reproducible RNG.
- Provides candidate table with synthetic features and a random PPI graph.
- Useful for quick smoke tests and method development.

2) Public tool-derived (Open Targets adapter)
- The `public_methods/opentargets.py` module fetches disease–gene associations
  from Open Targets GraphQL API.
- For demonstration, candidates are the top-N genes by association score; a
  self-contained ground truth is defined as the top-M genes by the same source.
- Limitations: Truth derived from the same source introduces circularity. For
  formal benchmarking, replace ground truth with independent curation (e.g.,
  ClinGen, PanelApp, literature).

Reproducibility
---------------
- All code paths are deterministic given seeds.
- Public tool calls are logged via JSON results in `results/` with method and
  parameter metadata.
- The `benchmark.py summarise` command collates all JSON metrics into a single CSV.

How to Extend
-------------
- Add new adapters under `public_methods/` following the pattern in
  `opentargets.py` and implement a `run_*` function that returns a dictionary
  with `ranking`, `metrics`, and metadata.
- Contribute alternative panel diversity graphs (e.g., STRING-derived) or
  additional metrics.
