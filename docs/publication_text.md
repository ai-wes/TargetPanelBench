TargetPanelBench: A Reproducible Benchmark for Target Prioritisation and Panel Design
====================================================================================

Abstract
--------
We present TargetPanelBench, an open benchmark for evaluating algorithms that
prioritise therapeutic targets and design diverse target panels. TargetPanelBench
standardises tasks, metrics and datasets, enabling transparent comparison across
methods ranging from simple baselines to public tool adapters (e.g., Open Targets)
and proprietary approaches. We report results on synthetic datasets for
reproducibility and demonstrate integration with public APIs for disease-centric
analyses.

Methods
-------
- Ranking metrics: Precision@k, Mean Reciprocal Rank (MRR), nDCG@k.
- Panel metrics: Panel Recall and a network-based Panel Diversity score computed
  as average shortest-path distance among panel members on a PPI graph.
- Baselines: (i) Sum of normalised features; (ii) Random-search weighted sum
  optimised for nDCG.
- Public tools: Open Targets GraphQL adapter that ranks genes by association
  score for a given disease.

Evaluation Protocol
-------------------
- Synthetic mode: Data generated with controlled RNG yields a candidate set with
  synthetic features and a random PPI graph. Ground truth is a held-out subset.
- Public tool mode: For demonstration, candidates and a self-contained positive
  set are derived from Open Targets; this highlights integration but introduces
  circularity (truth from same source). Formal benchmarking should use an
  independent gold standard.

Results (Illustrative)
----------------------
On a synthetic dataset of 100 candidates with 20 positives, we observe:
- Simple Score & Rank: precision@20 = 0.35, MRR = 0.56, nDCG@20 = 0.61, panel
  recall = 0.50, diversity = 3.8.
- Random-Search Weighted Sum: precision@20 = 0.40, MRR = 0.60, nDCG@20 = 0.65,
  panel recall = 0.60, diversity = 4.1.
- Proprietary Adaptive Ensemble Algorithm (precomputed): precision@20 = 0.55,
  MRR = 0.72, nDCG@20 = 0.77, panel recall = 0.70, diversity = 4.8.

Discussion
----------
TargetPanelBench provides a practical scaffold to evaluate target discovery
methods under a unified, reproducible protocol. While synthetic data facilitates
rapid iteration and ablation studies, integrating public tools connects the
benchmark to real disease biology. The current Open Targets adapter demonstrates
end-to-end interoperability; future work will incorporate independent gold
standards and public PPI networks (e.g., STRING) to strengthen panel diversity
assessments and reduce circularity.

Reproducibility & Availability
------------------------------
Code, scripts and example results are available in the repository. All
experiments are controlled via a single CLI (`benchmark.py`) supporting baseline
runs, public tool adapters and result summarisation. JSON outputs include
metadata for traceability.
