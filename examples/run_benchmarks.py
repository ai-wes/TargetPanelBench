"""Run TargetPanelBench baselines on a freshly generated dataset.

This example script performs a full end‑to‑end run of the benchmark:

1. Generate a small synthetic dataset in ``data/example``.
2. Execute both baseline ranking methods.
3. Save the results under ``results/example`` and print a summary of metrics.

The script is intended as a lightweight example showing how the library
components fit together.  It can be used as a starting point for integrating
new methods or experimenting with different parameters.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import sys

# Allow the script to be executed from the repository root without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from download_data import generate_candidates, generate_ppi_network, save_dataset
from baselines.simple_score_rank import run_simple_baseline
from baselines.cma_es import run_cma_es_baseline


def main() -> None:
    # Directories for the example run
    data_dir = Path("data/example")
    results_dir = Path("results/example")
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate a small synthetic dataset
    targets = generate_candidates(num_genes=50, num_positives=10, seed=0)
    edges = generate_ppi_network(list(targets["gene_id"]), average_degree=3, seed=0)
    save_dataset(targets, edges, str(data_dir))

    targets_path = data_dir / "targets.csv"
    edges_path = data_dir / "ppi_edges.csv"

    # 2. Run the simple score baseline
    simple_result = run_simple_baseline(str(targets_path), str(edges_path))
    with open(results_dir / "simple.json", "w") as f:
        json.dump(simple_result, f, indent=2)

    # 3. Run the CMA‑ES baseline with fewer iterations for speed
    cma_result = run_cma_es_baseline(
        str(targets_path), str(edges_path), iterations=50, seed=0
    )
    with open(results_dir / "cma_es.json", "w") as f:
        json.dump(cma_result, f, indent=2)

    # Print a concise summary of the metrics
    print("Simple baseline metrics:")
    print(json.dumps(simple_result["metrics"], indent=2))
    print({"panel_recall": simple_result["panel_recall"], "panel_diversity": simple_result["panel_diversity"]})

    print("\nCMA‑ES baseline metrics:")
    print(json.dumps(cma_result["metrics"], indent=2))
    print({"panel_recall": cma_result["panel_recall"], "panel_diversity": cma_result["panel_diversity"]})


if __name__ == "__main__":
    main()
