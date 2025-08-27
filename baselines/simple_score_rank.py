"""
simple_score_rank.py
====================

This script implements a naïve baseline for the target prioritisation task.  It
computes a simple sum of all numerical feature columns (normalised to the
range 0–1) and ranks genes by this aggregated score.  Although crude, this
method serves as a useful reference point against which more sophisticated
approaches can be compared.

The script reads the candidate data from ``data/targets.csv`` (or another
directory specified via the ``--data-dir`` argument), computes the ranking,
selects a final panel and evaluates the performance using standard metrics.

Example usage:

    python baselines/simple_score_rank.py --data-dir data --results-path results/simple_results.json

The results are printed to stdout and saved as a JSON file for later analysis.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np
import pandas as pd

import os
import sys

# Adjust Python path so that we can import the project modules when running as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation import (
    compute_ranking_metrics,
    panel_recall,
    panel_diversity_score,
    build_adjacency,
)
from panel_design import select_top_k, select_diverse_panel


def normalise_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Normalise specified columns to the [0, 1] range.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with numeric columns.
    feature_cols : List[str]
        Columns to normalise.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with normalised columns.
    """
    df_norm = df.copy()
    for col in feature_cols:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val == min_val:
            # Avoid division by zero; set to zero
            df_norm[col] = 0.0
        else:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    return df_norm


def run_simple_baseline(targets_path: str, edges_path: str, panel_size: int = 12, k: int = 20) -> dict:
    """Execute the simple sum-of-features baseline and return results.

    Parameters
    ----------
    targets_path : str
        Path to ``targets.csv``.
    edges_path : str
        Path to ``ppi_edges.csv``.  Required for panel diversity.
    panel_size : int, optional
        Number of genes to include in the final panel (default 12).
    k : int, optional
        Cut‑off for ranking metrics (default 20).

    Returns
    -------
    dict
        Dictionary containing the ranking, panel and evaluation metrics.
    """
    # Load data
    targets_df = pd.read_csv(targets_path)
    # Determine numeric feature columns automatically (exclude identifiers and labels)
    numeric_cols = [col for col in targets_df.columns if col not in {"gene_id", "gene_name", "ground_truth"}]
    # Normalise features
    targets_norm = normalise_features(targets_df, numeric_cols)
    # Compute simple score as the sum of all normalised features
    targets_norm["simple_score"] = targets_norm[numeric_cols].sum(axis=1)
    # Rank genes by descending simple score
    ranked_df = targets_norm.sort_values("simple_score", ascending=False).reset_index(drop=True)
    ranking = ranked_df["gene_id"].tolist()
    # Compute ranking metrics
    metrics = compute_ranking_metrics(targets_df, ranking, k=k)
    # Select final panel (we use the top_k baseline for this simple method)
    panel = select_top_k(ranking, panel_size)
    # Compute panel recall and diversity
    edges_df = pd.read_csv(edges_path)
    adjacency = build_adjacency(edges_df)
    panel_recall_val = panel_recall(panel, targets_df)
    panel_diversity_val = panel_diversity_score(panel, adjacency)
    # Build results dict
    result = {
        "ranking": ranking,
        "panel": panel,
        "metrics": metrics,
        "panel_recall": panel_recall_val,
        "panel_diversity": panel_diversity_val,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the simple sum-of-features baseline")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing targets.csv and ppi_edges.csv")
    parser.add_argument("--results-path", type=str, default="results/simple_results.json", help="Output JSON file to save results")
    parser.add_argument("--panel-size", type=int, default=12, help="Number of genes in the final panel")
    parser.add_argument("--k", type=int, default=20, help="Cut‑off for ranking metrics (precision@k and nDCG@k)")
    args = parser.parse_args()
    targets_path = os.path.join(args.data_dir, "targets.csv")
    edges_path = os.path.join(args.data_dir, "ppi_edges.csv")
    result = run_simple_baseline(targets_path, edges_path, panel_size=args.panel_size, k=args.k)
    # Ensure results directory exists
    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    with open(args.results_path, "w") as f:
        json.dump(result, f, indent=2)
    # Print summary metrics
    print(json.dumps(result["metrics"], indent=2))
    print({"panel_recall": result["panel_recall"], "panel_diversity": result["panel_diversity"]})


if __name__ == "__main__":
    main()