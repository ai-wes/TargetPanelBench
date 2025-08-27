"""
cma_es.py
=========

This script implements an evolutionary baseline for the target prioritisation
problem.  Although in a production setting you might use a sophisticated
Continuous Mutation / Covariance Matrix Adaptation Evolution Strategy (CMA‑ES),
for the purposes of this benchmark we employ a simpler random search over
linear combinations of features.  This approach is parameterised by the
number of iterations to perform and the objective metric to optimise.

The algorithm works as follows:

1. Load the candidate data and identify numeric feature columns.
2. For each iteration:
   a. Sample a random weight vector from the uniform distribution on
      [0, 1]^d (where d is the number of features).
   b. Combine the features using the sampled weights to produce a score.
   c. Sort genes by descending score to obtain a ranking.
   d. Compute the chosen objective metric (nDCG by default).
   e. Retain the best weight vector and ranking if the metric improves.
3. Select a diverse panel from the best ranking.
4. Evaluate the final ranking and panel using standard metrics.

You can adjust the optimisation target via the ``--objective`` argument.  By
default this is ``ndcg``; other options are ``precision`` and ``mrr``.

Example usage:

    python baselines/cma_es.py --data-dir data --results-path results/cma_es_results.json --iterations 100 --objective ndcg
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
    precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    panel_recall,
    panel_diversity_score,
    build_adjacency,
)
from panel_design import select_diverse_panel


def run_cma_es_baseline(
    targets_path: str,
    edges_path: str,
    iterations: int = 100,
    objective: str = "ndcg",
    panel_size: int = 12,
    k: int = 20,
    seed: int = 42,
) -> dict:
    """Execute a simple evolutionary baseline and return results.

    Parameters
    ----------
    targets_path : str
        Path to ``targets.csv``.
    edges_path : str
        Path to ``ppi_edges.csv``.
    iterations : int, optional
        Number of random weight vectors to evaluate (default 100).
    objective : str, optional
        Metric to maximise – one of {``ndcg``, ``precision``, ``mrr``}.
    panel_size : int, optional
        Number of genes in the final panel (default 12).
    k : int, optional
        Cut‑off for ranking metrics (default 20).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing the ranking, panel and evaluation metrics.
    """
    rng = np.random.default_rng(seed)
    # Load data
    targets_df = pd.read_csv(targets_path)
    numeric_cols = [col for col in targets_df.columns if col not in {"gene_id", "gene_name", "ground_truth"}]
    # Normalise features to [0, 1]
    def normalise(df: pd.DataFrame) -> pd.DataFrame:
        df_norm = df.copy()
        for col in numeric_cols:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            df_norm[col] = 0.0 if max_val == min_val else (df_norm[col] - min_val) / (max_val - min_val)
        return df_norm
    norm_df = normalise(targets_df)
    ground_truth_set = set(targets_df.loc[targets_df["ground_truth"] == 1, "gene_id"])
    # Prepare objective function
    def evaluate_ranking(ranking: List[str]) -> float:
        if objective == "precision":
            return precision_at_k(ranking, ground_truth_set, k)
        elif objective == "mrr":
            return mean_reciprocal_rank(ranking, ground_truth_set)
        else:
            return ndcg_at_k(ranking, ground_truth_set, k)
    best_score = -np.inf
    best_weights = None
    best_ranking = None
    # Iterate over random weight vectors
    for _ in range(iterations):
        weights = rng.random(len(numeric_cols))
        # Combine features using the sampled weights
        scores = (norm_df[numeric_cols].values * weights).sum(axis=1)
        # Rank genes by descending score
        order = np.argsort(-scores)
        ranking = norm_df.iloc[order]["gene_id"].tolist()
        score = evaluate_ranking(ranking)
        if score > best_score:
            best_score = score
            best_weights = weights
            best_ranking = ranking
    # If no ranking found (should not happen), fall back to simple baseline
    if best_ranking is None:
        best_ranking = norm_df.sort_values(numeric_cols, ascending=False).index.tolist()
    # Select final panel using greedy diversity selection
    edges_df = pd.read_csv(edges_path)
    adjacency = build_adjacency(edges_df)
    panel = select_diverse_panel(best_ranking, adjacency, panel_size)
    # Compute metrics on the final ranking and panel
    metrics = compute_ranking_metrics(targets_df, best_ranking, k=k)
    panel_recall_val = panel_recall(panel, targets_df)
    panel_diversity_val = panel_diversity_score(panel, adjacency)
    result = {
        "weights": best_weights.tolist() if best_weights is not None else None,
        "ranking": best_ranking,
        "panel": panel,
        "metrics": metrics,
        "panel_recall": panel_recall_val,
        "panel_diversity": panel_diversity_val,
        "objective": objective,
        "best_score": best_score,
        "iterations": iterations,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple evolutionary baseline for target ranking")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing targets.csv and ppi_edges.csv")
    parser.add_argument("--results-path", type=str, default="results/cma_es_results.json", help="Output JSON file to save results")
    parser.add_argument("--iterations", type=int, default=100, help="Number of random weight vectors to evaluate")
    parser.add_argument("--objective", type=str, default="ndcg", choices=["ndcg", "precision", "mrr"], help="Objective metric to maximise")
    parser.add_argument("--panel-size", type=int, default=12, help="Number of genes in the final panel")
    parser.add_argument("--k", type=int, default=20, help="Cut‑off for ranking metrics (precision@k and nDCG@k)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    targets_path = os.path.join(args.data_dir, "targets.csv")
    edges_path = os.path.join(args.data_dir, "ppi_edges.csv")
    result = run_cma_es_baseline(
        targets_path,
        edges_path,
        iterations=args.iterations,
        objective=args.objective,
        panel_size=args.panel_size,
        k=args.k,
        seed=args.seed,
    )
    # Write to file
    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    with open(args.results_path, "w") as f:
        json.dump(result, f, indent=2)
    # Print summary
    print(json.dumps(result["metrics"], indent=2))
    print({"panel_recall": result["panel_recall"], "panel_diversity": result["panel_diversity"]})


if __name__ == "__main__":
    main()