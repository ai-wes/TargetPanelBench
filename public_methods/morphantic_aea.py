"""
Morphantic AEA adapter for TargetPanelBench
==========================================

Uses the Morphantic AdvancedArchipelagoEvolution (AEA) engine to optimize a set
of feature weights for target ranking. The objective maximizes nDCG@k by
searching weights in [0,1]^d for the numeric feature columns in targets.csv.

Returns a standard results dict with ranking, panel, and metrics.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Import evaluation helpers from this repo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluation import (
    compute_ranking_metrics,
    panel_recall,
    panel_diversity_score,
    build_adjacency,
)
from panel_design import select_diverse_panel

# Wire Morphantic engine modules
_MORPHANTIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "morphantic_engine_server"))
if _MORPHANTIC_DIR not in sys.path:
    sys.path.insert(0, _MORPHANTIC_DIR)

from complete_teai_methods_slim_v2 import AdvancedArchipelagoEvolution  # type: ignore


def _normalise_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    df_norm = df.copy()
    for col in feature_cols:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
            df_norm[col] = 0.0
        else:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    return df_norm


def run_morphantic_aea(
    targets_path: str,
    edges_path: str,
    panel_size: int = 12,
    k: int = 20,
    seed: int = 42,
    pop_size: int = 32,
    max_generations: int = 40,
    n_islands: int = 2,
) -> Dict:
    """Run Morphantic AEA to optimize feature weights for ranking.

    Minimizes loss = 1 - nDCG@k so that maximizing nDCG is equivalent.
    """
    # Load data and identify numeric features
    targets_df = pd.read_csv(targets_path)
    numeric_cols = [c for c in targets_df.columns if c not in {"gene_id", "gene_name", "ground_truth"}]
    if not numeric_cols:
        raise RuntimeError("No numeric feature columns found in targets.csv")
    norm_df = _normalise_features(targets_df, numeric_cols)

    # Precompute structures for speed
    gene_ids = norm_df["gene_id"].tolist()
    feat_mat = norm_df[numeric_cols].to_numpy(dtype=float)
    ground_truth = set(targets_df.loc[targets_df["ground_truth"] == 1, "gene_id"])

    # Fitness: given weights w in [0,1]^d, rank by score = X @ w, then loss = 1 - nDCG@k
    def fitness_func(w: np.ndarray) -> float:
        try:
            w = np.asarray(w, dtype=float)
            if w.ndim != 1 or w.size != feat_mat.shape[1]:
                return 1.0
            # Normalize weights to sum to 1 to avoid trivial scaling
            s = float(np.sum(w))
            if s <= 1e-12:
                w_eff = np.ones_like(w) / max(1, w.size)
            else:
                w_eff = w / s
            scores = feat_mat @ w_eff
            order = np.argsort(-scores)
            ranking = [gene_ids[i] for i in order]
            # Compute nDCG@k directly
            rel = np.array([1 if g in ground_truth else 0 for g in ranking], dtype=float)
            # DCG
            kk = min(k, rel.size)
            denom = np.log2(np.arange(2, 2 + kk))
            dcg = float(np.sum(rel[:kk] / denom))
            ideal = np.sort(rel)[::-1]
            idcg = float(np.sum(ideal[:kk] / denom))
            ndcg = 0.0 if idcg <= 0.0 else (dcg / idcg)
            return float(1.0 - ndcg)  # minimize
        except Exception:
            return 1.0

    # Configure AEA
    aea = AdvancedArchipelagoEvolution(
        dimension=len(numeric_cols),
        bounds=(0.0, 1.0),
        pop_size=int(pop_size),
        max_generations=int(max_generations),
        n_islands=int(n_islands),
        seed=int(seed),
    )

    # Run optimization
    result, best_cell = aea.optimize(fitness_func=fitness_func)
    best_w = None
    if best_cell is not None:
        bw = best_cell.get_solution()
        if isinstance(bw, np.ndarray):
            best_w = bw.tolist()
        else:
            try:
                best_w = list(bw)
            except Exception:
                best_w = None

    # Build final ranking using best weights (fallback to equal weights)
    if best_w is None:
        best_w = [1.0 / len(numeric_cols)] * len(numeric_cols)
    w = np.asarray(best_w, dtype=float)
    s = float(np.sum(w))
    w_eff = w / s if s > 1e-12 else np.ones_like(w) / max(1, w.size)
    final_scores = feat_mat @ w_eff
    order = np.argsort(-final_scores)
    ranking = [gene_ids[i] for i in order]

    # Metrics
    metrics = compute_ranking_metrics(targets_df, ranking, k=k)

    # Panel selection + metrics
    edges_df = pd.read_csv(edges_path)
    adjacency = build_adjacency(edges_df)
    panel = select_diverse_panel(ranking, adjacency, panel_size)
    prec_panel = panel_recall(panel, targets_df)
    div_panel = panel_diversity_score(panel, adjacency)

    return {
        "weights": best_w,
        "ranking": ranking,
        "panel": panel,
        "metrics": metrics,
        "panel_recall": prec_panel,
        "panel_diversity": div_panel,
        "aea": {
            "pop_size": int(pop_size),
            "max_generations": int(max_generations),
            "n_islands": int(n_islands),
            "seed": int(seed),
            "final_fitness": getattr(result, "final_fitness", None),
        },
        "feature_columns": numeric_cols,
    }
