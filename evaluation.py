"""
evaluation.py
==============

This module implements standard metrics for ranking and panel design tasks.  It is
shared by the baseline scripts and can be used by third‑party algorithms to
evaluate their performance on the TargetPanelBench dataset.

The following metrics are provided:

* Precision@k – proportion of the top‑`k` ranked items that are in the ground truth.
* Mean Reciprocal Rank (MRR) – mean of the reciprocal ranks of the first relevant
  item.
* Normalised Discounted Cumulative Gain (nDCG@k) – measures how well ranked
  relevant items are distributed towards the top of the list.
* Panel Recall – proportion of ground truth items contained in the selected panel.
* Panel Diversity Score – average shortest path distance between all pairs of
  genes in the panel, computed on an undirected graph.

All functions are designed to operate on simple Python structures such as lists
and dictionaries and make minimal assumptions about the input data.  See the
docstrings for details.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


def precision_at_k(ranking: Sequence[str], ground_truth_set: set[str], k: int) -> float:
    """Compute Precision@k for a ranking.

    Parameters
    ----------
    ranking : Sequence[str]
        Ordered list of gene identifiers (best to worst).
    ground_truth_set : set[str]
        Set of identifiers representing true positives.
    k : int
        Number of top items to consider.

    Returns
    -------
    float
        Proportion of the top‑`k` ranked items that are in the ground truth.
    """
    if k <= 0:
        return 0.0
    top_k = ranking[:k]
    hits = sum(1 for item in top_k if item in ground_truth_set)
    return hits / k


def mean_reciprocal_rank(ranking: Sequence[str], ground_truth_set: set[str]) -> float:
    """Compute the mean reciprocal rank (MRR) of the ranking.

    MRR is defined as the average of the reciprocal of the rank position of the
    first relevant item.  If there are no relevant items in the ranking, the
    reciprocal rank is taken as zero.

    Parameters
    ----------
    ranking : Sequence[str]
        Ordered list of gene identifiers.
    ground_truth_set : set[str]
        Set of ground truth positives.

    Returns
    -------
    float
        Mean reciprocal rank (a single value because we evaluate one ranking).
    """
    for idx, item in enumerate(ranking, start=1):
        if item in ground_truth_set:
            return 1.0 / idx
    return 0.0


def dcg_at_k(relevances: Sequence[int], k: int) -> float:
    """Compute the Discounted Cumulative Gain at `k`.

    Parameters
    ----------
    relevances : Sequence[int]
        Binary sequence where 1 indicates a relevant item and 0 indicates
        non‑relevant.
    k : int
        Number of top items to consider.

    Returns
    -------
    float
        Discounted Cumulative Gain.
    """
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        rel = relevances[i]
        # Positions are 1‑indexed in DCG formula
        denom = math.log2(i + 2)
        dcg += rel / denom
    return dcg


def ndcg_at_k(ranking: Sequence[str], ground_truth_set: set[str], k: int) -> float:
    """Compute the normalised DCG at `k` for a binary relevance ranking.

    Parameters
    ----------
    ranking : Sequence[str]
        Ordered list of gene identifiers.
    ground_truth_set : set[str]
        Set of ground truth positives.
    k : int
        Number of top items to consider.

    Returns
    -------
    float
        Normalised Discounted Cumulative Gain (0–1).
    """
    relevances = [1 if item in ground_truth_set else 0 for item in ranking]
    dcg = dcg_at_k(relevances, k)
    # Compute ideal DCG – all relevant items ranked at the top
    ideal_relevances = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal_relevances, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def panel_recall(panel: Sequence[str], targets_df: pd.DataFrame) -> float:
    """Compute the recall of ground truth targets within the selected panel.

    Parameters
    ----------
    panel : Sequence[str]
        List of gene identifiers in the final selected panel.
    targets_df : pd.DataFrame
        DataFrame containing the `ground_truth` column with binary labels.

    Returns
    -------
    float
        Fraction of ground truth positives captured in the panel.
    """
    ground_truth_set = set(targets_df.loc[targets_df["ground_truth"] == 1, "gene_id"])
    hits = sum(1 for item in panel if item in ground_truth_set)
    if not ground_truth_set:
        return 0.0
    return hits / len(ground_truth_set)


def build_adjacency(edges_df: pd.DataFrame) -> dict[str, set[str]]:
    """Convert an edge list DataFrame into an adjacency dictionary.

    Parameters
    ----------
    edges_df : pd.DataFrame
        DataFrame with columns ``gene1`` and ``gene2`` describing undirected edges.

    Returns
    -------
    dict[str, set[str]]
        Mapping from each gene identifier to a set of its neighbours.
    """
    adjacency: dict[str, set[str]] = {}
    for row in edges_df.itertuples(index=False):
        u, v = row.gene1, row.gene2
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)
    return adjacency


def _shortest_path_length(adj: dict[str, set[str]], source: str, target: str) -> int:
    """Compute shortest path length between two nodes in an unweighted graph.

    Uses a breadth‑first search.  Returns ``math.inf`` if the nodes are not
    connected.
    """
    if source == target:
        return 0
    from collections import deque
    visited = {source}
    queue = deque([(source, 0)])
    while queue:
        node, dist = queue.popleft()
        for neighbour in adj.get(node, []):
            if neighbour == target:
                return dist + 1
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, dist + 1))
    # Disconnected
    return float('inf')


def panel_diversity_score(panel: Sequence[str], adjacency: dict[str, set[str]]) -> float:
    """Compute the diversity score of a selected target panel without networkx.

    The diversity score is the average shortest path length between all pairs
    of genes in the panel, using the adjacency list to compute distances.  If
    two genes are disconnected, a large penalty equal to twice the number of
    nodes in the graph is used.

    Parameters
    ----------
    panel : Sequence[str]
        List of gene identifiers in the final selected panel.
    adjacency : dict[str, set[str]]
        Mapping from gene identifiers to their neighbours.

    Returns
    -------
    float
        Average shortest path length between all distinct pairs in the panel.
        Returns 0.0 if there is only one or zero items in the panel.
    """
    n_nodes = len(adjacency)
    if len(panel) < 2:
        return 0.0
    lengths = []
    for i in range(len(panel)):
        for j in range(i + 1, len(panel)):
            u = panel[i]
            v = panel[j]
            dist = _shortest_path_length(adjacency, u, v)
            if dist == float('inf'):
                lengths.append(2 * n_nodes)
            else:
                lengths.append(dist)
    return float(np.mean(lengths))


def compute_ranking_metrics(targets_df: pd.DataFrame, ranking: Sequence[str], k: int = 20) -> dict:
    """Compute a dictionary of ranking metrics for a given ranking.

    Parameters
    ----------
    targets_df : pd.DataFrame
        DataFrame containing the `ground_truth` column and `gene_id` identifiers.
    ranking : Sequence[str]
        Ordered list of gene identifiers.
    k : int, optional
        Cut‑off for Precision@k and nDCG@k metrics (default is 20).

    Returns
    -------
    dict
        Dictionary with keys: ``precision_at_k``, ``mrr`` and ``ndcg_at_k``.
    """
    ground_truth_set = set(targets_df.loc[targets_df["ground_truth"] == 1, "gene_id"])
    return {
        "precision_at_%d" % k: precision_at_k(ranking, ground_truth_set, k),
        "mrr": mean_reciprocal_rank(ranking, ground_truth_set),
        "ndcg_at_%d" % k: ndcg_at_k(ranking, ground_truth_set, k),
    }