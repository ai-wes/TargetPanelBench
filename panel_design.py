"""
panel_design.py
===============

This module provides helper functions to construct a target panel from a ranked
list of candidates.  The goal is to produce a final set of targets that not
only covers the most promising genes but also maximises biological diversity.

Two key functions are provided:

* ``select_diverse_panel`` – greedily selects a panel of a specified size,
  balancing ranking score and network diversity.
* ``select_top_k`` – simple baseline that selects the top `k` items from the
  ranking without considering diversity.

The greedy approach uses a straightforward algorithm: start with the highest
ranked gene and iteratively add the gene that maximises the minimum network
distance to the already selected panel.  This encourages selecting genes that
are farther apart on the protein–protein interaction network.
"""

from __future__ import annotations

from typing import List, Sequence

import math
import numpy as np

def _shortest_path_length(adj: dict[str, set[str]], source: str, target: str) -> int:
    """Compute shortest path length between two nodes using BFS.

    Returns ``float('inf')`` if the nodes are disconnected.
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
    return float('inf')


def select_top_k(ranking: Sequence[str], panel_size: int) -> List[str]:
    """Select the top `panel_size` genes from the ranking.

    Parameters
    ----------
    ranking : Sequence[str]
        Ordered list of gene identifiers.
    panel_size : int
        Desired number of genes in the final panel.

    Returns
    -------
    List[str]
        Subset of the ranking containing the first ``panel_size`` genes.
    """
    return list(ranking[:panel_size])


def select_diverse_panel(ranking: Sequence[str], graph: nx.Graph, panel_size: int) -> List[str]:
    """Greedy selection of a diverse target panel from a ranked list.

    The algorithm works as follows:

    1. Start with the top‑ranked gene.
    2. While the panel has fewer than ``panel_size`` genes, consider the next
       candidates in the ranking.  For each candidate, compute its minimum
       shortest path distance to the genes already in the panel.
    3. Select the candidate that maximises this minimum distance.  If multiple
       candidates tie, pick the one that appears earlier in the ranking.

    If some genes are disconnected in the network (no path between them), the
    distance is treated as a large value so that disconnected genes are
    preferred.

    Parameters
    ----------
    ranking : Sequence[str]
        Ordered list of gene identifiers (best to worst).
    graph : nx.Graph
        Protein–protein interaction network with all candidate genes as nodes.
    panel_size : int
        Desired number of genes in the final panel.

    Returns
    -------
    List[str]
        Selected panel of genes.
    """
    if panel_size <= 0:
        return []
    # ``graph`` is an adjacency dictionary: {gene: set(neighbours)}
    adjacency = graph
    # Start with the top‑ranked gene
    panel = [ranking[0]]
    selected_set = {ranking[0]}
    index_map = {gene: idx for idx, gene in enumerate(ranking)}
    n_nodes = len(adjacency)
    # Greedy selection
    while len(panel) < panel_size and len(selected_set) < len(ranking):
        best_candidate = None
        best_distance = -1.0
        for candidate in ranking:
            if candidate in selected_set:
                continue
            # Compute minimum distance to current panel
            min_dist = float('inf')
            for selected in panel:
                dist = _shortest_path_length(adjacency, candidate, selected)
                if dist == float('inf'):
                    # Use large penalty for disconnected nodes
                    dist = 2 * n_nodes
                if dist < min_dist:
                    min_dist = dist
            # Choose candidate maximising this minimum distance
            if (min_dist > best_distance) or (
                math.isclose(min_dist, best_distance) and index_map[candidate] < index_map.get(best_candidate, float('inf'))
            ):
                best_distance = min_dist
                best_candidate = candidate
        if best_candidate is None:
            break
        panel.append(best_candidate)
        selected_set.add(best_candidate)
    return panel[:panel_size]