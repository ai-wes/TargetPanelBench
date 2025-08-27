"""
download_data.py
=================

This script generates a reproducible synthetic dataset for the TargetPanelBench
benchmark.  It produces two files in the ``data/`` folder:

* ``targets.csv`` containing candidate genes with randomly generated feature
  values and ground truth labels.
* ``ppi_edges.csv`` describing a random protein–protein interaction (PPI)
  network between the genes.

The synthetic data is designed to emulate the structure of real drug discovery
problems without relying on proprietary or external data sources.  You can
extend this script to fetch real data from OpenTargets, GTEx, STRING or other
public APIs as needed.

Usage
-----

Run the script from the root of the repository:

    python download_data.py

You can adjust the number of genes, the number of ground truth positives and
the network density using command‑line arguments.  See ``--help`` for more
details.
"""

import argparse
import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd



def generate_candidates(num_genes: int, num_positives: int, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic candidate gene table.

    Parameters
    ----------
    num_genes : int
        Total number of candidate genes to create.
    num_positives : int
        Number of genes to label as ground truth positives.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: gene_id, gene_name, ground_truth,
        genetic_association, gene_expression, druggability,
        literature_velocity.
    """
    rng = np.random.default_rng(seed)
    # Create gene identifiers and names
    gene_ids = [f"GENE{i:04d}" for i in range(num_genes)]
    gene_names = [f"Gene{i:04d}" for i in range(num_genes)]

    # Choose ground truth positives
    positives = set(rng.choice(gene_ids, size=num_positives, replace=False))
    ground_truth = [1 if gid in positives else 0 for gid in gene_ids]

    # Generate synthetic features uniformly between 0 and 1
    genetic_association = rng.random(num_genes)
    gene_expression = rng.random(num_genes)
    druggability = rng.random(num_genes)
    literature_velocity = rng.random(num_genes)

    df = pd.DataFrame({
        "gene_id": gene_ids,
        "gene_name": gene_names,
        "ground_truth": ground_truth,
        "genetic_association": genetic_association,
        "gene_expression": gene_expression,
        "druggability": druggability,
        "literature_velocity": literature_velocity,
    })

    return df


def generate_ppi_network(genes: List[str], average_degree: int, seed: int = 42) -> pd.DataFrame:
    """Generate a random protein–protein interaction network without networkx.

    The network is constructed by randomly connecting pairs of genes with a
    probability chosen to achieve the desired average degree.  After initial
    edge sampling, disconnected components are linked together to ensure the
    resulting graph is connected.

    Parameters
    ----------
    genes : List[str]
        List of gene identifiers.
    average_degree : int
        Approximate average degree (number of neighbours) of each node.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Edge list with columns: ``gene1``, ``gene2``, ``weight`` (always 1.0).
    """
    rng = np.random.default_rng(seed)
    n = len(genes)
    # Probability of edge creation to achieve desired average degree: p = avg_deg / (n - 1)
    p = min(1.0, average_degree / max(1, n - 1))
    # Sample edges
    edges: List[Tuple[str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((genes[i], genes[j]))
    # Ensure connectivity: connect components with a simple union‑find structure
    parent = {gene: gene for gene in genes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # Union existing edges
    for u, v in edges:
        union(u, v)
    # Connect remaining components in a chain
    # Map each root to its members
    components = {}
    for gene in genes:
        root = find(gene)
        components.setdefault(root, []).append(gene)
    component_roots = list(components.keys())
    for i in range(len(component_roots) - 1):
        # Connect a representative of component i to a representative of component i+1
        u = components[component_roots[i]][0]
        v = components[component_roots[i + 1]][0]
        edges.append((u, v))
        union(u, v)
    # Build DataFrame
    edge_records = [{"gene1": u, "gene2": v, "weight": 1.0} for u, v in edges]
    edge_df = pd.DataFrame(edge_records)
    return edge_df


def save_dataset(targets: pd.DataFrame, edges: pd.DataFrame, data_dir: str) -> None:
    """Write the generated dataset to disk.

    Parameters
    ----------
    targets : pd.DataFrame
        Candidate gene table.
    edges : pd.DataFrame
        Edge list describing the PPI network.
    data_dir : str
        Directory in which to store the CSV files.
    """
    os.makedirs(data_dir, exist_ok=True)
    targets.to_csv(os.path.join(data_dir, "targets.csv"), index=False)
    edges.to_csv(os.path.join(data_dir, "ppi_edges.csv"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for TargetPanelBench")
    parser.add_argument("--num-genes", type=int, default=100, help="Total number of candidate genes")
    parser.add_argument("--num-positives", type=int, default=20, help="Number of ground truth positive genes")
    parser.add_argument("--average-degree", type=int, default=4, help="Average degree of the PPI network")
    parser.add_argument("--data-dir", type=str, default="data", help="Output directory for data files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Generate candidates and interactions
    targets = generate_candidates(args.num_genes, args.num_positives, seed=args.seed)
    edges = generate_ppi_network(list(targets["gene_id"]), average_degree=args.average_degree, seed=args.seed)

    save_dataset(targets, edges, args.data_dir)
    print(f"Generated {args.num_genes} candidate genes with {args.num_positives} positives.")
    print(f"Saved targets.csv and ppi_edges.csv to {args.data_dir}.")


if __name__ == "__main__":
    main()