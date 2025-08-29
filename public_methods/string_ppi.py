"""
STRING PPI integration (local file)

Loads STRING edges from a TSV file and recomputes panel diversity for results.
Expected columns: preferredName_A, preferredName_B (or protein1/protein2 with mapped symbols).
"""
from __future__ import annotations

from typing import Dict
import os
import json
import pandas as pd

from evaluation import build_adjacency, panel_diversity_score


def load_string_edges(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # Auto-detect separator
    sep = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep)
    # Derive gene columns
    if {"preferredName_A", "preferredName_B"}.issubset(df.columns):
        df2 = pd.DataFrame({
            "gene1": df["preferredName_A"].astype(str),
            "gene2": df["preferredName_B"].astype(str),
        })
    elif {"protein1", "protein2"}.issubset(df.columns):
        # Hope they are symbols; otherwise expect pre-mapped input
        df2 = pd.DataFrame({
            "gene1": df["protein1"].astype(str),
            "gene2": df["protein2"].astype(str),
        })
    elif {"gene1", "gene2"}.issubset(df.columns):
        df2 = pd.DataFrame({
            "gene1": df["gene1"].astype(str),
            "gene2": df["gene2"].astype(str),
        })
    else:
        raise ValueError("STRING file must contain preferredName_A/B or protein1/protein2 (or gene1/gene2) columns")
    # Drop self-loops
    df2 = df2[df2["gene1"] != df2["gene2"]].dropna()
    return df2


def recompute_panel_diversity_for_results(results_path: str, string_edges_path: str) -> Dict:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    panel = data.get("panel") or []
    edges = load_string_edges(string_edges_path)
    adj = build_adjacency(edges)
    div = panel_diversity_score(panel, adj)
    data["panel_diversity_string"] = float(div)
    base, ext = os.path.splitext(results_path)
    out_path = base + "_string.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return {"out": out_path, "panel_diversity_string": float(div)}
