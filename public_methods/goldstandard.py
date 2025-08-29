"""
Gold standard loaders and evaluation helpers.

Supports loading an independent set of positive genes and re-evaluating
benchmarks against that set to avoid circularity.
"""
from __future__ import annotations

from typing import List, Set, Dict
import os
import json
import pandas as pd


def load_gold_set(path: str) -> Set[str]:
    """Load a gold-standard set of gene identifiers (symbols) from CSV/TSV/TXT.

    Accepted formats:
    - TXT: one gene symbol per line
    - CSV/TSV: a column named 'symbol' or 'gene_id' (case-insensitive)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    lower = path.lower()
    if lower.endswith(".txt") or lower.endswith(".lst"):
        syms = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    syms.append(s)
        return set(syms)
    # CSV/TSV
    sep = "," if lower.endswith(".csv") else "\t"
    df = pd.read_csv(path, sep=sep)
    cols = {c.lower(): c for c in df.columns}
    if "symbol" in cols:
        return set(map(str, df[cols["symbol"]].dropna().astype(str).tolist()))
    if "gene_id" in cols:
        return set(map(str, df[cols["gene_id"]].dropna().astype(str).tolist()))
    raise ValueError("Gold file must contain 'symbol' or 'gene_id' column for CSV/TSV, or be a TXT list")


def evaluate_results_vs_gold(results_path: str, gold_path: str, k: int = 20) -> Dict:
    """Re-evaluate a results JSON against a gold-standard set.

    Adds metrics keys with suffix '_gold': precision@k, mrr, ndcg@k, panel_recall.
    """
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ranking = data.get("ranking") or []
    panel = data.get("panel") or []
    gold = load_gold_set(gold_path)

    def precision_at_k(lst: List[str], k: int) -> float:
        if k <= 0:
            return 0.0
        top = lst[:k]
        return sum(1 for x in top if x in gold) / float(k)

    def mrr(lst: List[str]) -> float:
        for i, x in enumerate(lst, start=1):
            if x in gold:
                return 1.0 / i
        return 0.0

    import math
    def dcg_at_k(rels: List[int], k: int) -> float:
        s = 0.0
        for i in range(min(k, len(rels))):
            s += rels[i] / math.log2(i + 2)
        return s

    def ndcg_at_k(lst: List[str], k: int) -> float:
        rels = [1 if x in gold else 0 for x in lst]
        dcg = dcg_at_k(rels, k)
        ideal = sorted(rels, reverse=True)
        idcg = dcg_at_k(ideal, k)
        if idcg == 0:
            return 0.0
        return dcg / idcg

    def panel_recall(p: List[str], g: Set[str]) -> float:
        if not g:
            return 0.0
        hits = sum(1 for x in p if x in g)
        return hits / float(len(g))

    out = {
        f"precision_at_{k}_gold": precision_at_k(ranking, k),
        "mrr_gold": mrr(ranking),
        f"ndcg_at_{k}_gold": ndcg_at_k(ranking, k),
        "panel_recall_gold": panel_recall(panel, gold),
    }
    # merge (non-destructive) and write sibling file
    base, ext = os.path.splitext(results_path)
    out_path = base + "_with_gold.json"
    merged = dict(data)
    merged.update(out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    return {"out": out_path, **out}
