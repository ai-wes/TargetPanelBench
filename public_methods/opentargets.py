"""
OpenTargets adapter
===================

Fetches gene rankings for a disease using the Open Targets GraphQL API
and returns a ranked list of targets with association scores.

This module provides two main entry points:

- fetch_disease_associations(efo_id, size): returns a dataframe of
  targets for the disease with association scores and metadata.
- run_opentargets_ranking(efo_id, top_k_candidates, top_m_ground_truth):
  builds a candidate set and a ground-truth set directly from Open Targets
  and evaluates ranking metrics using a temporal-agnostic, self-contained
  protocol (top-m as positives).

Notes
-----
This adapter is intended to demonstrate how to integrate a public tool.
For rigorous benchmarking, ground truth should come from an independent
source (e.g., ClinGen, PanelApp, curated literature) rather than the same
signal used for ranking. See docs/benchmark_protocol.md for discussion.
"""
from __future__ import annotations

import json
import os
from typing import List, Dict, Tuple

import requests
import pandas as pd

OPENTARGETS_GQL = "https://api.platform.opentargets.org/api/v4/graphql"


def _insecure_verify() -> bool | None:
    # If TPB_INSECURE_SSL=1, disable SSL verification to work around corp proxies.
    return False if os.environ.get("TPB_INSECURE_SSL") == "1" else True


def _run_graphql(query: str, variables: dict | None = None) -> dict:
    resp = requests.post(
        OPENTARGETS_GQL,
        json={"query": query, "variables": variables or {}},
        timeout=60,
        verify=_insecure_verify(),
        headers={"User-Agent": "TargetPanelBench/1.0"},
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"OpenTargets GraphQL error: {data['errors']}")
    return data["data"]


def _fetch_disease_associations_rest(efo_id: str, size: int = 1000) -> pd.DataFrame:
    """Fallback using legacy REST v3 filter endpoint.
    Example: https://platform-api.opentargets.io/v3/platform/public/association/filter?disease=EFO_0003767&size=500
    """
    url = "https://platform-api.opentargets.io/v3/platform/public/association/filter"
    params = {"disease": efo_id, "size": int(size)}
    r = requests.get(url, params=params, timeout=60, verify=_insecure_verify(), headers={"User-Agent": "TargetPanelBench/1.0"})
    r.raise_for_status()
    js = r.json()
    data = js.get("data", [])
    rows = []
    for d in data:
        tgt = d.get("target", {})
        gene_info = tgt.get("gene_info", {})
        rows.append({
            "target_id": tgt.get("id"),
            "symbol": gene_info.get("symbol"),
            "score": d.get("association_score", {}).get("overall"),
            "datasourceCount": len(d.get("evidence", [])) if isinstance(d.get("evidence", []), list) else None,
        })
    df = pd.DataFrame(rows).dropna(subset=["symbol", "score"]).sort_values("score", ascending=False).reset_index(drop=True)
    return df


def fetch_disease_associations(efo_id: str, size: int = 1000) -> pd.DataFrame:
    """Fetch disease→target associations ranked by association score.

    Parameters
    ----------
    efo_id : str
        EFO disease identifier (e.g., 'EFO_0003767' for type 2 diabetes mellitus).
    size : int
        Max number of targets to return.

    Returns
    -------
    pd.DataFrame with columns: target_id, symbol, score, datasourceCount
    """
    query = """
    query diseaseAssociations($efoId: String!, $size: Int!) {
      disease(efoId: $efoId) {
        id
        name
        associatedTargets(page: {index: 0, size: $size}) {
          rows {
            target { id approvedSymbol }
            score
            datasourceScores { sourceId id score }
          }
        }
      }
    }
    """
    try:
        data = _run_graphql(query, {"efoId": efo_id, "size": int(size)})
        rows = (
            data
            .get("disease", {})
            .get("associatedTargets", {})
            .get("rows", [])
        )
        records = []
        for row in rows:
            tgt = (row or {}).get("target", {})
            score = (row or {}).get("score")
            ds = (row or {}).get("datasourceScores", []) or []
            ds_count = len(ds) if isinstance(ds, list) else 0
            records.append({
                "target_id": tgt.get("id"),
                "symbol": tgt.get("approvedSymbol"),
                "score": score,
                "datasourceCount": ds_count,
            })
        df = pd.DataFrame(records).dropna(subset=["symbol", "score"]).sort_values("score", ascending=False).reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError("Empty results from GraphQL")
        return df
    except Exception:
        # Fallback to REST v3 endpoint
        return _fetch_disease_associations_rest(efo_id, size)


def build_candidate_and_truth_from_ot(efo_id: str, n_candidates: int = 500, m_truth: int = 25) -> Tuple[pd.DataFrame, List[str]]:
    """Construct candidate set and ground-truth directly from Open Targets.

    Returns a DataFrame with at least 'gene_id' (HGNC symbol) and a list of
    positives (gene symbols) consisting of the top m_truth by OT score.
    """
    assoc = fetch_disease_associations(efo_id, size=max(n_candidates, m_truth) + 100)
    assoc = assoc.dropna(subset=["symbol"]).reset_index(drop=True)
    # Candidate set: top-n by score
    candidates = assoc.iloc[:n_candidates].copy()
    candidates.rename(columns={"symbol": "gene_id"}, inplace=True)
    # Ground truth: top-m by score
    positives = assoc.iloc[:m_truth]["symbol"].tolist()
    return candidates, positives


def evaluate_ranking_against_truth(ranking: List[str], positives: List[str], k: int = 20) -> Dict[str, float]:
    """Compute simple ranking metrics given a ranked list and positives.

    Uses precision@k, MRR and nDCG@k (binary relevance).
    """
    # Local re-implementation to avoid coupling to repo internals
    import math

    gt = set(positives)

    def precision_at_k(lst: List[str], k: int) -> float:
        if k <= 0:
            return 0.0
        top = lst[:k]
        return sum(1 for x in top if x in gt) / float(k)

    def mrr(lst: List[str]) -> float:
        for i, x in enumerate(lst, start=1):
            if x in gt:
                return 1.0 / i
        return 0.0

    def dcg_at_k(rels: List[int], k: int) -> float:
        s = 0.0
        for i in range(min(k, len(rels))):
            s += rels[i] / math.log2(i + 2)
        return s

    def ndcg_at_k(lst: List[str], k: int) -> float:
        rels = [1 if x in gt else 0 for x in lst]
        dcg = dcg_at_k(rels, k)
        ideal = sorted(rels, reverse=True)
        idcg = dcg_at_k(ideal, k)
        if idcg == 0:
            return 0.0
        return dcg / idcg

    return {
        f"precision_at_{k}": precision_at_k(ranking, k),
        "mrr": mrr(ranking),
        f"ndcg_at_{k}": ndcg_at_k(ranking, k),
    }


def run_opentargets_ranking(efo_id: str, n_candidates: int = 500, m_truth: int = 25, k: int = 20) -> Dict:
    """Run OpenTargets ranking for a disease and evaluate on self-contained truth.

    Returns a dictionary with ranking, metrics and metadata.
    """
    candidates, positives = build_candidate_and_truth_from_ot(efo_id, n_candidates, m_truth)
    ranking = candidates.sort_values("score", ascending=False)["gene_id"].tolist()
    metrics = evaluate_ranking_against_truth(ranking, positives, k=k)
    return {
        "disease_efo": efo_id,
        "candidates": candidates.to_dict(orient="records"),
        "positives": positives,
        "ranking": ranking,
        "metrics": metrics,
        "protocol": {
            "candidate_source": "OpenTargets associations",
            "truth_source": "Top-m OpenTargets (self-constructed)",
            "limitations": "Truth derived from same source; use with caution",
        }
    }


if __name__ == "__main__":
    import argparse, os, json
    parser = argparse.ArgumentParser(description="Run OpenTargets ranking for a disease EFO ID")
    parser.add_argument("--efo-id", required=True, help="EFO disease ID, e.g. EFO_0003767")
    parser.add_argument("--n-candidates", type=int, default=500)
    parser.add_argument("--m-truth", type=int, default=25)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--out", type=str, default="results/opentargets_EFO.json")
    args = parser.parse_args()
    res = run_opentargets_ranking(args.efo_id, args.n_candidates, args.m_truth, args.k)
    os.makedirs("results", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res["metrics"], indent=2))
