"""
OpenTargets Local adapter (Parquet)
===================================

Loads disease→target associations from local Open Targets Parquet datasets and
produces a ranked target list for a given Disease Ontology ID (DOID_*).

Expected directory structure (subset):
- associations_direct_overall/*.parquet with columns:
  ['diseaseId','targetId','score','evidenceCount']
- targets_core_by_annotation/*.parquet with columns including:
  ['id','approvedSymbol']

Example:
  python benchmark.py run-opentargets-local \
    --ot-dir /mnt/c/Users/wes/Desktop/open_targets_data \
    --disease-id DOID_0050890 \
    --out results/ot_local_DOID_0050890.json
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class OTLocalPaths:
    ot_dir: Path
    assoc_dir: Path
    targets_dir: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "OTLocalPaths":
        root = Path(root)
        return cls(
            ot_dir=root,
            assoc_dir=root / "associations_direct_overall",
            targets_dir=root / "targets_core_by_annotation",
        )


def _load_symbol_map(paths: OTLocalPaths) -> pd.DataFrame:
    # Concatenate minimal columns across parts for memory efficiency
    frames = []
    for f in sorted(paths.targets_dir.glob("*.parquet")):
        df = pd.read_parquet(f, columns=["id", "approvedSymbol"])  # ENSG -> symbol
        frames.append(df)
    sym = pd.concat(frames, ignore_index=True).dropna(subset=["approvedSymbol"]).drop_duplicates("id")
    return sym


def _iter_assoc_parts(paths: OTLocalPaths):
    for f in sorted(paths.assoc_dir.glob("*.parquet")):
        yield pd.read_parquet(f, columns=["diseaseId", "targetId", "score", "evidenceCount"])  # DOID, ENSG


def fetch_local_associations(paths: OTLocalPaths, disease_id: str, size: int = 1000) -> pd.DataFrame:
    """Collect top-N associations for a disease (DOID_*)."""
    candidates = []
    for part in _iter_assoc_parts(paths):
        sub = part.loc[part["diseaseId"] == disease_id, ["targetId", "score", "evidenceCount"]]
        if not sub.empty:
            candidates.append(sub)
    if not candidates:
        return pd.DataFrame(columns=["targetId", "score", "evidenceCount", "symbol"])  # empty
    df = pd.concat(candidates, ignore_index=True)
    # Aggregate in case target appears across parts
    agg = df.groupby("targetId", as_index=False).agg({"score": "max", "evidenceCount": "sum"})
    agg = agg.sort_values("score", ascending=False).head(size)
    # Map to symbols
    sym_map = _load_symbol_map(paths)
    out = agg.merge(sym_map, left_on="targetId", right_on="id", how="left").drop(columns=["id"]).rename(columns={"approvedSymbol": "symbol"})
    return out


def evaluate_ranking_against_truth(ranking: List[str], positives: List[str], k: int = 20) -> Dict[str, float]:
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


def run_opentargets_local(ot_dir: str, disease_id: str, n_candidates: int = 500, m_truth: int = 25, k: int = 20) -> Dict:
    paths = OTLocalPaths.from_root(ot_dir)
    assoc = fetch_local_associations(paths, disease_id, size=max(n_candidates, m_truth) + 100)
    if assoc.empty:
        return {
            "disease_id": disease_id,
            "error": f"No associations found for {disease_id} in {ot_dir}",
        }
    # Build candidate set and truth
    assoc = assoc.dropna(subset=["symbol"]).reset_index(drop=True)
    candidates = assoc.head(n_candidates).copy()
    positives = assoc.head(m_truth)["symbol"].tolist()
    ranking = candidates["symbol"].tolist()  # already sorted by score desc
    metrics = evaluate_ranking_against_truth(ranking, positives, k=k)
    return {
        "source": "OpenTargets local parquet",
        "disease_id": disease_id,
        "candidates": candidates.to_dict(orient="records"),
        "positives": positives,
        "ranking": ranking,
        "metrics": metrics,
        "protocol": {
            "candidate_source": "OT associations (local)",
            "truth_source": "Top-m from same source (demo)",
            "limitations": "Truth derived from same source; use independent gold standard for publication",
        }
    }
