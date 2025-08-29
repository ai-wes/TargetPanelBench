"""
STRING API fetcher and cache utilities.

Fetches STRING interactions for a list of gene symbols and writes a TSV with
columns preferredName_A,preferredName_B,score, suitable for recomputing
panel_diversity_string.
"""
from __future__ import annotations

import os
import time
from typing import Iterable, List, Dict

import requests
import pandas as pd

STRING_API = "https://string-db.org/api/tsv/network"


def fetch_string_network(genes: Iterable[str], species: int = 9606, required_score: int = 700, pause_s: float = 0.0) -> pd.DataFrame:
    ids = "\r".join(sorted(set(map(str, genes))))
    params = {
        "identifiers": ids,
        "species": int(species),
        "required_score": int(required_score),
        "caller_identity": "TargetPanelBench"
    }
    resp = requests.get(STRING_API, params=params, timeout=60)
    resp.raise_for_status()
    # STRING returns TSV text
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text), sep="\t")
    if pause_s > 0:
        time.sleep(pause_s)
    # Normalize columns to preferredName_A/B if present (else try mapping)
    cols = set(df.columns)
    if {"preferredName_A", "preferredName_B"}.issubset(cols):
        out = df[["preferredName_A", "preferredName_B", "score"]].copy()
    elif {"stringId_A", "stringId_B"}.issubset(cols):
        # Fallback; keep raw IDs
        out = df[["stringId_A", "stringId_B", "score"]].copy()
        out.rename(columns={"stringId_A": "preferredName_A", "stringId_B": "preferredName_B"}, inplace=True)
    else:
        # Return raw; caller can inspect
        out = df
    # Drop self-loops and duplicates
    out = out[out["preferredName_A"] != out["preferredName_B"]]
    out = out.dropna().drop_duplicates()
    return out


def cache_string_edges(genes: Iterable[str], cache_path: str, species: int = 9606, required_score: int = 700) -> str:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df = fetch_string_network(genes, species=species, required_score=required_score)
    df.to_csv(cache_path, sep="\t", index=False)
    return cache_path
