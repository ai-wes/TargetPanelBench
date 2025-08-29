"""
ClinGen gold-standard loader (local file utility).

Loads a TXT/CSV/TSV list of symbols or a column symbol/gene_id and writes a normalized
TXT gold list file.
"""
from __future__ import annotations

import os
import pandas as pd


def write_gold_from_file(input_path: str, out_path: str) -> str:
    lower = input_path.lower()
    if lower.endswith(".txt") or lower.endswith(".lst"):
        with open(input_path, "r", encoding="utf-8") as f:
            syms = [ln.strip() for ln in f if ln.strip()]
    else:
        sep = "," if lower.endswith(".csv") else "\t"
        df = pd.read_csv(input_path, sep=sep)
        cols = {c.lower(): c for c in df.columns}
        if "symbol" in cols:
            syms = df[cols["symbol"]].dropna().astype(str).tolist()
        elif "gene_id" in cols:
            syms = df[cols["gene_id"]].dropna().astype(str).tolist()
        else:
            raise ValueError("File must contain a 'symbol' or 'gene_id' column")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in syms:
            f.write(str(s) + "\n")
    return out_path
