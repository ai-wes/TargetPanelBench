"""
reporting.py
============

Generate publication-ready plots from the benchmark summary CSV.

Usage:
  python reporting.py --summary results/summary.csv --outdir results/figures
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure consistent method naming
    if "method" not in df.columns:
        df["method"] = df.get("file", "unknown")
    return df


def plot_metric_bar(df: pd.DataFrame, metric: str, outpath: Path) -> None:
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x="method", y=metric, color="#4C78A8")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Generate plots from summary CSV")
    p.add_argument("--summary", default="results/summary.csv")
    p.add_argument("--outdir", default="results/figures")
    args = p.parse_args()

    df = load_summary(args.summary)
    # Select only rows that have the metric columns
    metrics = [c for c in ["precision_at_20", "mrr", "ndcg_at_20"] if c in df.columns]
    for m in metrics:
        out = Path(args.outdir) / f"{m}.png"
        plot_metric_bar(df, m, out)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
