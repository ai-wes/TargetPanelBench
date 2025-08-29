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


def plot_ndcg_vs_diversity(df: pd.DataFrame, outpath: Path) -> None:
    if not {"ndcg_at_20", "panel_diversity"}.issubset(df.columns):
        return
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="ndcg_at_20", y="panel_diversity", hue="method", s=60)
    plt.xlabel("nDCG@20")
    plt.ylabel("Panel diversity (graph)")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_ndcg_vs_string_diversity(df: pd.DataFrame, outpath: Path) -> None:
    if not {"ndcg_at_20", "panel_diversity_string"}.issubset(df.columns):
        return
    sub = df.dropna(subset=["panel_diversity_string"]).copy()
    if sub.empty: return
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=sub, x="ndcg_at_20", y="panel_diversity_string", hue="method", s=60)
    plt.xlabel("nDCG@20")
    plt.ylabel("Panel diversity (STRING)")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_precision_gold_overlay(df: pd.DataFrame, outpath: Path) -> None:
    # Expect columns precision_at_20 and precision_at_20_gold
    if "precision_at_20" not in df.columns or "precision_at_20_gold" not in df.columns:
        return
    plt.figure(figsize=(7, 4))
    width = 0.35
    x = range(len(df))
    plt.bar([i - width/2 for i in x], df["precision_at_20"], width=width, label="p@20 (native)")
    plt.bar([i + width/2 for i in x], df["precision_at_20_gold"], width=width, label="p@20 (gold)")
    plt.xticks(list(x), df["method"], rotation=30, ha="right")
    plt.legend()
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
    # Scatter plots
    plot_ndcg_vs_diversity(df, Path(args.outdir) / "ndcg_vs_diversity.png")
    print(f"Wrote {Path(args.outdir) / 'ndcg_vs_diversity.png'}")
    plot_ndcg_vs_string_diversity(df, Path(args.outdir) / "ndcg_vs_string_diversity.png")
    print(f"Wrote {Path(args.outdir) / 'ndcg_vs_string_diversity.png'}")
    # Gold overlay if present
    if "precision_at_20_gold" in df.columns:
        plot_precision_gold_overlay(df, Path(args.outdir) / "precision_gold_overlay.png")
        print(f"Wrote {Path(args.outdir) / 'precision_gold_overlay.png'}")


if __name__ == "__main__":
    main()
