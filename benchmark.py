"""
benchmark.py
============

Unified CLI to run TargetPanelBench baselines and public tool adapters,
collect results and summarise metrics for reporting.

Examples
--------
# Generate synthetic data
python download_data.py --num-genes 200 --num-positives 40

# Run baselines on synthetic data
python benchmark.py run-baselines --data-dir data --out results/baselines_synth.json

# Run OpenTargets adapter for Type 2 Diabetes (EFO_0003767)
python benchmark.py run-opentargets --efo-id EFO_0003767 --out results/ot_t2d.json

# Summarise results directory
python benchmark.py summarise --results-dir results --out results/summary.csv
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Local imports for baselines
from baselines.simple_score_rank import run_simple_baseline
from baselines.cma_es import run_cma_es_baseline


def cmd_run_baselines(args: argparse.Namespace) -> None:
    targets_path = os.path.join(args.data_dir, "targets.csv")
    edges_path = os.path.join(args.data_dir, "ppi_edges.csv")
    results = {}
    # Simple baseline
    simple_res = run_simple_baseline(targets_path, edges_path, panel_size=args.panel_size, k=args.k)
    results["simple_score_rank"] = simple_res
    # CMA-ES baseline (random search)
    cma_res = run_cma_es_baseline(
        targets_path,
        edges_path,
        iterations=args.iterations,
        objective=args.objective,
        panel_size=args.panel_size,
        k=args.k,
        seed=args.seed,
    )
    results["cma_es_weighted_sum"] = cma_res

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    # Print a concise table of metrics
    rows = []
    for name, res in results.items():
        m = res.get("metrics", {})
        rows.append({"method": name, **m, "panel_recall": res.get("panel_recall"), "panel_diversity": res.get("panel_diversity")})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def cmd_run_opentargets(args: argparse.Namespace) -> None:
    from public_methods.opentargets import run_opentargets_ranking
    res = run_opentargets_ranking(args.efo_id, n_candidates=args.n_candidates, m_truth=args.m_truth, k=args.k)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res.get("metrics", res), indent=2))


def cmd_run_opentargets_local(args: argparse.Namespace) -> None:
    from public_methods.opentargets_local import run_opentargets_local
    res = run_opentargets_local(args.ot_dir, args.disease_id, n_candidates=args.n_candidates, m_truth=args.m_truth, k=args.k)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    if "metrics" in res:
        print(json.dumps(res["metrics"], indent=2))
    else:
        print(json.dumps(res, indent=2))


def cmd_run_morphantic_aea(args: argparse.Namespace) -> None:
    from public_methods.morphantic_aea import run_morphantic_aea
    targets_path = os.path.join(args.data_dir, "targets.csv")
    edges_path = os.path.join(args.data_dir, "ppi_edges.csv")
    res = run_morphantic_aea(
        targets_path,
        edges_path,
        panel_size=args.panel_size,
        k=args.k,
        seed=args.seed,
        pop_size=args.pop_size,
        max_generations=args.max_generations,
        n_islands=args.n_islands,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res.get("metrics", res), indent=2))


def _read_results_file(path: Path) -> List[Dict]:
    items = []
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "metrics" in data:
            items.append({"file": path.name, **data.get("metrics", {}), "method": path.stem})
        elif isinstance(data, dict):
            # probably a map of method -> result
            for mname, res in data.items():
                items.append({"file": path.name, **res.get("metrics", {}), "method": mname})
    except Exception as e:
        items.append({"file": path.name, "error": str(e)})
    return items


def cmd_summarise(args: argparse.Namespace) -> None:
    rows: List[Dict] = []
    for f in Path(args.results_dir).glob("*.json"):
        rows.extend(_read_results_file(f))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TargetPanelBench unified benchmark CLI")
    sp = p.add_subparsers(dest="cmd", required=True)

    b = sp.add_parser("run-baselines", help="Run built-in baselines on local dataset")
    b.add_argument("--data-dir", default="data")
    b.add_argument("--panel-size", type=int, default=12)
    b.add_argument("--k", type=int, default=20)
    b.add_argument("--iterations", type=int, default=100)
    b.add_argument("--objective", choices=["ndcg", "precision", "mrr"], default="ndcg")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--out", default="results/baselines.json")
    b.set_defaults(func=cmd_run_baselines)

    ot = sp.add_parser("run-opentargets", help="Run OpenTargets adapter for a disease")
    ot.add_argument("--efo-id", required=True)
    ot.add_argument("--n-candidates", type=int, default=500)
    ot.add_argument("--m-truth", type=int, default=25)
    ot.add_argument("--k", type=int, default=20)
    ot.add_argument("--out", default="results/opentargets.json")
    ot.set_defaults(func=cmd_run_opentargets)

    ma = sp.add_parser("run-morphantic-aea", help="Run Morphantic AEA on local dataset (optimize feature weights)")
    ma.add_argument("--data-dir", default="data")
    ma.add_argument("--panel-size", type=int, default=12)
    ma.add_argument("--k", type=int, default=20)
    ma.add_argument("--seed", type=int, default=42)
    ma.add_argument("--pop-size", type=int, default=32)
    ma.add_argument("--max-generations", type=int, default=40)
    ma.add_argument("--n-islands", type=int, default=2)
    ma.add_argument("--out", default="results/morphantic_aea.json")
    ma.set_defaults(func=cmd_run_morphantic_aea)

    otl = sp.add_parser("run-opentargets-local", help="Run OpenTargets adapter on local Parquet data")
    otl.add_argument("--ot-dir", required=True, help="Path to Open Targets parquet root (contains associations_direct_overall and targets_core_by_annotation)")
    otl.add_argument("--disease-id", required=True, help="Disease Ontology ID, e.g., DOID_0050890")
    otl.add_argument("--n-candidates", type=int, default=500)
    otl.add_argument("--m-truth", type=int, default=25)
    otl.add_argument("--k", type=int, default=20)
    otl.add_argument("--out", default="results/opentargets_local.json")
    otl.set_defaults(func=cmd_run_opentargets_local)

    s = sp.add_parser("summarise", help="Summarise all JSON results into a CSV")
    s.add_argument("--results-dir", default="results")
    s.add_argument("--out", default="results/summary.csv")
    s.set_defaults(func=cmd_summarise)

    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
