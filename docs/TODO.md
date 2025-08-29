TODO (TargetPanelBench)
=======================

1) STRING PPI fetch + cache
- Add CLI to fetch STRING interactions for a gene set, cache TSV in data/string_cache, and recompute panel_diversity_string.
- Inputs: genes file or infer from data/targets.csv; params: species, required_score, output label.

2) Gold standards (PanelApp/ClinGen) loaders
- PanelApp: fetch by panel ID or name, produce gold list (symbols), and cache to data/gold/.
- ClinGen: loader for local CSV/TSV/TXT plus optional API stub for future.
- CLI: evaluate-vs-gold already present; add fetch-panelapp-gold, generate-gold-from-file.

3) Reporting extensions
- Dual‑axis/scatter: nDCG@k vs panel diversity (and panel_diversity_string) per method.
- Bar overlay: precision@k vs precision@k_gold side‑by‑side per method.

4) MO sweeps for Pareto view
- CLI to sweep diversity weights (e.g., 0.0→0.5), low/med budgets, write a CSV of (w, nDCG, diversity, recall), and plot nDCG vs diversity.
- Defaults conservative to run quickly; allow tuning.

5) Run feasible offline steps
- Recompute STRING diversity when a TSV is available.
- Evaluate vs gold (provided PanelApp/ClinGen files).
- Update results/summary.csv and figures.
