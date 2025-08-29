"""
PanelApp gold-standard loader.

Fetches genes for a given PanelApp panel ID and writes a gold list (symbols).
If network is unavailable, allows reading from a local JSON export.
"""
from __future__ import annotations

from typing import List, Dict
import os
import json
import requests

PANELAPP_API = "https://panelapp.genomicsengland.co.uk/api/v1/panels/"


def fetch_panelapp_panel(panel_id: str) -> Dict:
    url = PANELAPP_API.rstrip("/") + f"/{panel_id}/"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def extract_symbols(panel_json: Dict, min_confidence: int = 2) -> List[str]:
    symbols: List[str] = []
    genes = (panel_json.get("genes", []) or [])
    for g in genes:
        # keep only confident genes at or above threshold
        if int(g.get("confidence_level", 0)) >= int(min_confidence):
            sym = g.get("gene_data", {}).get("gene_symbol") or g.get("gene_symbol") or g.get("symbol")
            if sym:
                symbols.append(str(sym))
    return sorted(set(symbols))


def write_gold_list(symbols: List[str], out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in symbols:
            f.write(s + "\n")
    return out_path
