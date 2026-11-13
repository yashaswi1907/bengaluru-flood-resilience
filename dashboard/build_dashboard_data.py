"""
Pre-computes model predictions for every dashboard scenario and writes dashboard/data.js.
Run from the project root:  python3 dashboard/build_dashboard_data.py
(Only needed again if you retrain the model.)
"""
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from predict import FloodPredictor  # noqa: E402
from reference_data import FLOOD_HOTSPOTS, LAKES, HOSPITALS  # noqa: E402

RAINS = [50, 75, 100, 125, 150, 175, 200]
DRAINS = [0, 15, 30, 50]
DESILT = [0, 1]
GREEN = [0, 1]          # 1 = convert 10% of impervious surface to green cover


def main():
    fp = FloodPredictor()
    g = fp.grid
    scen = {}
    for r, d, s, gr in itertools.product(RAINS, DRAINS, DESILT, GREEN):
        out = fp.scenario(r, drain_capacity_boost_pct=d, desilting=bool(s), green_cover_increase=0.1 * gr)
        scen[f"{r}_{d}_{s}_{gr}"] = {
            "p": (out.flood_probability.values * 100).round().astype(int).tolist(),
            "s": out.severity_class.astype(int).tolist(),
        }
    metrics = json.load(open(os.path.join(ROOT, "outputs", "metrics.json")))
    data = {
        "threshold": round(fp.thr * 100, 1),
        "model": metrics["best_model"],
        "metrics": {k: {m: round(v[m], 3) for m in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]}
                    for k, v in metrics["test_metrics"].items()},
        "hotspot_recall": metrics["hotspot_recall_top10pct_100mm"],
        "severity_f1": metrics["severity_report"]["macro avg"]["f1-score"],
        "cells": {"id": g.cell_id.tolist(), "lat": g.lat.tolist(), "lon": g.lon.tolist(),
                  "zone": g.zone.tolist(), "pop": g.pop_density_per_km2.astype(int).tolist(),
                  "elev": g.elevation_m.tolist(), "imperv": g.impervious_frac.tolist(),
                  "drain": g.drain_capacity_mm_hr.tolist()},
        "hotspots": [[h[0], h[1], h[2], h[3]] for h in FLOOD_HOTSPOTS],
        "lakes": [[l[0], l[1], l[2], l[3]] for l in LAKES],
        "hospitals": [[h[0], h[1], h[2], h[3]] for h in HOSPITALS],
        "scenarios": scen,
    }
    path = os.path.join(ROOT, "dashboard", "data.js")
    with open(path, "w") as f:
        f.write("window.FLOOD_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB, {len(scen)} scenarios)")


if __name__ == "__main__":
    main()
