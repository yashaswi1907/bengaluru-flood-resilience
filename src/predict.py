"""
What-if predictor - the interface Modules 2-7 (infrastructure, network, emergency, optimisation,
simulator, dashboard) call to get flood risk for any rainfall scenario or intervention.

Examples
--------
python src/predict.py --rain 120
python src/predict.py --rain 150 --drain-upgrade 25 --desilt     # simulate interventions
python src/predict.py --rain 100 --lat 12.9172 --lon 77.6229      # single location (Silk Board)

In Python:
    from predict import FloodPredictor
    fp = FloodPredictor()
    df = fp.scenario(rain_24h_mm=150, drain_capacity_boost_pct=25, desilting=True)
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd

from features import add_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FloodPredictor:
    def __init__(self):
        b = joblib.load(os.path.join(ROOT, "models", "flood_model.joblib"))
        self.model, self.sev, self.thr, self.feats = b["flood_model"], b["severity_model"], b["threshold"], b["features"]
        self.grid = pd.read_csv(os.path.join(ROOT, "data", "processed", "grid_cells.csv"))

    def scenario(self, rain_24h_mm, peak_fraction=0.4, antecedent_3day_mm=30, month=9,
                 drain_capacity_boost_pct=0.0, desilting=False, green_cover_increase=0.0, cells=None):
        g = self.grid.copy() if cells is None else self.grid[self.grid.cell_id.isin(cells)].copy()
        # ---- interventions (Module 5) ----
        g["drain_capacity_mm_hr"] *= 1 + drain_capacity_boost_pct / 100
        if desilting:
            g["drain_silt_index"] *= 0.3
        if green_cover_increase:
            g["impervious_frac"] = (g.impervious_frac - green_cover_increase).clip(0.1, 1)
            g["green_cover_frac"] = (g.green_cover_frac + green_cover_increase).clip(0, 0.9)
        g["rain_24h_mm"], g["peak_hourly_mm"] = rain_24h_mm, rain_24h_mm * peak_fraction
        g["antecedent_3day_mm"], g["month"] = antecedent_3day_mm, month
        X = add_features(g)[self.feats]
        p = self.model.predict_proba(X)[:, 1]
        P = self.sev.predict_proba(X)
        ent = -(np.clip(P, 1e-9, 1) * np.log(np.clip(P, 1e-9, 1))).sum(1) / np.log(P.shape[1])
        out = g[["cell_id", "lat", "lon", "zone", "pop_density_per_km2", "dist_hospital_km"]].copy()
        out["flood_probability"] = p.round(4)
        out["flood_predicted"] = (p >= self.thr).astype(int)
        out["severity_class"] = P.argmax(1)                 # 0 none, 1 minor, 2 moderate, 3 severe
        out["confidence_score"] = (1 - ent).round(3)
        out["people_at_risk"] = (out.flood_probability * out.pop_density_per_km2).round(0)  # per ~1 km2 cell
        return out

    def nearest_cell(self, lat, lon):
        d = (self.grid.lat - lat) ** 2 + (self.grid.lon - lon) ** 2
        return self.grid.cell_id.iloc[int(d.idxmin())]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rain", type=float, default=100)
    ap.add_argument("--drain-upgrade", type=float, default=0, help="% increase in drain capacity")
    ap.add_argument("--desilt", action="store_true")
    ap.add_argument("--green", type=float, default=0, help="fraction of impervious surface converted to green")
    ap.add_argument("--lat", type=float); ap.add_argument("--lon", type=float)
    a = ap.parse_args()
    fp = FloodPredictor()
    cells = [fp.nearest_cell(a.lat, a.lon)] if a.lat else None
    base = fp.scenario(a.rain, cells=cells)
    new = fp.scenario(a.rain, drain_capacity_boost_pct=a.drain_upgrade, desilting=a.desilt,
                      green_cover_increase=a.green, cells=cells)
    if cells:
        print(new.T.to_string())
    else:
        print(f"Scenario: {a.rain:.0f} mm / 24 h")
        print(f"  cells predicted to flood : {base.flood_predicted.sum()} -> {new.flood_predicted.sum()} (after interventions)")
        print(f"  expected people at risk  : {base.people_at_risk.sum():,.0f} -> {new.people_at_risk.sum():,.0f}")
        print("\nTop 10 highest-risk cells:")
        print(new.sort_values("flood_probability", ascending=False).head(10).to_string(index=False))
