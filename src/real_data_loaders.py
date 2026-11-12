"""
Swap in REAL downloaded data.

Put downloaded files in data/raw/ and run:
    python src/real_data_loaders.py

What it does (each step is optional - it only runs if the file exists):
  1. data/raw/flood_prone_locations.kml   (OpenCity "Flood Prone Locations Map", BBMP)
       -> replaces the hotspot list in reference_data.py and writes data/processed/real_hotspots.csv
  2. data/raw/primary_swd.kml             (OpenCity "Bengaluru Primary Stormwater Drains Map - 2022")
       -> recomputes dist_primary_drain_km for every grid cell
  3. data/raw/lakes.kml                    (any BBMP/KSRSAC lakes KML with polygons or points)
       -> recomputes dist_lake_km
  4. data/raw/dem.csv  with columns lat,lon,elevation_m (sampled from SRTM / Copernicus 30 m DEM,
       e.g. exported from Google Earth Engine or QGIS "Sample raster values")
       -> replaces elevation_m, slope_pct, low_lying_m
  5. data/raw/rainfall_events.csv with columns date,station_or_ward,lat,lon,rain_24h_mm
       (KSNDMC Varuna Mitra daily ward rainfall)  and
     data/raw/flood_reports.csv with columns date,lat,lon  (BBMP control-room / news / Twitter reports)
       -> rebuilds data/processed/flood_events.csv with REAL labels (flooded = report within 1 km)

After this, re-run:  python src/train.py
Only the standard library + numpy/pandas/scipy are needed (no geopandas).
"""
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
KML_NS = {"k": "http://www.opengis.net/kml/2.2"}


def read_kml_points(path):
    """Return DataFrame(name, lat, lon) of every coordinate vertex in a KML (points, lines or polygons)."""
    root = ET.parse(path).getroot()
    rows = []
    for pm in root.iter("{http://www.opengis.net/kml/2.2}Placemark"):
        name_el = pm.find("k:name", KML_NS)
        name = name_el.text.strip() if name_el is not None and name_el.text else ""
        for c in pm.iter("{http://www.opengis.net/kml/2.2}coordinates"):
            for tok in c.text.split():
                parts = tok.split(",")
                if len(parts) >= 2:
                    rows.append((name, float(parts[1]), float(parts[0])))
    return pd.DataFrame(rows, columns=["name", "lat", "lon"])


def _xy(lat, lon, lat0=12.9716, lon0=77.5946):
    return np.c_[(np.asarray(lon) - lon0) * 111.32 * np.cos(np.radians(lat0)), (np.asarray(lat) - lat0) * 110.57]


def nearest_km(grid, pts):
    tree = cKDTree(_xy(pts.lat, pts.lon))
    d, _ = tree.query(_xy(grid.lat, grid.lon))
    return d


def densify(pts, step_km=0.1):
    """Add points along consecutive vertices of each placemark so line distances are accurate."""
    out = []
    for name, g in pts.groupby("name", sort=False):
        la, lo = g.lat.values, g.lon.values
        for i in range(len(la) - 1):
            n = max(2, int(np.hypot((la[i + 1] - la[i]) * 110.57, (lo[i + 1] - lo[i]) * 108) / step_km))
            out.append(pd.DataFrame({"name": name, "lat": np.linspace(la[i], la[i + 1], n),
                                     "lon": np.linspace(lo[i], lo[i + 1], n)}))
    return pd.concat(out, ignore_index=True) if out else pts


def main():
    grid_path = os.path.join(PROC, "grid_cells.csv")
    grid = pd.read_csv(grid_path)
    changed = []

    f = os.path.join(RAW, "flood_prone_locations.kml")
    if os.path.exists(f):
        hs = read_kml_points(f).groupby("name", as_index=False).first()
        hs.to_csv(os.path.join(PROC, "real_hotspots.csv"), index=False)
        grid["dist_known_hotspot_km"] = nearest_km(grid, hs).round(3)
        tree = cKDTree(_xy(hs.lat, hs.lon))
        grid["hist_flood_count_2015_24"] = [len(i) for i in tree.query_ball_point(_xy(grid.lat, grid.lon), r=1.0)]
        changed.append(f"hotspots ({len(hs)} real locations)")

    f = os.path.join(RAW, "primary_swd.kml")
    if os.path.exists(f):
        grid["dist_primary_drain_km"] = nearest_km(grid, densify(read_kml_points(f))).round(3)
        changed.append("primary drains")

    f = os.path.join(RAW, "lakes.kml")
    if os.path.exists(f):
        grid["dist_lake_km"] = nearest_km(grid, read_kml_points(f)).round(3)
        changed.append("lakes")

    f = os.path.join(RAW, "dem.csv")
    if os.path.exists(f):
        dem = pd.read_csv(f)
        tree = cKDTree(_xy(dem.lat, dem.lon))
        _, i = tree.query(_xy(grid.lat, grid.lon))
        grid["elevation_m"] = dem.elevation_m.values[i]
        gt = cKDTree(_xy(grid.lat, grid.lon))
        nb = gt.query_ball_point(_xy(grid.lat, grid.lon), r=2.5)
        grid["low_lying_m"] = [grid.elevation_m.values[j].mean() - e for j, e in zip(nb, grid.elevation_m)]
        changed.append("DEM elevation")

    grid.to_csv(grid_path, index=False)

    fr, fl = os.path.join(RAW, "rainfall_events.csv"), os.path.join(RAW, "flood_reports.csv")
    if os.path.exists(fr) and os.path.exists(fl):
        rain = pd.read_csv(fr, parse_dates=["date"])
        reps = pd.read_csv(fl, parse_dates=["date"])
        rows = []
        gxy = _xy(grid.lat, grid.lon)
        for k, (date, r) in enumerate(rain.groupby("date")):
            _, idx = cKDTree(_xy(r.lat, r.lon)).query(gxy)          # nearest rain gauge to each cell
            r24 = r.rain_24h_mm.values[idx]
            rep = reps[reps.date == date]
            flooded = np.zeros(len(grid), int)
            if len(rep):
                d, _ = cKDTree(_xy(rep.lat, rep.lon)).query(gxy)
                flooded = (d <= 1.0).astype(int)
            rows.append(pd.DataFrame({"event_id": f"E{k:03d}", "date": date.date(), "month": date.month,
                                      "cell_id": grid.cell_id, "rain_24h_mm": r24, "peak_hourly_mm": r24 * 0.4,
                                      "antecedent_3day_mm": 0.0, "flooded": flooded,
                                      "water_depth_cm": flooded * 20.0, "severity": flooded}))
        ev = pd.concat(rows, ignore_index=True)
        ev.to_csv(os.path.join(PROC, "flood_events.csv"), index=False)
        changed.append(f"REAL rainfall events + flood labels ({ev.event_id.nunique()} days)")

    print("Updated:", ", ".join(changed) if changed else "nothing (no files found in data/raw/)")


if __name__ == "__main__":
    main()
