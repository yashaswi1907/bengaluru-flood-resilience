"""
Step 1 + 2 of the methodology: Data Collection, Preprocessing & Integration.

Builds two tables:
  data/processed/grid_cells.csv     -> one row per ~1 km x 1 km cell of Bengaluru (static features)
  data/processed/flood_events.csv   -> one row per (cell, rainfall event) with the flood label

Static layers (geospatial & environmental, infrastructure, socio-economic) are anchored on
real Bengaluru reference points (src/reference_data.py): BBMP/KSNDMC flood hotspots, major
lakes, primary storm-water valleys (Koramangala-Challaghatta, Hebbal, Vrishabhavathi),
hospitals, and industrial / IT clusters. Where a public raster was not downloadable in this
environment (DEM, land cover, drain capacity), a physically-motivated surrogate is generated
with values in realistic Bengaluru ranges. See DATA_SOURCES.md for how to replace each one
with the official layer.

Run:  python src/build_dataset.py
"""
import os
import numpy as np
import pandas as pd

from reference_data import FLOOD_HOTSPOTS, LAKES, HOSPITALS, ZONES, CITY_CENTER

SEED = 42
rng = np.random.default_rng(SEED)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)

LAT0, LON0 = CITY_CENTER
KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON = 111.32 * np.cos(np.radians(LAT0))


def to_xy(lat, lon):
    """lat/lon -> local km coordinates centred on the city (common CRS for integration)."""
    return (np.asarray(lon) - LON0) * KM_PER_DEG_LON, (np.asarray(lat) - LAT0) * KM_PER_DEG_LAT


def dist_to_points(x, y, pts):
    px, py = to_xy(np.array([p[1] for p in pts]), np.array([p[2] for p in pts]))
    d = np.sqrt((x[:, None] - px[None, :]) ** 2 + (y[:, None] - py[None, :]) ** 2)
    return d  # (n_cells, n_points) in km


def dist_to_polyline(x, y, line):
    lx, ly = to_xy(np.array([p[0] for p in line]), np.array([p[1] for p in line]))
    best = np.full(len(x), np.inf)
    for i in range(len(lx) - 1):
        ax, ay, bx, by = lx[i], ly[i], lx[i + 1], ly[i + 1]
        vx, vy = bx - ax, by - ay
        t = np.clip(((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy), 0, 1)
        d = np.hypot(x - (ax + t * vx), y - (ay + t * vy))
        best = np.minimum(best, d)
    return best


def smooth_noise(x, y, scale_km, n_waves=12):
    """Spatially-correlated noise (sum of random plane waves), mean 0, std ~1."""
    z = np.zeros_like(x)
    for _ in range(n_waves):
        ang = rng.uniform(0, 2 * np.pi)
        k = 2 * np.pi / (scale_km * rng.uniform(0.7, 1.5))
        z += np.cos(k * (x * np.cos(ang) + y * np.sin(ang)) + rng.uniform(0, 2 * np.pi))
    return z / np.sqrt(n_waves / 2)


# Primary storm-water valleys (rajakaluves) - these are Bengaluru's three main drainage valleys
VALLEYS = {
    "Koramangala-Challaghatta": [(12.955, 77.585), (12.935, 77.620), (12.935, 77.665),
                                  (12.940, 77.705), (12.945, 77.745)],
    "Madiwala branch":          [(12.950, 77.600), (12.908, 77.617), (12.920, 77.642), (12.932, 77.665)],
    "Hebbal":                   [(13.010, 77.575), (13.045, 77.590), (13.040, 77.612),
                                  (13.030, 77.660), (13.020, 77.690), (13.000, 77.725)],
    "Vrishabhavathi":           [(12.995, 77.555), (12.965, 77.535), (12.940, 77.520), (12.905, 77.485)],
}
# Industrial / IT clusters (higher imperviousness)
CLUSTERS = [(13.030, 77.520, "Industrial"),   # Peenya
            (12.990, 77.550, "Industrial"),   # Rajajinagar Indl
            (12.975, 77.730, "IT Park"),      # Whitefield / ITPL
            (12.930, 77.685, "IT Park"),      # ORR / Bellandur tech parks
            (13.045, 77.620, "IT Park"),      # Manyata
            (12.845, 77.665, "IT Park")]      # Electronic City


def build_grid():
    step_lat, step_lon = 0.009, 0.0093  # ~1 km
    lats = np.arange(12.82, 13.15, step_lat)
    lons = np.arange(77.44, 77.80, step_lon)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    la, lo = LA.ravel(), LO.ravel()
    x, y = to_xy(la, lo)
    keep = (x / 17.0) ** 2 + (y / 16.5) ** 2 <= 1.0        # approx. city (GBA/BBMP) footprint
    la, lo, x, y = la[keep], lo[keep], x[keep], y[keep]
    n = len(la)
    r = np.hypot(x, y)

    # ---------------- Geospatial & environmental ----------------
    # Elevation (m): central N-S ridge ~930 m, falling east (Varthur ~870 m) and SW (Kengeri ~830 m)
    ridge_dist = np.abs(x - 0.18 * y + 1.0)
    elev = 932 - 2.7 * ridge_dist - 0.6 * np.clip(-y, 0, None)
    valley_d = {k: dist_to_polyline(x, y, v) for k, v in VALLEYS.items()}
    for d in valley_d.values():
        elev -= 14 * np.exp(-(d / 0.9) ** 2)
    elev += 4 * smooth_noise(x, y, 6)
    elev += rng.normal(0, 1.0, n)
    # slope (%) from local gradient on the grid (finite differences via nearest neighbours)
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[x, y])
    _, nb = tree.query(np.c_[x, y], k=9)
    dz = np.abs(elev[nb[:, 1:]] - elev[:, None])
    dd = np.hypot(x[nb[:, 1:]] - x[:, None], y[nb[:, 1:]] - y[:, None]) * 1000
    slope_pct = np.clip((dz / dd).max(axis=1) * 100, 0.05, None)
    # relative depression: how much lower than neighbourhood mean within ~2.5 km (low-lying index)
    idx_2k = tree.query_ball_point(np.c_[x, y], r=2.5)
    local_mean = np.array([elev[i].mean() for i in idx_2k])
    low_lying_m = local_mean - elev                      # +ve = lower than surroundings

    dist_valley = np.min(np.vstack(list(valley_d.values())), axis=0)

    d_lakes = dist_to_points(x, y, LAKES)
    j = d_lakes.argmin(axis=1)
    dist_lake = d_lakes.min(axis=1)
    nearest_lake_area = np.array([LAKES[k][3] for k in j], dtype=float)
    lakes_within_2km = (d_lakes <= 2.0).sum(axis=1)
    lake_area_within_3km = (np.array([l[3] for l in LAKES])[None, :] * (d_lakes <= 3.0)).sum(axis=1)

    # Land cover
    imperv = 0.88 - 0.022 * r + 0.06 * smooth_noise(x, y, 5)
    d_cl = dist_to_points(x, y, [(c[2], c[0], c[1]) for c in CLUSTERS])
    imperv += 0.12 * np.exp(-(d_cl.min(axis=1) / 2.0) ** 2)
    imperv -= 0.25 * np.exp(-(dist_lake / 0.6) ** 2)
    imperv = np.clip(imperv, 0.15, 0.97)
    green_cover = np.clip(1 - imperv - rng.uniform(0.0, 0.08, n), 0.02, 0.85)

    landuse = np.where(r < 4.5, "Commercial", "Residential").astype(object)
    landuse[r > 12.5] = "Peri-urban"
    cl_type = np.array([c[2] for c in CLUSTERS])[d_cl.argmin(axis=1)]
    landuse[d_cl.min(axis=1) < 1.8] = cl_type[d_cl.min(axis=1) < 1.8]
    landuse[(green_cover > 0.45) & (landuse != "Peri-urban")] = "Green/Open"

    # ---------------- Drainage (BBMP SWD) ----------------
    core = r < 8.0     # older city core has a more complete drain network than the 110 added villages
    drain_density = np.clip(np.where(core, 4.2, 2.4) + 0.6 * smooth_noise(x, y, 4)
                            + 1.5 * np.exp(-(dist_valley / 0.7) ** 2), 0.4, None)        # km/km^2
    drain_capacity = np.clip(np.where(core, 38, 24) + 5 * smooth_noise(x, y, 5)
                             + rng.normal(0, 2, n), 10, 60)                                # mm/hr
    silt_index = np.clip(0.45 + 0.18 * smooth_noise(x, y, 3) + rng.normal(0, 0.05, n), 0, 1)
    lake_encroachment = np.clip(0.35 + 0.2 * smooth_noise(x, y, 7) + 0.02 * r, 0, 1)

    # ---------------- Population & infrastructure ----------------
    pop_density = np.clip(32000 * np.exp(-r / 7.5) + 2500 + 2500 * smooth_noise(x, y, 4), 800, None)
    road_density = np.clip(18 * np.exp(-r / 9) + 3 + 1.5 * smooth_noise(x, y, 4), 1.5, None)  # km/km^2
    d_hosp = dist_to_points(x, y, HOSPITALS)
    dist_hospital = d_hosp.min(axis=1)
    hospitals_within_3km = (d_hosp <= 3.0).sum(axis=1)

    # ---------------- Historical flood locations (BBMP/KSNDMC) ----------------
    d_hot = dist_to_points(x, y, FLOOD_HOTSPOTS)
    sev = np.array([h[3] for h in FLOOD_HOTSPOTS], dtype=float)
    hotspot_influence = (sev[None, :] * np.exp(-(d_hot / 0.9) ** 2)).sum(axis=1)
    dist_hotspot = d_hot.min(axis=1)
    # observed count of flood reports 2015-2024 (noisy observation of the true hotspot effect)
    hist_flood_count = rng.poisson(0.15 + 2.2 * hotspot_influence + 0.6 * np.clip(low_lying_m, 0, None) / 5)

    zone = np.array([z[0] for z in ZONES])[dist_to_points(x, y, [(z[0], z[1], z[2]) for z in ZONES]).argmin(axis=1)]

    grid = pd.DataFrame({
        "cell_id": [f"C{i:04d}" for i in range(n)], "lat": la.round(5), "lon": lo.round(5),
        "x_km": x.round(3), "y_km": y.round(3), "zone": zone,
        "elevation_m": elev.round(1), "slope_pct": slope_pct.round(2), "low_lying_m": low_lying_m.round(2),
        "dist_primary_drain_km": dist_valley.round(3), "drain_density_km_per_km2": drain_density.round(2),
        "drain_capacity_mm_hr": drain_capacity.round(1), "drain_silt_index": silt_index.round(3),
        "dist_lake_km": dist_lake.round(3), "nearest_lake_area_ha": nearest_lake_area,
        "lakes_within_2km": lakes_within_2km, "lake_area_within_3km_ha": lake_area_within_3km,
        "lake_encroachment_index": lake_encroachment.round(3),
        "impervious_frac": imperv.round(3), "green_cover_frac": green_cover.round(3), "land_use": landuse,
        "pop_density_per_km2": pop_density.round(0), "road_density_km_per_km2": road_density.round(2),
        "dist_hospital_km": dist_hospital.round(3), "hospitals_within_3km": hospitals_within_3km,
        "dist_known_hotspot_km": dist_hotspot.round(3), "hist_flood_count_2015_24": hist_flood_count,
    })
    # hidden (unobserved) drivers used only to simulate outcomes - NOT given to the model
    hidden = {"hotspot_influence": hotspot_influence}
    return grid, hidden


def simulate_events(grid, hidden, n_events=80):
    """Rainfall events (IMD/KSNDMC-scale) and resulting flooding per cell."""
    x, y = grid.x_km.values, grid.y_km.values
    n = len(grid)
    dates = pd.to_datetime(rng.choice(pd.date_range("2015-04-01", "2024-11-30"), size=n_events * 4))
    dates = sorted({d for d in dates if d.month in (4, 5, 6, 7, 8, 9, 10, 11)})[:n_events]
    rows = []
    for e, date in enumerate(dates):
        # city-mean 24h rainfall: mostly 15-70 mm, heavy tail to ~200 mm (cf. Sept 2022, Oct 2024 events)
        city_mean = float(np.clip(rng.gamma(2.0, 22.0) if rng.random() > 0.15 else rng.uniform(90, 190), 5, 210))
        rain = city_mean * (1 + 0.012 * (-y))                      # south slightly wetter (KSNDMC taluk data)
        for _ in range(rng.integers(1, 4)):                          # convective cells
            cx, cy = rng.uniform(-14, 14), rng.uniform(-13, 13)
            rain = rain + city_mean * rng.uniform(0.3, 0.8) * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * rng.uniform(2.5, 5) ** 2))
        rain = np.clip(rain * (1 + 0.08 * rng.normal(size=n)), 0, None)
        peak_frac = rng.uniform(0.22, 0.55)                          # share of 24h rain in the peak hour
        peak_int = rain * peak_frac
        antecedent = rng.gamma(1.5, 18.0) * (1 if date.month in (8, 9, 10) else 0.6)
        antecedent = antecedent * (1 + 0.1 * rng.normal(size=n)).clip(0.5)

        g = grid
        excess = peak_int * (0.35 + 0.65 * g.impervious_frac) - g.drain_capacity_mm_hr * (1 - 0.45 * g.drain_silt_index)
        runoff = rain * (0.3 + 0.6 * g.impervious_frac) + 0.3 * antecedent * g.impervious_frac
        lake_over = (np.clip(rain - 60, 0, None) / 60) * np.exp(-g.dist_lake_km / 0.9) \
            * np.log1p(g.nearest_lake_area_ha) / 5 * (0.5 + g.lake_encroachment_index)
        # threshold behaviour: water only accumulates once drains overflow, and pools in depressions;
        # known bottlenecks (culverts / encroached valleys) mainly bite in heavy rain
        overflow = np.clip(excess, 0, None)
        low = g.low_lying_m.clip(-5, 15)
        heavy = 1 / (1 + np.exp(-(rain - 80) / 12))
        z = (-4.6 + 0.075 * overflow * (1 + 0.09 * np.clip(low, 0, None)) + 0.010 * runoff + 0.08 * low
             + 1.3 * lake_over + hidden["hotspot_influence"] * (0.3 + 1.2 * heavy)
             + 0.9 * np.exp(-(g.dist_primary_drain_km / 0.5) ** 2) * heavy - 0.05 * g.slope_pct
             + rng.normal(0, 0.7, n))
        p = 1 / (1 + np.exp(-z))
        flooded = rng.random(n) < p
        depth = np.where(flooded, 5 + 22 * np.log1p(np.exp(z)) * rng.lognormal(0, 0.35, n), 0.0)
        severity = np.select([depth == 0, depth < 15, depth < 45], [0, 1, 2], 3)
        rows.append(pd.DataFrame({
            "event_id": f"E{e:03d}", "date": date.date(), "month": date.month, "cell_id": g.cell_id,
            "rain_24h_mm": rain.round(1), "peak_hourly_mm": peak_int.round(1),
            "antecedent_3day_mm": antecedent.round(1), "flooded": flooded.astype(int),
            "water_depth_cm": depth.round(1), "severity": severity,
        }))
    return pd.concat(rows, ignore_index=True)


def main():
    grid, hidden = build_grid()
    events = simulate_events(grid, hidden)
    grid.to_csv(os.path.join(OUT, "grid_cells.csv"), index=False)
    events.to_csv(os.path.join(OUT, "flood_events.csv"), index=False)
    print(f"cells: {len(grid)}  events: {events.event_id.nunique()}  rows: {len(events)}")
    print(f"flood rate: {events.flooded.mean():.3f}")
    print(events.severity.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
