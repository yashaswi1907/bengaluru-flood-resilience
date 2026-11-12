"""Feature engineering shared by training and prediction (Step 2: Feature Engineering)."""
import numpy as np
import pandas as pd

LAND_USES = ["Commercial", "Residential", "Peri-urban", "IT Park", "Industrial", "Green/Open"]

STATIC = [
    "elevation_m", "slope_pct", "low_lying_m", "dist_primary_drain_km", "drain_density_km_per_km2",
    "drain_capacity_mm_hr", "drain_silt_index", "dist_lake_km", "nearest_lake_area_ha",
    "lakes_within_2km", "lake_area_within_3km_ha", "lake_encroachment_index", "impervious_frac",
    "green_cover_frac", "pop_density_per_km2", "road_density_km_per_km2",
]
HISTORICAL = ["dist_known_hotspot_km", "hist_flood_count_2015_24"]
DYNAMIC = ["rain_24h_mm", "peak_hourly_mm", "antecedent_3day_mm", "month"]
DERIVED = ["excess_over_drain_mm_hr", "runoff_index", "lake_overflow_index", "monsoon_peak"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # how much the peak-hour rain exceeds what the drains can carry (silt reduces capacity)
    df["excess_over_drain_mm_hr"] = (df.peak_hourly_mm * (0.35 + 0.65 * df.impervious_frac)
                                     - df.drain_capacity_mm_hr * (1 - 0.45 * df.drain_silt_index))
    df["runoff_index"] = df.rain_24h_mm * df.impervious_frac + 0.3 * df.antecedent_3day_mm * df.impervious_frac
    df["lake_overflow_index"] = (np.clip(df.rain_24h_mm - 60, 0, None) * np.log1p(df.nearest_lake_area_ha)
                                 * np.exp(-df.dist_lake_km))
    df["monsoon_peak"] = df.month.isin([8, 9, 10]).astype(int)
    for lu in LAND_USES:
        df[f"lu_{lu}"] = (df.land_use == lu).astype(int)
    return df


def feature_columns(include_history=True):
    cols = STATIC + DYNAMIC + DERIVED + [f"lu_{lu}" for lu in LAND_USES]
    return cols + HISTORICAL if include_history else cols
