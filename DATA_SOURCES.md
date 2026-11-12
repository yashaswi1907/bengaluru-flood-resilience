# Data Sources (Step 1: Data Collection)

| Layer (methodology box) | Used in this build | Real source to download | How to plug it in |
|---|---|---|---|
| **Historical flood locations** | 40 real BBMP/KSNDMC hotspots (geocoded, `src/reference_data.py`) | OpenCity: *Flooding locations in Bengaluru Urban* (KML, BBMP/KSRSAC) – https://data.opencity.in/dataset/flooding-locations-in-bengaluru-urban | Save as `data/raw/flood_prone_locations.kml` |
| **Lakes & drains (BBMP)** | 24 real major lakes + Bengaluru's 3 primary valleys (Koramangala-Challaghatta, Hebbal, Vrishabhavathi) | OpenCity: *Bengaluru Primary Stormwater Drains Map – 2022* (KML) – https://data.opencity.in/dataset/bengaluru-stormwater-drains-maps | `data/raw/primary_swd.kml`, `data/raw/lakes.kml` |
| **Rainfall (IMD, KSNDMC)** | 80 simulated storms (2015–2024) at realistic IMD/KSNDMC magnitudes (15–200 mm/24 h) | Daily ward rainfall: KSNDMC *Varuna Mitra* · Taluk/hobli rainfall 2017–2024: https://data.opencity.in/dataset/bengaluru-urban-annual-rainfall-taluks-and-hoblis · IMD Bengaluru 1901–2023: https://data.opencity.in/dataset/bengaluru-rainfall · Kaggle *Rainfall in India 1901–2015* (IMD sub-divisions) | `data/raw/rainfall_events.csv` (date, lat, lon, rain_24h_mm) |
| **Elevation & slope (DEM)** | Surrogate terrain (central ridge ~930 m falling to valleys ~865 m) | SRTM 30 m or Copernicus GLO-30 DEM (USGS EarthExplorer / Google Earth Engine / OpenTopography) | Sample at grid cell centres → `data/raw/dem.csv` |
| **Land use / land cover** | Surrogate imperviousness, anchored on real IT/industrial clusters (Peenya, Whitefield, ORR, Manyata, Electronic City) | ESA WorldCover 10 m or ESRI Land Cover; Bhuvan LULC (NRSC) | Replace `impervious_frac`, `land_use` columns |
| **Road network** | Road-density surrogate | OpenStreetMap via `osmnx.graph_from_place("Bengaluru")` (needed for Module 3) | Module 3 |
| **Critical infrastructure** | 18 real hospitals | OSM `amenity=hospital`, BBMP health dept. list | `HOSPITALS` in `reference_data.py` |
| **Population density** | Surrogate (decays from core) | WorldPop 100 m India; Census 2011 ward population (OpenCity) | Replace `pop_density_per_km2` |
| **Flood labels per storm** | Simulated from hydrology (drain overflow, depressions, lake overflow, hotspot bottlenecks) | BBMP control-room complaints, news reports, citizen tweets with dates & locations | `data/raw/flood_reports.csv` (date, lat, lon) |

Then run:
```bash
python src/real_data_loaders.py   # merges whatever real files exist in data/raw/
python src/train.py               # retrain on the updated data
```

**Be upfront in your report/viva:** the reference points are real, the per-storm flood outcomes and some raster layers are simulated because no public cell-level flood record for Bengaluru exists. The pipeline is built so the real layers drop in without code changes.
