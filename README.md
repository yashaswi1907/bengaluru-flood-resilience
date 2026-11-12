# Bengaluru Flood Resilience & Mobility Decision System — Module 1: Flood Intelligence

Covers methodology **Step 1 (Data Collection)**, **Step 2 (Preprocessing & Integration)**, **Module 1 (Flood Intelligence: RF / Gradient Boosting)** and **Step 4 (Model Evaluation & Validation)**, plus a what-if function that Modules 2–7 can call.

## Quick start
```bash
pip install -r requirements.txt
python src/build_dataset.py      # Step 1+2  -> data/processed/*.csv
python src/train.py              # Module 1  -> models/, outputs/  (~2 min)
python src/predict.py --rain 150 --drain-upgrade 30 --desilt     # what-if scenario
```
Or open `Bengaluru_Flood_Intelligence.ipynb` and run all cells (it does the same with explanations and charts).

## Dataset
* **881 grid cells** (~1 km × 1 km) covering the city, **80 storms (2015–2024)** → **70,480 rows**
* 26 features per cell: elevation, slope, low-lying depth, distance to primary drain, drain density/capacity/silt, lake distance/area/encroachment, impervious & green cover, land use, population density, road density, hospital access, historical flood count
* Per-storm features: 24 h rainfall, peak hourly rainfall, 3-day antecedent rainfall, month
* Targets: `flooded` (0/1) and `severity` (0 none · 1 minor <15 cm · 2 moderate 15–45 cm · 3 severe >45 cm)
* Real anchors: 40 BBMP/KSNDMC flood hotspots, 24 lakes, 3 primary valleys, 18 hospitals. Simulated: terrain surrogate, drain capacity, per-storm outcomes. See **DATA_SOURCES.md** for what's real, what's simulated, and how to swap in downloaded files.

## Results (test set = 16 most recent storms the model never saw)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.915 | 0.560 | 0.604 | 0.582 | 0.890 | 0.605 |
| Random Forest | 0.920 | 0.593 | 0.574 | 0.583 | 0.890 | 0.615 |
| **Gradient Boosting (best)** | 0.915 | 0.558 | 0.612 | **0.584** | **0.891** | **0.621** |

* Only ~8.5% of cell-storms flood, so accuracy alone is misleading. **PR-AUC and F1 are the main metrics** (a model that always says "no flood" scores 91.5% accuracy).
* Severity (4-class) macro-F1: **0.544**
* **Validation against real locations:** 67.5% of the BBMP/KSNDMC hotspots fall inside the model's top 10% risk cells at 100 mm rainfall (random = 10%).
* **Sensitivity:** without the historical-flood features, PR-AUC holds at 0.607 and hotspot recall drops to 42.5%, so the physical features carry most of the signal.
* **Top drivers:** rain in excess of drain capacity, low-lying terrain, proximity to known hotspots, distance to primary drains.
* **Zone ranking (100 mm):** Mahadevapura and Bommanahalli are most at risk, which matches BBMP's own reporting.
* **What-if:** at 150 mm, a 30% drain-capacity upgrade plus desilting cuts flooded cells from 396 to 160.

## Files
```
src/reference_data.py     real hotspots, lakes, hospitals, zones
src/build_dataset.py      Step 1+2: grid, layers, storms, labels
src/features.py           feature engineering
src/train.py              Module 1 training, tuning, evaluation, figures
src/predict.py            FloodPredictor: what-if scenarios & interventions (for Modules 5-7)
src/real_data_loaders.py  merge real KML/CSV downloads from data/raw/
data/processed/           grid_cells.csv, flood_events.csv
models/flood_model.joblib flood + severity models, threshold, feature list
outputs/cell_risk_scores.csv   hand-off to Modules 2-7 (risk per cell at 50/100/150/200 mm)
outputs/metrics.json, model_comparison.csv, figures/*.png
PROMPTS.md                ready-to-use prompts for building the remaining modules
```

## Methodology notes (for the report / viva)
* **No leakage:** cross-validation uses GroupKFold by storm, and the test set is the most recent storms (time-based split).
* **Threshold:** chosen to maximise F1 on out-of-fold predictions, not on the test set.
* **Confidence score:** 1 − normalised entropy of the severity probabilities (1 = model is sure).
* **XGBoost:** `train.py` uses it automatically if `pip install xgboost` is available; otherwise it uses sklearn's HistGradientBoosting (same algorithm family).
