"""
Module 1 - Flood Intelligence Module (+ Step 4 Output & Evaluation).

  * Flood probability  : binary classifier (flooded / not flooded)
  * Flood severity     : multi-class classifier (0 none, 1 minor <15 cm, 2 moderate 15-45 cm, 3 severe >45 cm)
  * Confidence score   : 1 - normalised entropy of the severity distribution

Models compared: Logistic Regression (baseline), Random Forest, Gradient Boosting
(sklearn HistGradientBoosting; XGBoost is used automatically if installed).
Hyper-parameters tuned with RandomizedSearchCV using GroupKFold by rainfall event,
so the model is always tested on storms it has never seen. The final test set is the
most recent 20% of storms (time-based hold-out).

Run:  python src/train.py
"""
import json
import os
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve, classification_report)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import add_features, feature_columns
from reference_data import FLOOD_HOTSPOTS, LAKES, HOSPITALS

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed")
OUT_M = os.path.join(ROOT, "models")
OUT_F = os.path.join(ROOT, "outputs", "figures")
OUT_O = os.path.join(ROOT, "outputs")
for d in (OUT_M, OUT_F):
    os.makedirs(d, exist_ok=True)
SEED = 42
N_ITER = {"Logistic Regression": 4, "Random Forest": 4, "Gradient Boosting": 6, "XGBoost": 6}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})
C_BLUE, C_ORANGE, C_GREEN, C_GREY = "#2563eb", "#ea580c", "#16a34a", "#6b7280"


def load():
    grid = pd.read_csv(os.path.join(DATA, "grid_cells.csv"))
    ev = pd.read_csv(os.path.join(DATA, "flood_events.csv"), parse_dates=["date"])
    df = add_features(ev.merge(grid, on="cell_id", how="left"))
    # Data cleaning: duplicates / missing values / inconsistent values
    df = df.drop_duplicates(subset=["event_id", "cell_id"])
    num = df.select_dtypes("number").columns
    df[num] = df[num].fillna(df[num].median())
    df["rain_24h_mm"] = df.rain_24h_mm.clip(lower=0)
    return grid, df


def time_split(df, test_frac=0.2):
    order = df.groupby("event_id").date.first().sort_values()
    n_test = int(round(len(order) * test_frac))
    test_events = set(order.index[-n_test:])
    return df[~df.event_id.isin(test_events)], df[df.event_id.isin(test_events)]


def candidate_models():
    models = {
        "Logistic Regression": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
            {"logisticregression__C": [0.01, 0.1, 1, 10]}),
        "Random Forest": (
            RandomForestClassifier(n_jobs=-1, random_state=SEED, class_weight="balanced_subsample"),
            {"n_estimators": [150], "max_depth": [10, 16, None], "min_samples_leaf": [10, 20],
             "max_features": ["sqrt", 0.4]}),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = (XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1),
                             {"n_estimators": [200, 400], "max_depth": [3, 5, 7], "learning_rate": [0.03, 0.1],
                              "subsample": [0.8, 1.0], "colsample_bytree": [0.6, 0.9]})
    except ImportError:
        models["Gradient Boosting"] = (
            HistGradientBoostingClassifier(random_state=SEED),
            {"learning_rate": [0.03, 0.06, 0.1], "max_depth": [4, 6, None], "max_leaf_nodes": [15, 31, 63],
             "min_samples_leaf": [20, 50, 100], "l2_regularization": [0, 0.5, 1.0]})
    return models


def metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    return {"accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat), "f1": f1_score(y, yhat), "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p), "threshold": thr}


def best_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.clip(pr + rc, 1e-9, None)
    return float(th[np.nanargmax(f1[:-1])])


def scenario_frame(grid, rain, peak_frac=0.4, antecedent=30, month=9):
    s = grid.copy()
    s["rain_24h_mm"], s["peak_hourly_mm"] = float(rain), float(rain) * peak_frac
    s["antecedent_3day_mm"], s["month"] = float(antecedent), month
    return add_features(s)


def confidence_from_proba(P):
    P = np.clip(P, 1e-9, 1)
    ent = -(P * np.log(P)).sum(axis=1) / np.log(P.shape[1])
    return 1 - ent


def main():
    t0 = time.time()
    grid, df = load()
    feats = feature_columns(include_history=True)
    train, test = time_split(df)
    Xtr, ytr, gtr = train[feats], train.flooded.values, train.event_id.values
    Xte, yte = test[feats], test.flooded.values
    print(f"train rows {len(train)} ({train.event_id.nunique()} storms) | test rows {len(test)} "
          f"({test.event_id.nunique()} most recent storms) | flood rate {df.flooded.mean():.3f}")

    cv = GroupKFold(n_splits=3)
    results, fitted, oof = {}, {}, {}
    for name, (est, grid_params) in candidate_models().items():
        n_iter = min(N_ITER.get(name, 6), int(np.prod([len(v) for v in grid_params.values()])))
        search = RandomizedSearchCV(est, grid_params, n_iter=n_iter, scoring="average_precision", cv=cv,
                                    random_state=SEED, n_jobs=1)
        print(f"tuning {name} ...", flush=True)
        search.fit(Xtr, ytr, groups=gtr)
        best = search.best_estimator_
        oof[name] = cross_val_predict(best, Xtr, ytr, groups=gtr, cv=cv, method="predict_proba")[:, 1]
        thr = best_f1_threshold(ytr, oof[name])
        p_te = best.predict_proba(Xte)[:, 1]
        m = metrics(yte, p_te, thr)
        m["cv_pr_auc"] = float(search.best_score_)
        m["best_params"] = {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in search.best_params_.items()}
        results[name], fitted[name] = m, (best, thr, p_te)
        print(f"{name:20s} CV PR-AUC {search.best_score_:.3f} | test ROC-AUC {m['roc_auc']:.3f} "
              f"PR-AUC {m['pr_auc']:.3f} F1 {m['f1']:.3f} P {m['precision']:.3f} R {m['recall']:.3f}")

    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    model, thr, p_te = fitted[best_name]
    print(f"\nBest model: {best_name}")

    # ---------- Sensitivity / ablation: no historical-flood features ----------
    feats_nohist = feature_columns(include_history=False)
    abl = HistGradientBoostingClassifier(random_state=SEED, learning_rate=0.06, max_leaf_nodes=31,
                                         min_samples_leaf=50).fit(train[feats_nohist], ytr)
    p_abl = abl.predict_proba(test[feats_nohist])[:, 1]
    ablation = {"pr_auc_without_history": average_precision_score(yte, p_abl),
                "roc_auc_without_history": roc_auc_score(yte, p_abl)}

    # ---------- Severity model ----------
    sev = HistGradientBoostingClassifier(random_state=SEED, learning_rate=0.06, max_leaf_nodes=31,
                                         min_samples_leaf=50, class_weight="balanced")
    sev.fit(Xtr, train.severity.values)
    sev_pred = sev.predict(Xte)
    sev_report = classification_report(test.severity.values, sev_pred, output_dict=True, zero_division=0,
                                       target_names=["none", "minor", "moderate", "severe"])
    sev_cm = confusion_matrix(test.severity.values, sev_pred)
    print("Severity macro-F1:", round(sev_report["macro avg"]["f1-score"], 3))

    # ---------- Permutation importance (on test sample) ----------
    samp = test.sample(min(8000, len(test)), random_state=SEED)
    pi = permutation_importance(model, samp[feats], samp.flooded, scoring="average_precision",
                                n_repeats=5, random_state=SEED, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)

    # ---------- Scenario risk scores for every cell (input for Modules 2-6) ----------
    scen_rows = []
    for rain in (50, 100, 150, 200):
        s = scenario_frame(grid, rain)
        prob = model.predict_proba(s[feats])[:, 1]
        P = sev.predict_proba(s[feats])
        scen_rows.append(pd.DataFrame({
            "cell_id": grid.cell_id, "lat": grid.lat, "lon": grid.lon, "zone": grid.zone,
            "scenario_rain_24h_mm": rain, "flood_probability": prob.round(4),
            "flood_predicted": (prob >= thr).astype(int), "severity_class": P.argmax(axis=1),
            "expected_depth_class": (P * np.arange(4)).sum(axis=1).round(3),
            "confidence_score": confidence_from_proba(P).round(3),
            "pop_density_per_km2": grid.pop_density_per_km2, "dist_hospital_km": grid.dist_hospital_km,
        }))
    scen = pd.concat(scen_rows, ignore_index=True)
    scen["risk_level"] = pd.cut(scen.flood_probability, [-0.01, 0.1, 0.3, 0.6, 1.0],
                                labels=["Low", "Moderate", "High", "Very High"])
    scen.to_csv(os.path.join(OUT_O, "cell_risk_scores.csv"), index=False)

    # ---------- Validation against real BBMP/KSNDMC hotspot list ----------
    s100 = scen[scen.scenario_rain_24h_mm == 100].reset_index(drop=True)
    top10 = s100.flood_probability >= s100.flood_probability.quantile(0.9)
    from build_dataset import to_xy
    hx, hy = to_xy(np.array([h[1] for h in FLOOD_HOTSPOTS]), np.array([h[2] for h in FLOOD_HOTSPOTS]))
    cx, cy = grid.x_km.values, grid.y_km.values
    nearest = [int(np.argmin((cx - a) ** 2 + (cy - b) ** 2)) for a, b in zip(hx, hy)]
    hotspot_hit = float(np.mean([top10[i] for i in nearest]))
    # same check with the no-history model (purely physical features)
    s = scenario_frame(grid, 100)
    p_nh = abl.predict_proba(s[feats_nohist])[:, 1]
    hotspot_hit_nohist = float(np.mean([p_nh[i] >= np.quantile(p_nh, 0.9) for i in nearest]))
    zone_summary = (s100.groupby("zone").agg(mean_flood_prob=("flood_probability", "mean"),
                                             high_risk_cells=("flood_predicted", "sum"))
                    .sort_values("mean_flood_prob", ascending=False).round(3))
    print("Real hotspots in top-10% risk cells (100 mm):", round(hotspot_hit, 3),
          "| without history features:", round(hotspot_hit_nohist, 3))

    # ---------- Save model bundle ----------
    joblib.dump({"flood_model": model, "severity_model": sev, "threshold": thr, "features": feats,
                 "model_name": best_name}, os.path.join(OUT_M, "flood_model.joblib"))

    summary = {"best_model": best_name, "test_metrics": results, "ablation": ablation,
               "severity_report": sev_report, "severity_confusion": sev_cm.tolist(),
               "hotspot_recall_top10pct_100mm": hotspot_hit,
               "hotspot_recall_top10pct_100mm_no_history": hotspot_hit_nohist,
               "zone_summary_100mm": zone_summary.reset_index().to_dict(orient="records"),
               "feature_importance": imp.round(5).to_dict(),
               "n_train_rows": len(train), "n_test_rows": len(test)}
    with open(os.path.join(OUT_O, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    pd.DataFrame(results).T.drop(columns=["best_params"]).round(4).to_csv(os.path.join(OUT_O, "model_comparison.csv"))

    # ======================= Figures =======================
    # 1. ROC + PR curves
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    for (name, (_, _, p)), c in zip(fitted.items(), [C_GREY, C_GREEN, C_BLUE, C_ORANGE]):
        fpr, tpr, _ = roc_curve(yte, p)
        ax[0].plot(fpr, tpr, color=c, lw=1.8, label=f"{name} (AUC {roc_auc_score(yte, p):.3f})")
        pr, rc, _ = precision_recall_curve(yte, p)
        ax[1].plot(rc, pr, color=c, lw=1.8, label=f"{name} (AP {average_precision_score(yte, p):.3f})")
    ax[0].plot([0, 1], [0, 1], ls=":", c="#9ca3af")
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (unseen storms)")
    ax[1].axhline(yte.mean(), ls=":", c="#9ca3af")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")
    for a in ax:
        a.legend(frameon=False, fontsize=7.5, loc="lower right" if a is ax[0] else "upper right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_F, "01_roc_pr_curves.png")); plt.close(fig)

    # 2. Confusion matrices
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    cm = confusion_matrix(yte, (p_te >= thr).astype(int))
    for a, M, labels, title in [(ax[0], cm, ["No flood", "Flood"], f"Flood / no flood ({best_name})"),
                                (ax[1], sev_cm, ["None", "Minor", "Moderate", "Severe"], "Severity class")]:
        a.imshow(M, cmap="Blues")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                a.text(j, i, f"{M[i, j]:,}", ha="center", va="center", fontsize=8,
                       color="white" if M[i, j] > M.max() * 0.5 else "#111827")
        a.set_xticks(range(len(labels)), labels); a.set_yticks(range(len(labels)), labels)
        a.set(xlabel="Predicted", ylabel="Actual", title=title)
        a.spines[:].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_F, "02_confusion_matrices.png")); plt.close(fig)

    # 3. Feature importance
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(top.index, top.values, color=C_BLUE)
    ax.set(xlabel="Drop in PR-AUC when feature is shuffled", title="What drives flood risk? (permutation importance)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_F, "03_feature_importance.png")); plt.close(fig)

    # 4. Risk maps for 4 rainfall scenarios
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
    lk = np.array([(l[1], l[2]) for l in LAKES]); hs = np.array([(h[1], h[2]) for h in FLOOD_HOTSPOTS])
    hp = np.array([(h[1], h[2]) for h in HOSPITALS])
    for a, rain in zip(axes, (50, 100, 150, 200)):
        s = scen[scen.scenario_rain_24h_mm == rain]
        sc = a.scatter(s.lon, s.lat, c=s.flood_probability, cmap="YlOrRd", vmin=0, vmax=1, s=26, marker="s",
                       linewidths=0)
        a.scatter(lk[:, 1], lk[:, 0], s=22, c="#0ea5e9", edgecolors="white", linewidths=0.5, label="Lake")
        a.scatter(hs[:, 1], hs[:, 0], s=14, facecolors="none", edgecolors="#111827", linewidths=0.7,
                  label="BBMP/KSNDMC hotspot")
        a.scatter(hp[:, 1], hp[:, 0], s=22, marker="P", c="#16a34a", label="Hospital")
        a.set(title=f"{rain} mm in 24 h", xlabel="Longitude"); a.set_aspect("equal")
    axes[0].set_ylabel("Latitude"); axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    cb = fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.01); cb.set_label("Flood probability")
    fig.suptitle("Bengaluru flood-risk map by rainfall scenario (~1 km cells)", fontsize=11)
    fig.savefig(os.path.join(OUT_F, "04_risk_maps_scenarios.png"), bbox_inches="tight"); plt.close(fig)

    # 5. Flood probability vs rainfall (calibration-style curve)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = pd.cut(test.rain_24h_mm, [0, 20, 40, 60, 80, 100, 130, 160, 250])
    t = pd.DataFrame({"bin": bins, "obs": yte, "pred": p_te}).groupby("bin", observed=True).mean()
    xs = range(len(t))
    ax.plot(xs, t.obs, "o-", c=C_GREY, label="Observed flood rate")
    ax.plot(xs, t.pred, "o-", c=C_BLUE, label="Mean predicted probability")
    ax.set_xticks(list(xs), [str(i) for i in t.index], rotation=30, fontsize=7)
    ax.set(xlabel="24 h rainfall (mm)", ylabel="Share of cells flooded", title="Flood rate rises sharply above ~100 mm")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_F, "05_rainfall_response.png")); plt.close(fig)

    # 6. Zone ranking at 100 mm
    fig, ax = plt.subplots(figsize=(6, 3.4))
    zs = zone_summary.sort_values("mean_flood_prob")
    ax.barh(zs.index, zs.mean_flood_prob, color=C_ORANGE)
    ax.set(xlabel="Mean flood probability (100 mm scenario)", title="Zone vulnerability ranking")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_F, "06_zone_ranking.png")); plt.close(fig)

    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
