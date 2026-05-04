"""
Week 4 — Controlled Experiment Set for News Sentiment → Stock Direction
=======================================================================

Runs 8 controlled experiments on top of the frozen baseline, holding split,
metric, seed, and feature set constant. Each experiment varies ONE factor
relative to the baseline so that any change in val macro F1 can be attributed
to that single factor.

Outputs (written to outputs/):
    - experiment_matrix.csv    : 8 rows x metrics + factor varied
    - per_ticker_best.csv      : per-ticker F1 for the best model
    - error_table_best.csv     : per-row validation errors with diagnostics
    - error_taxonomy.csv       : counts per error bucket
    - metric_over_time.png     : val macro F1 + accuracy across experiments

Run:
    python outputs/controlled_experiments.py
"""
from __future__ import annotations
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make the project's prepare.py importable
PROJECT_DIR = "/Users/valeriechu/Desktop/m7-news-sentiment"
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)  # so dataset.csv resolves

from prepare import load_data, evaluate, _read_and_split, _to_arrays  # noqa: E402

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.metrics import f1_score, confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

OUT_DIR = os.path.join(PROJECT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42

# ───────────────────────────────────────────────────────────────────────────
# Load data once (same split, same features for every experiment)
# ───────────────────────────────────────────────────────────────────────────
train_df, val_df, _ = _read_and_split()
X_train, y_train, feature_names = _to_arrays(train_df)
X_val, y_val, _ = _to_arrays(val_df)

print(f"Loaded data: {X_train.shape[0]} train, {X_val.shape[0]} val, "
      f"{len(feature_names)} features")
print(f"Train up-rate: {y_train.mean():.3f} | Val up-rate: {y_val.mean():.3f}\n")


# ───────────────────────────────────────────────────────────────────────────
# Define the 8 controlled experiments — each varies ONE factor from baseline
# ───────────────────────────────────────────────────────────────────────────
EXPERIMENTS = [
    # (id, factor_varied, change_from_baseline, build_fn)
    ("E1_baseline",
     "—",
     "Baseline anchor",
     lambda: Pipeline([
         ("scaler", StandardScaler()),
         ("model",  LogisticRegression(penalty=None, max_iter=2000,
                                       random_state=SEED)),
     ])),

    ("E2_class_weight",
     "Class weight",
     "class_weight='balanced' (else baseline)",
     lambda: Pipeline([
         ("scaler", StandardScaler()),
         ("model",  LogisticRegression(penalty=None, class_weight="balanced",
                                       max_iter=2000, random_state=SEED)),
     ])),

    ("E3_l2_reg",
     "Regularization",
     "L2 penalty (C=0.1) + balanced",
     lambda: Pipeline([
         ("scaler", StandardScaler()),
         ("model",  LogisticRegression(penalty="l2", C=0.1,
                                       class_weight="balanced",
                                       max_iter=2000, random_state=SEED)),
     ])),

    ("E4_robust_scaler",
     "Scaler",
     "RobustScaler instead of StandardScaler (+ balanced)",
     lambda: Pipeline([
         ("scaler", RobustScaler()),
         ("model",  LogisticRegression(penalty=None, class_weight="balanced",
                                       max_iter=2000, random_state=SEED)),
     ])),

    ("E5_poly_features",
     "Feature engineering",
     "PolynomialFeatures(deg=2, interaction_only) (+ balanced)",
     lambda: Pipeline([
         ("scaler", StandardScaler()),
         ("poly",   PolynomialFeatures(degree=2, interaction_only=True,
                                       include_bias=False)),
         ("model",  LogisticRegression(penalty="l2", C=0.1,
                                       class_weight="balanced",
                                       max_iter=2000, random_state=SEED)),
     ])),

    ("E6_random_forest",
     "Model family",
     "RandomForest (n=300, depth=6, balanced)",
     lambda: Pipeline([
         ("model", RandomForestClassifier(n_estimators=300, max_depth=6,
                                          min_samples_leaf=5,
                                          class_weight="balanced",
                                          random_state=SEED, n_jobs=-1)),
     ])),

    ("E7_hist_gbt",
     "Model family",
     "HistGradientBoosting (iter=300, depth=4, lr=0.05)",
     lambda: Pipeline([
         ("model", HistGradientBoostingClassifier(max_iter=300, max_depth=4,
                                                  learning_rate=0.05,
                                                  random_state=SEED)),
     ])),

    ("E8_extra_trees",
     "Model family",
     "ExtraTrees (n=300, depth=6, balanced) — current best",
     lambda: Pipeline([
         ("model", ExtraTreesClassifier(n_estimators=300, max_depth=6,
                                        min_samples_leaf=5,
                                        class_weight="balanced",
                                        random_state=SEED, n_jobs=-1)),
     ])),
]


# ───────────────────────────────────────────────────────────────────────────
# Run all experiments
# ───────────────────────────────────────────────────────────────────────────
rows = []
fitted = {}
print(f"{'id':18s} {'factor':22s} f1_macro  acc    rec_pos  f1_down f1_up   time")
print("-" * 95)

for exp_id, factor, change, build in EXPERIMENTS:
    t0 = time.time()
    model = build()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    f1, acc, rec_pos = evaluate(model, X_val, y_val)
    yp = model.predict(X_val)
    f1_per = f1_score(y_val, yp, average=None)

    print(f"{exp_id:18s} {factor:22s} {f1:.4f}    {acc:.4f} {rec_pos:.4f}   "
          f"{f1_per[0]:.3f}   {f1_per[1]:.3f}   {elapsed:5.2f}s")

    rows.append({
        "experiment_id":   exp_id,
        "factor_varied":   factor,
        "change_from_baseline": change,
        "val_f1_macro":    round(f1, 4),
        "val_accuracy":    round(acc, 4),
        "val_recall_pos":  round(rec_pos, 4),
        "f1_class_down":   round(float(f1_per[0]), 4),
        "f1_class_up":     round(float(f1_per[1]), 4),
        "runtime_sec":     round(elapsed, 3),
    })
    fitted[exp_id] = model


# ───────────────────────────────────────────────────────────────────────────
# Decide KEEP / DISCARD using the 0.02 noise-floor rule from program.md
# Rule: keep an experiment if its F1 is at least 0.02 above the BASELINE.
# (This is the rule documented in program.md and README.md.)
# ───────────────────────────────────────────────────────────────────────────
baseline_f1 = rows[0]["val_f1_macro"]
for r in rows:
    if r["experiment_id"] == "E1_baseline":
        r["status"] = "baseline"
    else:
        r["status"] = "keep" if (r["val_f1_macro"] - baseline_f1) >= 0.02 else "discard"

matrix = pd.DataFrame(rows)
matrix.to_csv(os.path.join(OUT_DIR, "experiment_matrix.csv"), index=False)
print(f"\nWrote {OUT_DIR}/experiment_matrix.csv")


# ───────────────────────────────────────────────────────────────────────────
# Identify best model and generate diagnostics
# ───────────────────────────────────────────────────────────────────────────
best_row = matrix.sort_values("val_f1_macro", ascending=False).iloc[0]
best_id = best_row["experiment_id"]
best_model = fitted[best_id]
print(f"\nBest model: {best_id}  (val F1 = {best_row['val_f1_macro']:.4f})\n")

# Per-ticker F1 for best model
yp_best = best_model.predict(X_val)
val_with_pred = val_df.reset_index(drop=True).copy()
val_with_pred["pred"] = yp_best
val_with_pred["true"] = y_val

per_ticker = []
for t, sub in val_with_pred.groupby("ticker"):
    yt = sub["true"].astype(int).to_numpy()
    yh = sub["pred"].astype(int).to_numpy()
    if len(np.unique(yt)) < 2:
        f1m = float("nan")
    else:
        f1m = f1_score(yt, yh, average="macro")
    per_ticker.append({
        "ticker": t,
        "n_val": len(sub),
        "up_rate": round(float(yt.mean()), 3),
        "accuracy": round(float((yt == yh).mean()), 3),
        "f1_macro": round(float(f1m), 3),
    })

per_ticker_df = pd.DataFrame(per_ticker).sort_values("f1_macro")
per_ticker_df.to_csv(os.path.join(OUT_DIR, "per_ticker_best.csv"), index=False)
print(per_ticker_df.to_string(index=False))


# ───────────────────────────────────────────────────────────────────────────
# Error table — every misclassified val row + diagnostic columns
# ───────────────────────────────────────────────────────────────────────────
try:
    probs = best_model.predict_proba(X_val)
    confidence = probs.max(axis=1)
except Exception:
    confidence = np.full(len(y_val), np.nan)

errors_mask = (yp_best != y_val)
err_df = val_with_pred[errors_mask].copy()
err_df["confidence"] = confidence[errors_mask]

# Pick diagnostic columns we know exist
diag_cols = ["date", "ticker", "true", "pred", "confidence"]
for c in ["article_count", "tone_weighted", "tone_stddev", "realized_vol5",
          "earnings_week", "tone_lag1"]:
    if c in val_with_pred.columns:
        err_df[c] = val_with_pred.loc[errors_mask, c].values
        diag_cols.append(c)

err_df = err_df[diag_cols].sort_values("confidence", ascending=False)
err_df.to_csv(os.path.join(OUT_DIR, "error_table_best.csv"), index=False)
print(f"\nWrote {len(err_df)} validation errors to error_table_best.csv")


# ───────────────────────────────────────────────────────────────────────────
# Error TAXONOMY — bucket every error
# ───────────────────────────────────────────────────────────────────────────
# Threshold helpers based on the FULL val set (not just errors)
low_news_thr   = val_with_pred["article_count"].quantile(0.25) \
    if "article_count" in val_with_pred.columns else np.nan
high_vol_thr   = val_with_pred["realized_vol5"].quantile(0.75) \
    if "realized_vol5" in val_with_pred.columns else np.nan

def bucket(row):
    """Assign each error to ONE bucket (priority order)."""
    # 1. TSLA-specific (worst per-ticker performer)
    if row["ticker"] == "TSLA":
        return "B1_TSLA_specific"
    # 2. Earnings-week
    if "earnings_week" in row and row.get("earnings_week", 0) == 1:
        return "B2_Earnings_week"
    # 3. Low-news day (article_count in bottom quartile)
    if "article_count" in row and row["article_count"] <= low_news_thr:
        return "B3_Low_news_day"
    # 4. High-volatility day (realized_vol5 above 75th pct)
    if "realized_vol5" in row and row["realized_vol5"] >= high_vol_thr:
        return "B4_High_volatility"
    # 5. Confident-wrong (model says one class with > 0.6 prob, was the other)
    if not np.isnan(row.get("confidence", np.nan)) and row["confidence"] >= 0.6:
        if row["pred"] == 1:
            return "B5_Confident_wrong_UP"
        else:
            return "B5_Confident_wrong_DOWN"
    # 6. Catch-all
    return "B6_Ambiguous_low_confidence"

err_df["bucket"] = err_df.apply(bucket, axis=1)
taxonomy = (err_df.groupby("bucket")
                  .agg(n_errors=("bucket", "count"),
                       avg_confidence=("confidence", "mean"))
                  .reset_index()
                  .sort_values("n_errors", ascending=False))
taxonomy["pct_of_errors"] = (taxonomy["n_errors"] / len(err_df) * 100).round(1)
taxonomy["avg_confidence"] = taxonomy["avg_confidence"].round(3)

taxonomy.to_csv(os.path.join(OUT_DIR, "error_taxonomy.csv"), index=False)
print("\nError Taxonomy:")
print(taxonomy.to_string(index=False))


# ───────────────────────────────────────────────────────────────────────────
# Metric-over-time plot
# ───────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

color_map = {"baseline": "#3498db", "keep": "#2ecc71", "discard": "#e74c3c"}
colors = [color_map[s] for s in matrix["status"]]

# Top: macro F1
ax1.scatter(range(len(matrix)), matrix["val_f1_macro"], c=colors, s=120,
            zorder=3, edgecolors="white", linewidth=1)
ax1.plot(range(len(matrix)), matrix["val_f1_macro"], "k--", alpha=0.25, zorder=2)

best_so_far = np.maximum.accumulate(matrix["val_f1_macro"].values)
ax1.plot(range(len(matrix)), best_so_far, color="#2ecc71", linewidth=2.5,
         label="Best so far")

ax1.axhline(0.5, ls="--", color="grey", lw=0.8, alpha=0.5,
            label="Chance (0.50)")
ax1.axhline(baseline_f1, ls=":", color="#3498db", lw=1.0, alpha=0.7,
            label=f"Baseline ({baseline_f1:.3f})")
ax1.axhline(baseline_f1 + 0.02, ls=":", color="#27ae60", lw=1.0, alpha=0.7,
            label=f"Keep threshold (+0.02)")

ax1.set_ylabel("Validation Macro F1", fontsize=12)
ax1.set_title("Controlled Experiment Set — Metric Over Time\n"
              "News Sentiment → Tech Stock Direction (Week 4)",
              fontsize=13, fontweight="bold")
ax1.legend(loc="lower right", fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.40, 0.60)

# Bottom: accuracy
ax2.scatter(range(len(matrix)), matrix["val_accuracy"], c=colors, s=120,
            zorder=3, edgecolors="white", linewidth=1)
ax2.plot(range(len(matrix)), matrix["val_accuracy"], "k--", alpha=0.25, zorder=2)
ax2.axhline(0.5, ls="--", color="grey", lw=0.8, alpha=0.5)
ax2.set_ylabel("Validation Accuracy", fontsize=12)
ax2.set_xlabel("Experiment", fontsize=12)
ax2.set_xticks(range(len(matrix)))
ax2.set_xticklabels(matrix["experiment_id"], rotation=30, ha="right", fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.40, 0.60)

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "metric_over_time.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\nWrote {plot_path}")
plt.close()

print("\nDone.")
