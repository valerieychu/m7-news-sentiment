"""
Goal #3 — Feature Importance Analysis
======================================
Identify whether any specific sentiment dimension is more predictive than
"overall tone alone" (tone_weighted / tone_avg).

Method
------
1. Train the current best model (ExtraTrees, same hyperparams as model.py)
   on the locked train split.
2. Extract (a) built-in mean-decrease-impurity importances and
            (b) permutation importances on the val split (more reliable for
               correlated features).
3. Group every feature into one of five buckets:
       overall_tone    – tone_weighted, tone_avg
       sentiment_dim   – all other tone/sentiment columns
       price_vol       – OHLCV, daily_return, realized_vol5
       news_volume     – article_count, unique_source_count,
                         positive/negative/neutral article counts,
                         avg_activity_density
       earnings        – earnings_week
4. Print ranked tables and save two figures:
       5_feature_importance_bar_2.png  – all features, both importance methods
       5_sentiment_importance_2.png    – sentiment features zoomed in, with
                                     overall-tone benchmark line
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

# ── path so we can import prepare.py from the same directory ──────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

from prepare import load_data

# ── Feature groupings ──────────────────────────────────────────────────────
OVERALL_TONE = ["tone_weighted", "tone_avg"]

SENTIMENT_DIMS = [
    "tone_positive_weighted",
    "tone_negative_weighted",
    "tone_polarity_weighted",
    "tone_stddev",
    "tone_lag1", "tone_lag2", "tone_lag3",
    "polarity_lag1", "polarity_lag2", "polarity_lag3",
    "tone_rolling5",
    "tone_momentum",
    "tone_vol5",
]

PRICE_VOL = [
    "open", "high", "low", "close", "volume",
    "daily_return", "realized_vol5",
]

NEWS_VOL = [
    "article_count", "unique_source_count",
    "positive_article_count", "negative_article_count", "neutral_article_count",
    "avg_activity_density",
    "articles_lag1", "articles_lag2", "articles_lag3",
]

EARNINGS = ["earnings_week"]


def bucket(name):
    if name in OVERALL_TONE:    return "overall_tone"
    if name in SENTIMENT_DIMS:  return "sentiment_dim"
    if name in PRICE_VOL:       return "price_vol"
    if name in NEWS_VOL:        return "news_volume"
    if name in EARNINGS:        return "earnings"
    return "other"


BUCKET_COLORS = {
    "overall_tone":  "#3498db",   # blue  — the benchmark
    "sentiment_dim": "#e67e22",   # orange — the challengers
    "price_vol":     "#95a5a6",   # grey
    "news_volume":   "#9b59b6",   # purple
    "earnings":      "#2ecc71",   # green
    "other":         "#bdc3c7",
}


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("Loading data …")
    X_train, y_train, X_val, y_val, feature_names = load_data()
    n_features = len(feature_names)
    print(f"  Train: {X_train.shape}   Val: {X_val.shape}   Features: {n_features}")

    # ── 1. Train best model ────────────────────────────────────────────────
    print("\nFitting ExtraTrees (best model) …")
    t0 = time.time()
    clf = ExtraTreesClassifier(
        n_estimators=300, max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── 2a. Built-in (MDI) importances ────────────────────────────────────
    mdi = clf.feature_importances_

    # ── 2b. Permutation importances on val set ────────────────────────────
    print("Computing permutation importances on val set (n_repeats=30) …")
    t0 = time.time()
    perm_result = permutation_importance(
        clf, X_val, y_val,
        n_repeats=30, random_state=42,
        scoring="f1_macro", n_jobs=-1,
    )
    perm_mean = perm_result.importances_mean
    perm_std  = perm_result.importances_std
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── 3. Build summary DataFrame ─────────────────────────────────────────
    df = pd.DataFrame({
        "feature": feature_names,
        "mdi":     mdi,
        "perm":    perm_mean,
        "perm_std": perm_std,
        "bucket":  [bucket(f) for f in feature_names],
    })
    df["color"] = df["bucket"].map(BUCKET_COLORS)

    df_mdi  = df.sort_values("mdi",  ascending=False).reset_index(drop=True)
    df_perm = df.sort_values("perm", ascending=False).reset_index(drop=True)

    # ── 4. Print ranked tables ─────────────────────────────────────────────
    print("\n═══ MDI importance — top 20 ═══")
    print(df_mdi[["feature","bucket","mdi"]].head(20).to_string(index=False))

    print("\n═══ Permutation importance — top 20 ═══")
    print(df_perm[["feature","bucket","perm","perm_std"]].head(20).to_string(index=False))

    # ── 5. Overall-tone benchmark (mean across the two overall-tone cols) ──
    tone_mdi_score  = df.loc[df["feature"].isin(OVERALL_TONE), "mdi"].mean()
    tone_perm_score = df.loc[df["feature"].isin(OVERALL_TONE), "perm"].mean()
    print(f"\nOverall-tone MDI benchmark  : {tone_mdi_score:.5f}")
    print(f"Overall-tone Perm benchmark : {tone_perm_score:.5f}")

    # ── 6. Which sentiment dims beat overall tone? ─────────────────────────
    dims = df[df["bucket"] == "sentiment_dim"]
    beats_mdi  = dims[dims["mdi"]  > tone_mdi_score ].sort_values("mdi",  ascending=False)
    beats_perm = dims[dims["perm"] > tone_perm_score].sort_values("perm", ascending=False)

    print("\n── Sentiment dims that beat overall-tone (MDI) ──")
    if beats_mdi.empty:
        print("  (none)")
    else:
        print(beats_mdi[["feature","mdi"]].to_string(index=False))

    print("\n── Sentiment dims that beat overall-tone (Permutation) ──")
    if beats_perm.empty:
        print("  (none)")
    else:
        print(beats_perm[["feature","perm","perm_std"]].to_string(index=False))

    # ── 7. Figure A — all features bar chart (MDI + Perm side-by-side) ────
    print("\nSaving 5_feature_importance_bar_2.png …")
    top_n = 20
    top_feats_mdi = df_mdi.head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, col, title in [
        (axes[0], "mdi",  "MDI (Mean Decrease Impurity)"),
        (axes[1], "perm", "Permutation Importance (val F1 drop)"),
    ]:
        sub = df.sort_values(col, ascending=False).head(top_n)
        bars = ax.barh(
            sub["feature"][::-1], sub[col][::-1],
            color=[BUCKET_COLORS[b] for b in sub["bucket"][::-1]],
            edgecolor="white", linewidth=0.4,
        )
        if col == "perm":
            ax.barh(
                sub["feature"][::-1],
                sub["perm_std"][::-1] * 2,
                left=sub[col][::-1] - sub["perm_std"][::-1],
                color="black", alpha=0.15, height=0.6,
            )
        # draw benchmark line for overall tone
        bench = tone_mdi_score if col == "mdi" else tone_perm_score
        ax.axvline(bench, color="#3498db", lw=1.8, ls="--",
                   label=f"Overall-tone avg ({bench:.4f})")
        ax.set_title(f"Top {top_n} Features — {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance score", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    # legend patches
    patches = [mpatches.Patch(color=v, label=k) for k, v in BUCKET_COLORS.items()
               if k != "other"]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=9, frameon=True, title="Feature bucket",
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig("5_feature_importance_bar_2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved.")

    # ── 8. Figure B — sentiment features zoomed in ────────────────────────
    print("Saving 5_sentiment_importance_2.png …")
    sent_cols = OVERALL_TONE + SENTIMENT_DIMS
    df_sent = df[df["feature"].isin(sent_cols)].copy()
    df_sent_perm = df_sent.sort_values("perm", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, col, bench, title in [
        (axes[0], "mdi",  tone_mdi_score,  "MDI"),
        (axes[1], "perm", tone_perm_score, "Permutation (val F1 drop)"),
    ]:
        sub = df_sent.sort_values(col, ascending=False)
        colors_list = [BUCKET_COLORS[b] for b in sub["bucket"]]
        ax.barh(
            sub["feature"][::-1], sub[col][::-1],
            color=colors_list[::-1],
            edgecolor="white", linewidth=0.4,
        )
        if col == "perm":
            ax.barh(
                sub["feature"][::-1],
                sub["perm_std"][::-1] * 2,
                left=sub[col][::-1] - sub["perm_std"][::-1],
                color="black", alpha=0.2, height=0.5,
            )
        ax.axvline(bench, color="#3498db", lw=2, ls="--",
                   label=f"Overall-tone avg ({bench:.4f})")
        ax.set_title(f"Sentiment Features — {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance score", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    # color legend
    patches = [
        mpatches.Patch(color=BUCKET_COLORS["overall_tone"],  label="Overall tone (benchmark)"),
        mpatches.Patch(color=BUCKET_COLORS["sentiment_dim"], label="Sentiment dimension"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Sentiment Dimension Importance vs. Overall-Tone Baseline",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig("sentiment_importance_2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved.")

    # ── 9. Save numeric results to CSV ─────────────────────────────────────
    df.sort_values("perm", ascending=False).to_csv(
        "5_feature_importance_results_2.csv", index=False
    )
    print("  5_feature_importance_results_2.csv saved.")

    # ── 10. Return key numbers for the summary ─────────────────────────────
    return {
        "beats_mdi":  beats_mdi,
        "beats_perm": beats_perm,
        "tone_mdi":   tone_mdi_score,
        "tone_perm":  tone_perm_score,
        "df_perm":    df_perm,
    }


if __name__ == "__main__":
    results = main()
