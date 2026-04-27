"""
Failure Mode Analysis for News Sentiment → Stock Direction
==========================================================
Run from the m7-news-sentiment-copy directory:
    python failure_mode_analysis.py

Sections
--------
1. Common failure modes summary (printed)
2. Misclassified examples (feature vectors)
3. Confusion matrix
4. High-confidence-but-wrong predictions
5. Per-class (underperforming class) report
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — saves PNGs instead of showing windows
import matplotlib.pyplot as plt

# ── make sure imports resolve whether script is run from project root or sub-dir ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prepare import load_data
from model import build_model


# ══════════════════════════════════════════════════════════════════════════════
# 0. Load data & fit model
# ══════════════════════════════════════════════════════════════════════════════
X_train, y_train, X_val, y_val, feature_names = load_data()

model = build_model()
model.fit(X_train, y_train)

print("Model fitted.\n")
print(f"Val set size : {len(y_val)} examples")
print(f"Class 0 (down/flat) : {(y_val == 0).sum()}")
print(f"Class 1 (up)        : {(y_val == 1).sum()}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Common Failure Modes Encountered So Far
# ══════════════════════════════════════════════════════════════════════════════
FAILURE_MODES = """
╔══════════════════════════════════════════════════════════════════════════════╗
║          COMMON FAILURE MODES — News Sentiment Stock Direction Project       ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. NEAR-RANDOM MACRO F1 (~0.485)
   ─────────────────────────────
   The baseline logistic regression scores a macro F1 of only ~0.485 — slightly
   below random chance for a balanced binary task.  This suggests the linear
   decision boundary cannot separate the sentiment signal from noise.  Root
   causes: (a) sentiment features are noisy proxies for next-day return, and
   (b) news events are already partially priced in by the time articles are
   published (efficient-market effect).

2. CLASS IMBALANCE IN RECALL
   ──────────────────────────
   The baseline achieves recall ~0.64 for the positive class (up-day) while
   presumably under-predicting the negative class (down/flat).  Models tend to
   predict "up" more often because macro-averaged losses reward majority-class
   prediction.  Fix: use class_weight='balanced' or SMOTE-style oversampling.

3. SMALL VALIDATION SET
   ─────────────────────
   ~210 validation rows (1.5 months × 7 tickers) means a single volatile week
   on one ticker can swing macro F1 by ±0.02.  Real improvements must exceed
   this threshold to be meaningful; apparent improvements smaller than ~0.02
   are likely noise.

4. WEAK / NOISY SENTIMENT FEATURES
   ──────────────────────────────────
   GDELT tone scores are article-level averages and may not capture intra-day
   sentiment shifts.  Highly negative headlines for NVDA may still precede
   an up-day if the news was already expected.  Models trained on sentiment
   alone struggle to disentangle "bad news already priced in" vs. genuine
   negative surprise.

5. TEMPORAL LEAKAGE RISK (guarded)
   ─────────────────────────────────
   Columns direction_t2, direction_t3, return_t1 are future labels and are
   explicitly blocked.  However, rolling features (tone_rolling5, tone_lag*)
   encode recent history and can create soft leakage if a particularly
   newsworthy multi-day event dominates all lag windows simultaneously.

6. TICKER-SPECIFIC HETEROGENEITY
   ─────────────────────────────────
   M7 stocks behave differently: TSLA reacts sharply to Musk tweets, NVDA
   to GPU/AI supply news, AAPL to product cycles.  A global model that treats
   all tickers identically mixes incompatible signal structures, hurting
   per-class recall for tickers with idiosyncratic dynamics.

7. FEATURE SCALE SENSITIVITY
   ──────────────────────────
   Logistic regression is sensitive to feature scale.  StandardScaler is
   applied, but tone_stddev and article_count have heavy right tails; extreme
   news cycles (earnings weeks, crashes) create outlier rows that pull
   decision boundaries.  RobustScaler or QuantileTransformer may help.

8. OVERFITTING ON POLYNOMIAL FEATURES
   ──────────────────────────────────────
   Adding PolynomialFeatures(degree=3) over 32 inputs creates >5000 features,
   many spuriously correlated with the target on the small training set.
   This caused a run to be discarded (status = 'discard') due to val
   performance dropping below baseline despite near-perfect training accuracy.
"""

print(FAILURE_MODES)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Look at Misclassified Examples
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 2 — MISCLASSIFIED EXAMPLES")
print("=" * 70)

y_pred = model.predict(X_val)
errors = np.where(y_pred != y_val)[0]

print(f"\nTotal errors : {len(errors)}  out of  {len(y_val)}  "
      f"({100 * len(errors) / len(y_val):.1f}% error rate)\n")

# Top-5 most informative features to display per example (highest abs weight
# for logistic regression; fall back to first 5 if model doesn't expose coef_)
try:
    # Works for Pipeline ending in LogisticRegression
    coef = np.abs(model.named_steps["model"].coef_[0])
    top_feat_idx = np.argsort(coef)[::-1][:5]
    top_feat_names = [feature_names[i] for i in top_feat_idx]
except Exception:
    top_feat_idx = list(range(5))
    top_feat_names = feature_names[:5]

label_map = {0: "Down/Flat (0)", 1: "Up (1)"}

print(f"Showing top-10 misclassified examples.")
print(f"Feature snapshot: {top_feat_names}\n")

for i in errors[:10]:
    print("──────────────────────────────────────────────────────────────────")
    print(f"  True label  : {label_map[y_val[i]]}")
    print(f"  Predicted   : {label_map[y_pred[i]]}")
    print("  Top features:")
    for j, fname in zip(top_feat_idx, top_feat_names):
        print(f"    {fname:<30s}: {X_val[i, j]:+.4f}")

print()


# ══════════════════════════════════════════════════════════════════════════════
# 3. Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 3 — CONFUSION MATRIX")
print("=" * 70)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_val, y_pred)
print("\n  Rows = true label | Cols = predicted label")
print(f"  Labels: 0 = Down/Flat, 1 = Up\n")
print(cm)

# Save figure
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down/Flat", "Up"])
disp.plot(ax=ax, colorbar=False)
ax.set_title("Confusion Matrix — Validation Set")
plt.tight_layout()
save_cm = os.path.join(os.path.dirname(os.path.abspath(__file__)), "confusion_matrix.png")
plt.savefig(save_cm, dpi=150, bbox_inches="tight")
print(f"\nConfusion matrix saved → {save_cm}\n")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 4. High-Confidence But Wrong
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 4 — HIGH-CONFIDENCE BUT WRONG PREDICTIONS")
print("=" * 70)

try:
    probs      = model.predict_proba(X_val)
    y_pred_p   = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    errors_p   = np.where(y_pred_p != y_val)[0]

    # Sort by confidence descending
    high_conf_errors = sorted(errors_p, key=lambda i: -confidence[i])

    print(f"\nTop-10 most confident wrong predictions:\n")
    for rank, i in enumerate(high_conf_errors[:10], 1):
        print(f"  [{rank}] True: {label_map[y_val[i]]}  |  "
              f"Pred: {label_map[y_pred_p[i]]}  |  "
              f"Confidence: {confidence[i]:.3f}")
        for j, fname in zip(top_feat_idx, top_feat_names):
            print(f"       {fname:<30s}: {X_val[i, j]:+.4f}")
        print()

    # Distribution of confidence for correct vs. wrong predictions
    correct_conf = confidence[y_pred_p == y_val]
    wrong_conf   = confidence[y_pred_p != y_val]
    print(f"  Mean confidence — correct predictions : {correct_conf.mean():.3f}")
    print(f"  Mean confidence — wrong predictions   : {wrong_conf.mean():.3f}")
    print(f"  (A large gap means the model is over-confident on errors)\n")

    # Save confidence histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(correct_conf, bins=20, alpha=0.6, label="Correct", color="#2ecc71")
    ax.hist(wrong_conf,   bins=20, alpha=0.6, label="Wrong",   color="#e74c3c")
    ax.set_xlabel("Predicted Probability (max class)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution: Correct vs. Wrong Predictions")
    ax.legend()
    plt.tight_layout()
    save_conf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "confidence_hist.png")
    plt.savefig(save_conf, dpi=150, bbox_inches="tight")
    print(f"Confidence histogram saved → {save_conf}\n")
    plt.close()

except AttributeError:
    print("  (This model does not support predict_proba — skipping section 4)\n")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Underperforming Classes — Full Classification Report
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 5 — PER-CLASS PERFORMANCE (classification_report)")
print("=" * 70)

from sklearn.metrics import classification_report

print()
print(classification_report(
    y_val, y_pred,
    target_names=["Down/Flat (0)", "Up (1)"],
    digits=4,
))

# Per-class F1 breakdown
from sklearn.metrics import f1_score
f1_per_class = f1_score(y_val, y_pred, average=None)
print(f"  F1 — Down/Flat (0) : {f1_per_class[0]:.4f}")
print(f"  F1 — Up       (1)  : {f1_per_class[1]:.4f}")
print(f"  Macro F1           : {f1_per_class.mean():.4f}")

if f1_per_class[0] < f1_per_class[1] - 0.05:
    print("\n  ⚠ Class 0 (Down/Flat) is underperforming — "
          "consider class_weight='balanced' or a higher recall threshold.")
elif f1_per_class[1] < f1_per_class[0] - 0.05:
    print("\n  ⚠ Class 1 (Up) is underperforming — "
          "the model may be biased toward predicting down/flat days.")
else:
    print("\n  ✓ Both classes are performing similarly (within 0.05 F1).")

print("\nDone.\n")
