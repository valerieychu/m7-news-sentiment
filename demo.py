"""
Demo: simulate an AutoResearch agent loop with 8 iterations.

This script demonstrates the full keep/discard workflow end-to-end on your
own project:
  1. Run baseline model
  2. Try modifications one by one
  3. Keep improvements, discard regressions  (higher macro F1 = better)
  4. Plot the trajectory

Usage: python demo.py
"""
import os
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from prepare import load_data, evaluate, log_result, plot_results, RESULTS_FILE


# ── Each "iteration" is a model the agent might try ────────
ITERATIONS = [
    {
        "id": 1,
        "description": "baseline: LogReg (no penalty)",
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(penalty=None, max_iter=1000)),
        ]),
    },
    {
        "id": 2,
        "description": "LogReg L2 (C=1.0)",
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(penalty="l2", C=1.0, max_iter=1000)),
        ]),
    },
    {
        "id": 3,
        "description": "LogReg L1 sparse (C=0.5)",
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(
                penalty="l1", C=0.5, solver="liblinear", max_iter=1000,
            )),
        ]),
    },
    {
        "id": 4,
        "description": "PolyFeatures(2, inter) + LogReg L2",
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("poly",   PolynomialFeatures(degree=2, interaction_only=True)),
            ("model",  LogisticRegression(penalty="l2", C=1.0, max_iter=2000)),
        ]),
    },
    {
        "id": 5,
        "description": "PolyFeatures(3, full) + LogReg -- overshoot",
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("poly",   PolynomialFeatures(degree=3)),
            ("model",  LogisticRegression(penalty="l2", C=0.1, max_iter=5000)),
        ]),
    },
    {
        "id": 6,
        "description": "RandomForest(n=200)",
        "model": Pipeline([
            ("model",  RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1,
            )),
        ]),
    },
    {
        "id": 7,
        "description": "GradientBoosting(n=200, depth=3)",
        "model": Pipeline([
            ("model",  GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
            )),
        ]),
    },
    {
        "id": 8,
        "description": "HistGBT(iter=300, tuned)",
        "model": Pipeline([
            ("model",  HistGradientBoostingClassifier(
                max_iter=300, max_depth=6, learning_rate=0.05,
                min_samples_leaf=20, random_state=42,
            )),
        ]),
    },
]


def main():
    # Clean previous results
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
        print(f"Cleared previous {RESULTS_FILE}\n")

    # Load data once (train + val only — test set is not touched here)
    X_train, y_train, X_val, y_val, feature_names = load_data()
    print("Dataset: News Sentiment Predicting Tech Stock Direction")
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Train class balance: up={y_train.mean():.4f}  "
          f"down={1 - y_train.mean():.4f}")
    print(f"  Val   class balance: up={y_val.mean():.4f}  "
          f"down={1 - y_val.mean():.4f}")
    print("=" * 70 + "\n")

    best_f1 = -float("inf")  # higher is better 

    for it in ITERATIONS:
        exp_id = it["id"]
        desc   = it["description"]
        model  = it["model"]

        print(f"── Experiment {exp_id}: {desc}")

        # Train
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        # Evaluate on val
        val_f1, val_acc, val_recall = evaluate(model, X_val, y_val)

        # Keep / discard decision — higher macro F1 is better
        if exp_id == 1:
            status = "baseline"
            best_f1 = val_f1
            decision_msg = "BASELINE established"
        elif val_f1 > best_f1:
            status = "keep"
            improvement = (val_f1 - best_f1) / max(abs(best_f1), 1e-9) * 100
            decision_msg = f"KEEP  (improved {improvement:.1f}% over best)"
            best_f1 = val_f1
        else:
            status = "discard"
            regression = (best_f1 - val_f1) / max(abs(best_f1), 1e-9) * 100
            decision_msg = f"DISCARD (regressed {regression:.1f}% vs best)"

        # Log
        log_result(f"exp-{exp_id:03d}", val_f1, val_acc, val_recall,
                   train_time, status, desc)

        # Print
        print(f"   F1_macro: {val_f1:.4f}  |  Acc: {val_acc:.4f}  |  "
              f"Recall: {val_recall:.4f}  |  Time: {train_time:.2f}s")
        print(f"   >>> {decision_msg}")
        print()

    # Summary
    print("=" * 70)
    print(f"Best macro F1 achieved: {best_f1:.4f}")
    print(f"Results saved to:       {RESULTS_FILE}")
    print()

    # Plot
    plot_results("performance.png")
    print("\nDone. Open performance.png to see the trajectory.")


if __name__ == "__main__":
    main()
