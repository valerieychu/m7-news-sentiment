"""
final_eval.py — the SINGLE legitimate test-set evaluation for the project.

Run exactly once:
    UNLOCK_TEST_SET=final_eval python final_eval.py | tee output/final_eval.txt
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix

from prepare import load_data, load_test, _read_and_split, TICKERS
from model import build_model


def main():
    # ── Step 1: load train + val ──
    X_train, y_train, X_val, y_val, _ = load_data()

    # ── Step 2: load the LOCKED test set ──
    X_test, y_test = load_test()
    print(f"Loaded test set: {X_test.shape[0]} rows, {X_test.shape[1]} features")

    # ── Step 3: refit on train + val combined ──
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    print(f"Refitting on combined train + val: {X_full.shape[0]} rows")

    pipe = build_model()
    # Force deterministic threading for bit-for-bit reproducibility of the test number
    pipe.named_steps["model"].set_params(n_jobs=1)
    pipe.fit(X_full, y_full)

    # ── Step 4: predict and evaluate on the test set ──
    y_pred = pipe.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("FINAL TEST-SET RESULTS")
    print("=" * 60)
    print(f"  Macro F1:      {f1:.4f}")
    print(f"  Accuracy:      {acc:.4f}")
    print(f"  Recall (pos):  {rec:.4f}")
    print(f"  Confusion matrix:\n{cm}")

    # ── Step 5: per-ticker breakdown using the locked split ──
    _, _, test_df = _read_and_split()
    test_df = test_df.copy()
    assert len(test_df) == len(y_test), (
        f"Mismatch: test_df has {len(test_df)} rows but y_test has {len(y_test)}"
    )
    test_df["pred"] = y_pred

    print("\nPer-ticker test F1:")
    print(f"  {'Ticker':8s} {'n':>4s}  {'up_rate':>8s}  {'acc':>6s}  {'f1':>6s}")
    for t in TICKERS:
        g = test_df[test_df["ticker"] == t]
        if len(g) == 0:
            continue
        t_acc = accuracy_score(g["direction_t1"], g["pred"])
        t_f1 = f1_score(g["direction_t1"], g["pred"], average="macro")
        up_rate = g["direction_t1"].mean()
        print(f"  {t:8s} {len(g):4d}  {up_rate:8.3f}  {t_acc:6.3f}  {t_f1:6.3f}")

    # ── Step 6: persist predictions for future analysis without re-opening test ──
    out = test_df[["date", "ticker", "direction_t1", "pred"]].copy()
    out.to_csv("output/final_test_predictions.csv", index=False)
    print(f"\nSaved predictions to output/final_test_predictions.csv")

    # ── Reference numbers from search phase ──
    print("\nReference (search-phase val F1, for the report):")
    print(f"  Current model.py (10-seed mean):  0.559 ± 0.006")
    print(f"  PRICE-only ablation (10-seed):    0.579 ± 0.006")
    print(f"  Logistic baseline (single seed):  0.483")
    print(f"  Null baseline (DummyClassifier):  0.307")


if __name__ == "__main__":
    main()