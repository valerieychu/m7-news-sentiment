# 7_per_ticker_models.py
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, accuracy_score
from prepare import _read_and_split, FORBIDDEN_COLS, TICKERS

train_df, val_df, _ = _read_and_split()
overall_true, overall_pred = [], []

print(f"{'Ticker':8s}  {'n_train':>8s}  {'n_val':>6s}  {'pooled_f1':>10s}  {'per_ticker_f1':>14s}")
print("-" * 70)

for ticker in TICKERS:
    tr = train_df[train_df["ticker"] == ticker]
    vl = val_df[val_df["ticker"] == ticker]
    if len(vl) == 0:
        continue
    y_tr = tr["direction_t1"].astype(int).to_numpy()
    y_vl = vl["direction_t1"].astype(int).to_numpy()
    X_tr = tr.drop(columns=FORBIDDEN_COLS).to_numpy(dtype=float)
    X_vl = vl.drop(columns=FORBIDDEN_COLS).to_numpy(dtype=float)

    model = ExtraTreesClassifier(n_estimators=300, max_depth=6,
                                 min_samples_leaf=5, class_weight="balanced",
                                 random_state=42, n_jobs=1)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_vl)

    f1_per = f1_score(y_vl, y_pred, average="macro")
    overall_true.extend(y_vl); overall_pred.extend(y_pred)
    print(f"{ticker:8s}  {len(tr):8d}  {len(vl):6d}  {'-':>10s}  {f1_per:14.4f}")

# Pool the per-ticker predictions and compute one overall macro F1
f1_pooled = f1_score(overall_true, overall_pred, average="macro")
acc_pooled = accuracy_score(overall_true, overall_pred)
print(f"\nPer-ticker ensemble (concat preds): F1={f1_pooled:.4f}  acc={acc_pooled:.4f}")
print(f"For comparison, pooled (one model on all tickers) F1 ≈ 0.559 ± 0.006")