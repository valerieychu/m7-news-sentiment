"""
Run a battery of ablation experiments. Each ablation slices columns out of
the feature matrix and logs F1 / accuracy / recall / runtime to results.tsv.

The structure mirrors run.py but iterates a list of (description, drop_cols)
pairs instead of running just the current model.py.
"""
import time
import subprocess
import numpy as np

from prepare import load_data, evaluate, log_result
from model import build_model


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


X_train, y_train, X_val, y_val, feature_names = load_data()
name2idx = {name: i for i, name in enumerate(feature_names)}

SENTIMENT = [n for n in feature_names if any(k in n for k in
    ["tone", "polarity", "article", "source", "activity"])]
PRICE = ["open", "high", "low", "close", "volume", "daily_return", "realized_vol5"]
POLARITY_LAGS = ["polarity_lag1", "polarity_lag2", "polarity_lag3"]

ABLATIONS = [
    ("ablation: ALL features (cleaned data, ffill+missing flag)", []),
    ("ablation: drop gdelt_missing only",                          ["gdelt_missing"]),
    ("ablation: drop earnings_week (top permutation feature)",     ["earnings_week"]),
    ("ablation: drop all polarity_lag1/2/3",                       POLARITY_LAGS),
    ("ablation: SENTIMENT-only (drop price + earnings)",           PRICE + ["earnings_week"]),
    ("ablation: PRICE-only (drop all sentiment)",                  SENTIMENT),
]

experiment_id = get_git_hash()
print(f"Experiment: {experiment_id}\n")

for desc, drop_cols in ABLATIONS:
    drop_idx = [name2idx[c] for c in drop_cols if c in name2idx]
    keep_idx = [i for i in range(X_train.shape[1]) if i not in drop_idx]
    Xtr = X_train[:, keep_idx]
    Xv  = X_val[:, keep_idx]

    pipe = build_model()
    t0 = time.time()
    pipe.fit(Xtr, y_train)
    runtime = time.time() - t0
    f1, acc, rec = evaluate(pipe, Xv, y_val)

    log_result(experiment_id, f1, acc, rec, runtime, "keep", desc)
    print(f"  {desc}")
    print(f"    F1={f1:.4f}  acc={acc:.4f}  recall={rec:.4f}  "
          f"runtime={runtime:.3f}s  kept_features={Xtr.shape[1]}")
