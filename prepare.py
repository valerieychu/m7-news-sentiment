"""
FROZEN -- Do not modify this file.
Data loading, train/val/test split, evaluation metrics, and plotting.

Project : News Sentiment Predicting Tech Stock Direction  (STAT 390 capstone)
Task    : Binary classification. Target = direction_t1 (1 = next-day return > 0).

Split (LOCKED)
--------------
    Train  :  2025-01-01  to  2025-12-31         (12 months, ~1750 rows)
    Val    :  2026-01-01  to  2026-02-14         (1.5 months, ~210 rows)
    Test   :  2026-02-15  to  end of data        (2 months, ~266 rows)

Autoresearch discipline
-----------------------
During the SEARCH phase (iterating on models), the agent / human evaluates on
the VAL set only. The TEST set is never touched until the final evaluation at
the end of the project. To enforce this at the code level, `load_data()`
returns only train + val. The test set is gated behind a separate
`load_test()` call that emits a visible warning.

Metric
------
Macro F1 on validation  (PRIMARY — higher is better).
Accuracy and positive-class recall are logged alongside for diagnostics.

Per-experiment workflow
-----------------------
    1. X_train, y_train, X_val, y_val, feats = load_data()
    2. (optional) scale, fit whatever model you like on (X_train, y_train)
    3. f1, acc, recall = evaluate(model, X_val, y_val)
    4. log_result(experiment_id, f1, acc, recall, runtime_sec, status, desc)
    5. plot_results()              # refresh performance.png
"""
import csv
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.metrics import accuracy_score, recall_score, f1_score

# ── Constants ──────────────────────────────────────────────
RANDOM_SEED   = 42
DATASET_PATH  = "dataset.csv"
RESULTS_FILE  = "results.tsv"

TRAIN_START = pd.Timestamp("2025-01-01")
TRAIN_END   = pd.Timestamp("2025-12-31")   # inclusive
VAL_START   = pd.Timestamp("2026-01-01")   # inclusive
VAL_END     = pd.Timestamp("2026-02-14")   # inclusive  (mid-Feb boundary)
TEST_START  = pd.Timestamp("2026-02-15")   # inclusive
TEST_END    = pd.Timestamp("2026-04-14")   # inclusive  (dataset actually ends 2026-04-13)

# Columns that must never enter the model as raw values.
#   target              : direction_t1
#   future-leaking      : direction_t2, direction_t3, return_t1
#   identifiers         : date, ticker
# NOTE: `ticker` is dropped as a raw string, but re-introduced as 7 one-hot
# indicator columns (ticker_AAPL ... ticker_TSLA) — see _to_arrays().
FORBIDDEN_COLS = [
    "direction_t1",
    "direction_t2", "direction_t3",
    "return_t1",
    "date", "ticker",
]

# Magnificent-7 tickers, in a fixed order so one-hot columns are identical
# across train / val / test even if a split happens to omit a ticker.
TICKERS = ("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA")


# ── Internal helpers ───────────────────────────────────────
def _read_and_split():
    """Read dataset.csv, dropna, cut into train / val / test DataFrames."""
    df = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    df = df.dropna()  # drops first ~3 lag-warmup days per ticker + last day per ticker

    train = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)]
    val   = df[(df["date"] >= VAL_START)   & (df["date"] <= VAL_END)]
    test  = df[(df["date"] >= TEST_START)  & (df["date"] <= TEST_END)]
    return train, val, test


def _to_arrays(subset):
    """Convert a split DataFrame to (X, y, feature_names) — UNSCALED."""

    y = subset["direction_t1"].astype(np.int64).to_numpy()

    X_df = subset.drop(columns=FORBIDDEN_COLS)

    X = X_df.to_numpy(dtype=np.float64)
    return X, y, list(X_df.columns)


# ── Data ───────────────────────────────────────────────────
def load_data():
    """Load train + val arrays for the search phase.

    Returns
    -------
    X_train, y_train, X_val, y_val : np.ndarray
        Raw (UNSCALED) feature matrices as float64, binary int labels.
        Each experiment is expected to do its own scaling so that
        scale-invariant models (trees, boosting) aren't forced to receive
        scaled inputs.
    feature_names : list[str]
        Column order of X. Same for train and val.

    Notes
    -----
    * df.dropna() drops the first ~3 days per ticker (NaN lag features) and
      the last day per ticker (NaN direction_t1).  2261 raw -> 2233 labelled.
    * The split is strictly by calendar date and fully deterministic.
    * FORBIDDEN_COLS are removed so no target / future / identifier info
      can leak into any model.
    * THE TEST SET IS NOT RETURNED HERE. Use load_test() only at the final
      evaluation, never during iteration.
    """
    train_df, val_df, _ = _read_and_split()
    X_train, y_train, feature_names = _to_arrays(train_df)
    X_val,   y_val,   _             = _to_arrays(val_df)
    return X_train, y_train, X_val, y_val, feature_names


def load_test():
    """Load the LOCKED test set. Only call at the very end of the project.

    Two layers of protection:

    1. **Env-var gate (hard lock).** Raises RuntimeError unless the
       environment variable ``UNLOCK_TEST_SET=final_eval`` is set. This
       means accidental calls during the search phase crash immediately
       instead of silently evaluating on held-out data.
    2. **RuntimeWarning (visible trail).** Once unlocked, still emits a
       warning so every legitimate test-set access is logged to stdout.

    Intended invocation (exactly once, at the end of the project)::

        UNLOCK_TEST_SET=final_eval python final_eval.py

    Returns
    -------
    X_test, y_test : np.ndarray
        Raw (UNSCALED) test features and labels.
    """
    if os.environ.get("UNLOCK_TEST_SET") != "final_eval":
        raise RuntimeError(
            "load_test() is LOCKED during the search phase. "
            "Set the environment variable UNLOCK_TEST_SET=final_eval only "
            "when running final_eval.py at the end of the project."
        )
    warnings.warn(
        "load_test() called — the test set should only be accessed at the "
        "final evaluation, never during the search phase.",
        RuntimeWarning,
        stacklevel=2,
    )
    _, _, test_df = _read_and_split()
    X_test, y_test, _ = _to_arrays(test_df)
    return X_test, y_test


# ── Evaluation (frozen metric) ─────────────────────────────
def evaluate(model, X, y):
    """Score a fitted classifier on a given (X, y) split.

    Typically called with (X_val, y_val) during search, and with
    (X_test, y_test) at the final run only.

    Returns
    -------
    f1_macro : float   (PRIMARY METRIC — higher is better)
    accuracy : float
    recall   : float   (positive class, i.e. up-day recall)
    """
    y_pred   = model.predict(X)
    f1_macro = float(f1_score(y, y_pred, average="macro"))
    accuracy = float(accuracy_score(y, y_pred))
    recall   = float(recall_score(y, y_pred))
    return f1_macro, accuracy, recall


# ── Logging ────────────────────────────────────────────────
def log_result(experiment_id, val_f1, val_acc, val_recall,
               runtime_sec, status, description):
    """Append one row to results.tsv.

    Parameters
    ----------
    experiment_id : int or str
    val_f1        : float   — macro F1 on validation (primary metric)
    val_acc       : float   — accuracy on validation
    val_recall    : float   — positive-class recall on validation
    runtime_sec   : float   — wall-clock time to fit + predict
    status        : {"baseline", "keep", "discard"}
    description   : str     — short human-readable label (model + features)
    """
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow([
                "experiment", "val_f1_macro", "val_accuracy",
                "val_recall", "runtime_sec", "status", "description",
            ])
        writer.writerow([
            experiment_id,
            f"{val_f1:.6f}", f"{val_acc:.6f}", f"{val_recall:.6f}",
            f"{runtime_sec:.4f}", status, description,
        ])


# ── Plotting ───────────────────────────────────────────────
def plot_results(save_path="performance.png"):
    """Plot validation macro-F1 and accuracy over experiments.

    Top panel    : Macro F1  (higher is better)  — PRIMARY metric
    Bottom panel : Accuracy  (higher is better)
    Points colored by status:  baseline=blue,  keep=green,  discard=red.
    """
    if not os.path.exists(RESULTS_FILE):
        print("No results.tsv found. Run some experiments first.")
        return

    experiments, f1s, accs, statuses, descriptions = [], [], [], [], []
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            experiments.append(row["experiment"])
            f1s.append(float(row["val_f1_macro"]))
            accs.append(float(row["val_accuracy"]))
            statuses.append(row["status"])
            descriptions.append(row["description"])

    color_map = {"keep": "#2ecc71", "discard": "#e74c3c", "baseline": "#3498db"}
    colors = [color_map.get(s, "#95a5a6") for s in statuses]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    def _lower_fence(values, floor=0.0):
        if len(values) < 2:
            return max(floor, min(values) - 0.05)
        q25 = np.percentile(values, 25)
        q75 = np.percentile(values, 75)
        iqr = q75 - q25
        return max(floor, q25 - 2.5 * max(iqr, 0.02))

    # ── Top: Macro F1 ─────────────────────────────────────
    ax1.scatter(range(len(f1s)), f1s, c=colors, s=80, zorder=3,
                edgecolors="white", linewidth=0.5)
    ax1.plot(range(len(f1s)), f1s, "k--", alpha=0.2, zorder=2)

    best = -float("inf"); best_so_far = []
    for v in f1s:
        best = max(best, v); best_so_far.append(best)
    ax1.plot(range(len(f1s)), best_so_far,
             color="#2ecc71", linewidth=2.5, label="Best so far")

    f1_lo = _lower_fence(f1s, floor=0.0)
    f1_hi = min(1.0, max(f1s) * 1.05 + 0.02)
    ax1.set_ylim(max(0.0, f1_lo - 0.02), f1_hi)
    ax1.axhline(0.5, ls="--", color="grey", lw=0.8, alpha=0.5)
    ax1.set_ylabel("Validation Macro F1 (higher is better)", fontsize=12)
    ax1.set_title(
        "AutoResearch: News Sentiment → Tech Stock Direction",
        fontsize=14, fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)

    for i, v in enumerate(f1s):
        if v < f1_lo - 0.02:
            ax1.annotate(
                f"{v:.2f}", xy=(i, f1_lo),
                fontsize=8, ha="center", color="#e74c3c", fontweight="bold",
            )
            ax1.annotate(
                "", xy=(i, f1_lo - 0.005), xytext=(i, f1_lo + 0.02),
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
            )

    # ── Bottom: Accuracy ─────────────────────────────────
    ax2.scatter(range(len(accs)), accs, c=colors, s=80, zorder=3,
                edgecolors="white", linewidth=0.5)
    ax2.plot(range(len(accs)), accs, "k--", alpha=0.2, zorder=2)

    best = -float("inf"); best_acc = []
    for v in accs:
        best = max(best, v); best_acc.append(best)
    ax2.plot(range(len(accs)), best_acc,
             color="#2ecc71", linewidth=2.5, label="Best so far")

    acc_lo = _lower_fence(accs, floor=0.0)
    acc_hi = min(1.0, max(accs) * 1.05 + 0.02)
    ax2.set_ylim(max(0.0, acc_lo - 0.02), acc_hi)
    ax2.axhline(0.5, ls="--", color="grey", lw=0.8, alpha=0.5)
    ax2.set_xlabel("Experiment #", fontsize=12)
    ax2.set_ylabel("Validation Accuracy (higher is better)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    short_labels = [(d[:22] + "..") if len(d) > 24 else d for d in descriptions]
    ax2.set_xticks(range(len(f1s)))
    ax2.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#3498db", markersize=10, label="baseline"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#2ecc71", markersize=10, label="keep (improved)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#e74c3c", markersize=10, label="discard (regressed)"),
        Line2D([0], [0], color="#2ecc71", linewidth=2.5, label="Best so far"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_results()
