"""
FROZEN -- Do not modify this file.
Data loading, train/val split, evaluation metric, and plotting.
"""
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import csv
import os

# ── Constants ──────────────────────────────────────────────
RANDOM_SEED = 42
VAL_FRACTION = 0.2
RESULTS_FILE = "results.tsv"

# ── Data ───────────────────────────────────────────────────
def load_data():
    """Load and split California Housing dataset.
    Target: median house value (in $100k).
    8 features, ~20k samples. No external download needed.
    """
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_FRACTION, random_state=RANDOM_SEED
    )
    return X_train, y_train, X_val, y_val, data.feature_names


# ── Evaluation (frozen metric) ─────────────────────────────
def evaluate(model, X_val, y_val):
    """Compute validation RMSE (lower is better)."""
    y_pred = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    r2 = float(r2_score(y_val, y_pred))
    return rmse, r2


# ── Logging ────────────────────────────────────────────────
def log_result(experiment_id, val_rmse, val_r2, status, description):
    """Append one row to results.tsv."""
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow(["experiment", "val_rmse", "val_r2", "status", "description"])
        writer.writerow([experiment_id, f"{val_rmse:.6f}", f"{val_r2:.6f}", status, description])


# ── Plotting ───────────────────────────────────────────────
def plot_results(save_path="performance.png"):
    """Plot validation RMSE over experiments from results.tsv."""
    if not os.path.exists(RESULTS_FILE):
        print("No results.tsv found. Run some experiments first.")
        return

    experiments, rmses, r2s, statuses, descriptions = [], [], [], [], []
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            experiments.append(row["experiment"])
            rmses.append(float(row["val_rmse"]))
            r2s.append(float(row["val_r2"]))
            statuses.append(row["status"])
            descriptions.append(row["description"])

    # Colors: green=keep, red=discard, blue=baseline
    color_map = {"keep": "#2ecc71", "discard": "#e74c3c", "baseline": "#3498db"}
    colors = [color_map.get(s, "#95a5a6") for s in statuses]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # ── Outlier handling: clip axes to reasonable range ──
    rmse_sorted = sorted(rmses)
    q75 = np.percentile(rmses, 75)
    iqr = np.percentile(rmses, 75) - np.percentile(rmses, 25)
    rmse_upper = q75 + 2.5 * max(iqr, 0.1)  # generous fence

    r2_sorted = sorted(r2s)
    r2_lower = max(min(r2s), np.percentile(r2s, 25) - 2.5 * max(
        np.percentile(r2s, 75) - np.percentile(r2s, 25), 0.1))

    # ── Top: RMSE ──
    ax1.scatter(range(len(rmses)), rmses, c=colors, s=80, zorder=3, edgecolors="white", linewidth=0.5)
    ax1.plot(range(len(rmses)), rmses, "k--", alpha=0.2, zorder=2)

    # Best-so-far envelope
    best_so_far = []
    current_best = float("inf")
    for r in rmses:
        current_best = min(current_best, r)
        best_so_far.append(current_best)
    ax1.plot(range(len(rmses)), best_so_far, color="#2ecc71", linewidth=2.5, label="Best so far")

    # Clip y-axis: show main range, mark outliers with arrow
    reasonable_max = min(max(rmses), rmse_upper)
    ax1.set_ylim(min(rmses) * 0.9, reasonable_max * 1.1)
    for i, r in enumerate(rmses):
        if r > reasonable_max:
            ax1.annotate(
                f"{r:.2f}", xy=(i, reasonable_max * 1.05),
                fontsize=8, ha="center", color="#e74c3c", fontweight="bold",
            )
            ax1.annotate(
                "", xy=(i, reasonable_max * 1.08), xytext=(i, reasonable_max * 1.02),
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
            )

    ax1.set_ylabel("Validation RMSE (lower is better)", fontsize=12)
    ax1.set_title("AutoResearch Demo: California Housing Regression", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # ── Bottom: R² ──
    ax2.scatter(range(len(r2s)), r2s, c=colors, s=80, zorder=3, edgecolors="white", linewidth=0.5)
    ax2.plot(range(len(r2s)), r2s, "k--", alpha=0.2, zorder=2)

    best_r2 = []
    current_best_r2 = -float("inf")
    for r in r2s:
        current_best_r2 = max(current_best_r2, r)
        best_r2.append(current_best_r2)
    ax2.plot(range(len(r2s)), best_r2, color="#2ecc71", linewidth=2.5, label="Best so far")

    # Clip y-axis for R²
    reasonable_r2_min = max(min(r2s), r2_lower)
    ax2.set_ylim(min(reasonable_r2_min * 1.1 if reasonable_r2_min < 0 else reasonable_r2_min * 0.9, -0.1),
                 max(r2s) * 1.05 if max(r2s) > 0 else 0.1)
    for i, r in enumerate(r2s):
        if r < reasonable_r2_min:
            ypos = ax2.get_ylim()[0] * 0.95
            ax2.annotate(
                f"{r:.1f}", xy=(i, ypos),
                fontsize=8, ha="center", color="#e74c3c", fontweight="bold",
            )

    ax2.set_xlabel("Experiment #", fontsize=12)
    ax2.set_ylabel("Validation R² (higher is better)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # x-tick labels = short descriptions
    short_labels = [d[:22] + ".." if len(d) > 24 else d for d in descriptions]
    ax2.set_xticks(range(len(rmses)))
    ax2.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)

    # Legend for status colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=10, label="baseline"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=10, label="keep (improved)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=10, label="discard (regressed)"),
        Line2D([0], [0], color="#2ecc71", linewidth=2.5, label="Best so far"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_results()
