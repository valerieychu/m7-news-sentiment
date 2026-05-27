"""Efficiency-vs-effectiveness scatter: val F1 vs training runtime per experiment."""
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adjustText import adjust_text

rows = []
with open("results.tsv") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        rows.append(row)

COLORS = {"baseline": "#3498db", "keep": "#2ecc71", "discard": "#e74c3c"}

fig, ax = plt.subplots(figsize=(11, 7))
for row in rows:
    f1 = float(row["val_f1_macro"])
    rt = float(row["runtime_sec"])
    status = row["status"]
    ax.scatter(rt, f1, s=120,
               color=COLORS.get(status, "#888888"),
               edgecolors="white", linewidth=0.8, zorder=3)

# Annotate notable experiments (current best + ablation rows)
labels_to_show = []
for row in rows:
    f1 = float(row["val_f1_macro"])
    rt = float(row["runtime_sec"])
    desc = row["description"]
    if f1 >= 0.55 or "ablation:" in desc or "baseline" in desc:
        short = desc.replace("ablation: ", "")[:38]
        labels_to_show.append(ax.text(rt, f1, short, fontsize=8))

adjust_text(
    labels_to_show, 
    expand=(1.2, 1.4), 
    arrowprops=dict(arrowstyle="->", color="grey", lw=0.5, alpha=0.6)
)

ax.axhline(0.5, ls="--", color="grey", alpha=0.4, label="Random baseline (F1=0.5)")
ax.set_xscale("log")
ax.set_xlabel("Training runtime (seconds, log scale)", fontsize=12)
ax.set_ylabel("Validation Macro F1 (higher is better)", fontsize=12)
ax.set_title("Efficiency vs. Effectiveness — All experiments",
             fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)


ax.legend(handles=[
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#3498db",
           markersize=10, label="baseline"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#2ecc71",
           markersize=10, label="keep"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#e74c3c",
           markersize=10, label="discard"),
    Line2D([0],[0], ls="--", color="grey", label="Random baseline"),
], loc="lower right")

plt.tight_layout()
plt.savefig("efficiency_vs_effectiveness.png", dpi=150, bbox_inches="tight")
print("Saved efficiency_vs_effectiveness.png")
