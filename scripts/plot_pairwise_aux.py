import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

FILES = [
    "outputs/eval/pairwise_closed_vs_always.summary.json",
    "outputs/eval/pairwise_closed_vs_selective.summary.json",
    "outputs/eval/pairwise_always_vs_selective.summary.json",
]

OUT_PATH = "outputs/figures_final/pairwise_aux_stacked_bar.png"


def pretty_comp(left, right):
    short = {
        "TunedClosed": "Closed",
        "AlwaysRet": "Always",
        "SelectiveRet": "Selective",
    }
    return f"{short.get(left, left)} vs {short.get(right, right)}"


def pretty_metric(metric):
    if metric == "factual_consistency":
        return "Factual Consistency"
    if metric == "lower_toxicity":
        return "Less Harmful / Toxic"
    return metric


rows = []
for fp in FILES:
    with open(fp, "r", encoding="utf-8") as f:
        obj = json.load(f)

    comp = pretty_comp(obj["left_label"], obj["right_label"])

    for metric, vals in obj["summary"].items():
        rows.append({
            "label": f"{comp} | {pretty_metric(metric)}",
            "left": vals["left_win_rate"] * 100,
            "right": vals["right_win_rate"] * 100,
            "tie": vals["tie_rate"] * 100,
        })

labels = [r["label"] for r in rows]
left_vals = np.array([r["left"] for r in rows])
right_vals = np.array([r["right"] for r in rows])
tie_vals = np.array([r["tie"] for r in rows])

y = np.arange(len(labels))

plt.figure(figsize=(10, 5.8))
plt.barh(y, left_vals, label="Left win")
plt.barh(y, right_vals, left=left_vals, label="Right win")
plt.barh(y, tie_vals, left=left_vals + right_vals, label="Tie")

plt.yticks(y, labels, fontsize=9)
plt.xlabel("Percentage (%)")
plt.xlim(0, 100)
plt.title("Pairwise Auxiliary Comparison on Saturated Dimensions")
plt.legend(loc="lower right")

for i in range(len(labels)):
    if left_vals[i] > 0:
        plt.text(left_vals[i] / 2, i, f"{left_vals[i]:.0f}", va="center", ha="center", fontsize=8)
    if right_vals[i] > 0:
        plt.text(left_vals[i] + right_vals[i] / 2, i, f"{right_vals[i]:.0f}", va="center", ha="center", fontsize=8)
    if tie_vals[i] > 0:
        plt.text(left_vals[i] + right_vals[i] + tie_vals[i] / 2, i, f"{tie_vals[i]:.0f}", va="center", ha="center", fontsize=8)

plt.tight_layout()
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
print(f"[DONE] saved {OUT_PATH}")
