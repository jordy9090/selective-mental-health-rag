import json
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt


out_dir = Path("outputs/figures_final")
out_dir.mkdir(parents=True, exist_ok=True)

settings = [
    {
        "tau": "2.25",
        "label": r"$\tau=2.25$",
        "eval_path": "outputs/eval/gated_retrieval_tau2p25.jsonl",
        "adv_path": "outputs/adv/gated_retrieval_tau2p25.jsonl",
    },
    {
        "tau": "3.25",
        "label": r"$\tau=3.25$",
        "eval_path": "outputs/eval/gated_retrieval.jsonl",
        "adv_path": "outputs/adv/gated_retrieval.jsonl",
    },
]


def get_activation(path):
    total = 0
    retrieved = 0
    routes = Counter()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            total += 1
            if r.get("retrieve"):
                retrieved += 1
                routes[r.get("route", "none")] += 1

    return {
        "total": total,
        "retrieved": retrieved,
        "rate": retrieved / total if total else 0.0,
        "routes": routes,
    }


eval_rates = []
adv_rates = []
labels = []

print("=== Activation summary ===")
for s in settings:
    eval_stat = get_activation(s["eval_path"])
    adv_stat = get_activation(s["adv_path"])

    labels.append(s["label"])
    eval_rates.append(eval_stat["rate"])
    adv_rates.append(adv_stat["rate"])

    print(
        f"tau={s['tau']} | "
        f"Eval: {eval_stat['retrieved']}/{eval_stat['total']} = {eval_stat['rate']:.3f}, routes={dict(eval_stat['routes'])} | "
        f"Adv: {adv_stat['retrieved']}/{adv_stat['total']} = {adv_stat['rate']:.3f}, routes={dict(adv_stat['routes'])}"
    )


x = range(len(labels))
width = 0.34

plt.figure(figsize=(6.2, 3.6))

eval_x = [i - width / 2 for i in x]
adv_x = [i + width / 2 for i in x]

plt.bar(eval_x, eval_rates, width=width, label="Eval")
plt.bar(adv_x, adv_rates, width=width, label="Adv")

for xpos, val in zip(eval_x, eval_rates):
    plt.text(xpos, val + 0.012, f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9)

for xpos, val in zip(adv_x, adv_rates):
    plt.text(xpos, val + 0.012, f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9)

plt.xticks(list(x), labels)
plt.xlabel("Threshold")
plt.ylabel("Retrieval activation rate")
plt.ylim(0, 0.5)
plt.title("Retrieval activation under threshold settings")
plt.legend()
plt.tight_layout()

out_path = out_dir / "tau_actual_activation_barplot_main.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print("saved:", out_path)
