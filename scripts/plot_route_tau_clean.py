import json
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

EVAL_PATH = Path("outputs/eval/gated_retrieval.jsonl")
ADV_PATH = Path("outputs/adv/gated_retrieval.jsonl")
OUT_DIR = Path("outputs/figures_route_tau_clean")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MEAN_TAU = 3.25
GAMMAS = [2, 3, 4, 5]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_df(rows, dataset_name):
    out = []
    for i, row in enumerate(rows):
        u_info = row.get("u_info")
        u_cope = row.get("u_cope")
        u_spec = row.get("u_spec")
        r_safe = int(row.get("r_safe", 0))

        if u_info is None or u_cope is None or u_spec is None:
            raise ValueError(f"Missing soft scores at idx={i} in {dataset_name}")

        route_score = max(u_info, u_cope)
        mean_need = (u_info + u_cope + u_spec) / 3.0

        out.append({
            "dataset": dataset_name,
            "idx": i,
            "u_info": u_info,
            "u_cope": u_cope,
            "u_spec": u_spec,
            "route_score": route_score,
            "mean_need": mean_need,
            "r_safe": r_safe,
        })
    return pd.DataFrame(out)


def plot_route_score_distribution(df, dataset_name):
    sub = df[df["dataset"] == dataset_name]
    counts = Counter(sub["route_score"].tolist())
    xs = [1, 2, 3, 4, 5]
    ys = [counts.get(x, 0) for x in xs]

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.bar([str(x) for x in xs], ys)
    ax.set_xlabel(r"Route score $\max(u_{\mathrm{info}}, u_{\mathrm{cope}})$")
    ax.set_ylabel("Number of questions")
    ax.set_title(f"{dataset_name}: route-score distribution")

    for i, v in enumerate(ys):
        ax.text(i, v + 0.5, str(int(v)), ha="center", va="bottom", fontsize=8)

    # gamma = 4 marker (between x=3 and x=4 bars in categorical layout)
    ax.axvline(x=2.5, linestyle="--", linewidth=1)
    ax.text(2.55, max(ys) * 0.93 if max(ys) > 0 else 1, r"$\gamma=4$", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{dataset_name.lower()}_route_score_distribution_clean.png", dpi=300)
    plt.close(fig)


def summarize_gamma(df, dataset_name):
    sub = df[df["dataset"] == dataset_name]
    rows = []
    n = sub["idx"].nunique()

    for gamma in GAMMAS:
        high_axis = (sub["route_score"] >= gamma).sum()
        mean_gate = (sub["mean_need"] >= MEAN_TAU).sum()
        hard_safety = sub["r_safe"].sum()

        retrieve = ((sub["r_safe"] == 1) |
                    (sub["mean_need"] >= MEAN_TAU) |
                    (sub["route_score"] >= gamma)).sum()

        rows.append({
            "dataset": dataset_name,
            "gamma": gamma,
            "high_axis_count": int(high_axis),
            "mean_gate_count": int(mean_gate),
            "hard_safety_count": int(hard_safety),
            "retrieve_count": int(retrieve),
            "n": int(n),
        })

    return pd.DataFrame(rows)


def plot_gamma_activation(summary, dataset_name):
    sub = summary[summary["dataset"] == dataset_name]
    xs = sub["gamma"].tolist()
    ys = sub["high_axis_count"].tolist()

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.bar([str(x) for x in xs], ys)
    ax.set_xlabel(r"Route threshold $\gamma$")
    ax.set_ylabel("Number of questions")
    ax.set_title(f"{dataset_name}: high-axis activation by route threshold")

    for i, v in enumerate(ys):
        ax.text(i, v + 0.5, str(int(v)), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{dataset_name.lower()}_gamma_activation_clean.png", dpi=300)
    plt.close(fig)


def main():
    eval_df = build_df(load_jsonl(EVAL_PATH), "Eval")
    adv_df = build_df(load_jsonl(ADV_PATH), "Adv")
    df = pd.concat([eval_df, adv_df], ignore_index=True)

    plot_route_score_distribution(df, "Eval")
    plot_route_score_distribution(df, "Adv")

    summary = pd.concat([
        summarize_gamma(df, "Eval"),
        summarize_gamma(df, "Adv"),
    ], ignore_index=True)

    plot_gamma_activation(summary, "Eval")
    plot_gamma_activation(summary, "Adv")

    summary.to_csv(OUT_DIR / "gamma_activation_summary.csv", index=False)

    print("[OK] saved to", OUT_DIR)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
