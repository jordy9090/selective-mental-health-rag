import json
import re
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt


EVAL_PATH = Path("outputs/eval/gated_retrieval.jsonl")
ADV_PATH = Path("outputs/adv/gated_retrieval.jsonl")
OUT_DIR = Path("outputs/figures_route_tau_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROUTE_TAUS = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]
MEAN_TAU = 3.25


def norm_key(k):
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def walk(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield norm_key(k), v
            yield from walk(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from walk(x)


def first_float(row, candidates):
    cand = {norm_key(x) for x in candidates}
    for k, v in walk(row):
        if k in cand and v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return None


def first_bool(row, candidates):
    cand = {norm_key(x) for x in candidates}
    for k, v in walk(row):
        if k in cand and v is not None:
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                return v.strip().lower() in {"1", "true", "yes", "y"}
    return False


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_scores(row):
    u_info = first_float(row, [
        "u_info", "uinfo", "info_need", "information_need",
        "utility_info", "psychoeducation_need"
    ])
    u_cope = first_float(row, [
        "u_cope", "ucope", "cope_need", "coping_need",
        "utility_cope", "coping_strategy_need"
    ])
    u_spec = first_float(row, [
        "u_spec", "uspec", "specificity_need",
        "specific_need", "utility_spec"
    ])

    hard_safety = first_bool(row, [
        "r_safe", "rsafe", "safety_trigger",
        "hard_safety", "hard_safety_trigger",
        "is_safety_query"
    ])

    return u_info, u_cope, u_spec, hard_safety


def choose_route(u_info, u_cope):
    if u_info is None or u_cope is None:
        return "all_non_safety"
    if u_cope >= u_info:
        return "coping"
    return "psychoeducation"


def build_sweep(rows, dataset_name):
    records = []
    missing_soft = 0

    for i, row in enumerate(rows):
        u_info, u_cope, u_spec, hard_safety = extract_scores(row)

        if u_info is None or u_cope is None:
            route_score = None
            missing_soft += 1
        else:
            route_score = max(u_info, u_cope)

        if u_info is None or u_cope is None or u_spec is None:
            mean_need = None
        else:
            mean_need = (u_info + u_cope + u_spec) / 3.0

        for route_tau in ROUTE_TAUS:
            high_axis_gate = route_score is not None and route_score >= route_tau
            mean_gate = mean_need is not None and mean_need >= MEAN_TAU
            retrieve = hard_safety or high_axis_gate or mean_gate

            if hard_safety:
                sim_route = "safety"
            elif not retrieve:
                sim_route = "none"
            elif high_axis_gate:
                sim_route = choose_route(u_info, u_cope)
            elif mean_gate:
                sim_route = "all_non_safety"
            else:
                sim_route = "none"

            records.append({
                "dataset": dataset_name,
                "idx": i,
                "route_tau": route_tau,
                "u_info": u_info,
                "u_cope": u_cope,
                "u_spec": u_spec,
                "mean_need": mean_need,
                "route_score": route_score,
                "hard_safety": int(hard_safety),
                "high_axis_gate": int(high_axis_gate),
                "mean_gate": int(mean_gate),
                "retrieve": int(retrieve),
                "sim_route": sim_route,
            })

    print(f"[INFO] {dataset_name}: rows={len(rows)}, missing soft route scores={missing_soft}")
    return pd.DataFrame(records)


def plot_route_score_distribution(df, dataset_name):
    sub = df[(df["dataset"] == dataset_name) & (df["route_tau"] == ROUTE_TAUS[0])]
    scores = [x for x in sub["route_score"].dropna().tolist()]

    rounded = [round(x, 2) for x in scores]
    counts = Counter(rounded)
    xs = sorted(counts.keys())
    ys = [counts[x] for x in xs]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar([str(x) for x in xs], ys)
    ax.set_xlabel(r"route score = max($u_{info}$, $u_{cope}$)")
    ax.set_ylabel("Number of questions")
    ax.set_title(f"{dataset_name}: route-driving score distribution")

    if xs:
        for i, v in enumerate(ys):
            ax.text(i, v + 0.5, str(int(v)), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{dataset_name.lower()}_route_score_distribution.png", dpi=300)
    plt.close(fig)


def plot_high_axis_sweep(df, dataset_name):
    sub = df[df["dataset"] == dataset_name]

    summary = (
        sub.groupby("route_tau")
        .agg(
            high_axis_count=("high_axis_gate", "sum"),
            retrieve_count=("retrieve", "sum"),
            hard_safety_count=("hard_safety", "sum"),
            n=("idx", "nunique"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    xlabels = [str(x) for x in summary["route_tau"].tolist()]
    vals = summary["high_axis_count"].tolist()

    ax.bar(xlabels, vals)
    ax.set_xlabel(r"route threshold $\gamma$ for max($u_{info}$, $u_{cope}$)")
    ax.set_ylabel("Number of questions")
    ax.set_title(f"{dataset_name}: high-utility routing trigger by threshold")
    ax.tick_params(axis="x", rotation=35)

    for i, v in enumerate(vals):
        ax.text(i, v + 0.5, str(int(v)), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{dataset_name.lower()}_route_tau_high_axis_barplot.png", dpi=300)
    plt.close(fig)

    return summary


def plot_stacked_routes(df, dataset_name):
    sub = df[df["dataset"] == dataset_name]
    order = ["safety", "psychoeducation", "coping", "all_non_safety", "none"]

    pivot = (
        sub.groupby(["route_tau", "sim_route"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=order, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    xlabels = [str(x) for x in pivot.index.tolist()]
    bottom = [0] * len(pivot)

    for route in order:
        vals = pivot[route].tolist()
        ax.bar(xlabels, vals, bottom=bottom, label=route)
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_xlabel(r"route threshold $\gamma$")
    ax.set_ylabel("Number of questions")
    ax.set_title(f"{dataset_name}: simulated route distribution by route threshold")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{dataset_name.lower()}_route_tau_stacked_barplot.png", dpi=300)
    plt.close(fig)


def main():
    eval_rows = load_jsonl(EVAL_PATH)
    adv_rows = load_jsonl(ADV_PATH)

    df = pd.concat([
        build_sweep(eval_rows, "Eval"),
        build_sweep(adv_rows, "Adv"),
    ], ignore_index=True)

    df.to_csv(OUT_DIR / "route_tau_sweep_rows.csv", index=False)

    summaries = []
    for dataset_name in ["Eval", "Adv"]:
        plot_route_score_distribution(df, dataset_name)
        summaries.append(plot_high_axis_sweep(df, dataset_name))
        plot_stacked_routes(df, dataset_name)

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(OUT_DIR / "route_tau_activation_summary.csv", index=False)

    print("\n[OK] saved to:", OUT_DIR)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
