import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EVAL_PATH = Path("outputs/eval/gated_retrieval.jsonl")
ADV_PATH = Path("outputs/adv/gated_retrieval.jsonl")
OUT_DIR = Path("outputs/figures_route_threshold_combined")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def deep_get(obj, candidate_paths, default=None):
    """
    Try multiple nested key paths.
    Example path: ["gate", "u_info"]
    """
    for path in candidate_paths:
        cur = obj
        ok = True
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok:
            return cur
    return default


def extract_scores(row):
    """
    Robustly extract route utility scores from slightly different jsonl schemas.
    Expected scores:
      u_info, u_cope, u_spec
    Hard safety:
      r_safe / hard_safety / safety_trigger
    """

    u_info = deep_get(row, [
        ["u_info"],
        ["info"],
        ["info_score"],
        ["gate", "u_info"],
        ["gate", "info"],
        ["gate_scores", "u_info"],
        ["gate_scores", "info"],
        ["soft_gate", "u_info"],
        ["soft_gate", "info"],
        ["route_scores", "u_info"],
        ["route_scores", "info"],
    ])

    u_cope = deep_get(row, [
        ["u_cope"],
        ["cope"],
        ["cope_score"],
        ["gate", "u_cope"],
        ["gate", "cope"],
        ["gate_scores", "u_cope"],
        ["gate_scores", "cope"],
        ["soft_gate", "u_cope"],
        ["soft_gate", "cope"],
        ["route_scores", "u_cope"],
        ["route_scores", "cope"],
    ])

    u_spec = deep_get(row, [
        ["u_spec"],
        ["spec"],
        ["specificity"],
        ["specificity_score"],
        ["gate", "u_spec"],
        ["gate", "spec"],
        ["gate", "specificity"],
        ["gate_scores", "u_spec"],
        ["gate_scores", "spec"],
        ["gate_scores", "specificity"],
        ["soft_gate", "u_spec"],
        ["soft_gate", "spec"],
        ["route_scores", "u_spec"],
        ["route_scores", "spec"],
    ])

    hard_safety = deep_get(row, [
        ["r_safe"],
        ["hard_safety"],
        ["safety_trigger"],
        ["is_safety"],
        ["gate", "r_safe"],
        ["gate", "hard_safety"],
        ["gate", "safety_trigger"],
        ["gate_scores", "r_safe"],
        ["gate_scores", "hard_safety"],
        ["soft_gate", "r_safe"],
    ], default=False)

    retrieved = deep_get(row, [
        ["retrieved"],
        ["retrieve"],
        ["do_retrieve"],
        ["use_retrieval"],
        ["retrieval_activated"],
        ["gate", "retrieved"],
        ["gate", "retrieve"],
        ["gate", "do_retrieve"],
        ["decision", "retrieved"],
        ["decision", "retrieve"],
    ], default=None)

    # Normalize bool-ish values.
    if isinstance(hard_safety, str):
        hard_safety = hard_safety.lower() in {"true", "yes", "1", "safety"}
    else:
        hard_safety = bool(hard_safety)

    if isinstance(retrieved, str):
        retrieved = retrieved.lower() in {"true", "yes", "1", "retrieve", "retrieved"}
    elif retrieved is not None:
        retrieved = bool(retrieved)

    # Convert scores.
    try:
        u_info = float(u_info)
        u_cope = float(u_cope)
        u_spec = float(u_spec)
    except Exception:
        return None

    route_score = max(u_info, u_cope)
    mean_need = float(np.mean([u_info, u_cope, u_spec]))

    return {
        "u_info": u_info,
        "u_cope": u_cope,
        "u_spec": u_spec,
        "route_score": route_score,
        "mean_need": mean_need,
        "hard_safety": hard_safety,
        "retrieved": retrieved,
    }


def build_df(path: Path, split_name: str):
    raw_rows = load_jsonl(path)
    parsed = []

    missing = []
    for i, row in enumerate(raw_rows):
        item = extract_scores(row)
        if item is None:
            missing.append(i)
            continue
        item["idx"] = i
        item["split"] = split_name
        parsed.append(item)

    if missing:
        print(f"[WARN] {split_name}: missing route scores = {len(missing)} | indices={missing[:20]}")
    else:
        print(f"[INFO] {split_name}: rows={len(parsed)}, missing route scores=0")

    return pd.DataFrame(parsed)


def main():
    eval_df = build_df(EVAL_PATH, "Eval")
    adv_df = build_df(ADV_PATH, "Adv")
    df = pd.concat([eval_df, adv_df], ignore_index=True)

    if df.empty:
        raise RuntimeError("No valid rows found. Check score field names in gated_retrieval.jsonl.")

    # Since scores are 1--5 integer-like, integer thresholds are what matter.
    score_values = [1, 2, 3, 4, 5]
    gammas = [2, 3, 4, 5]

    # Distribution table.
    dist_rows = []
    for split, sub in df.groupby("split"):
        rounded_scores = sub["route_score"].round().astype(int)
        counts = Counter(rounded_scores)
        for s in score_values:
            dist_rows.append({
                "split": split,
                "route_score": s,
                "count": counts.get(s, 0),
                "n": len(sub),
                "rate": counts.get(s, 0) / len(sub),
            })
    dist_df = pd.DataFrame(dist_rows)

    # Threshold activation table.
    activation_rows = []
    for split, sub in df.groupby("split"):
        n = len(sub)
        hard_safety_count = int(sub["hard_safety"].sum())

        for gamma in gammas:
            high_axis = sub["route_score"] >= gamma
            high_axis_count = int(high_axis.sum())

            # This is not necessarily the exact final retrieval decision if your main gate
            # also uses mean threshold tau. Here we isolate the route threshold itself.
            activation_rows.append({
                "split": split,
                "gamma": gamma,
                "high_axis_count": high_axis_count,
                "high_axis_rate": high_axis_count / n,
                "hard_safety_count": hard_safety_count,
                "n": n,
            })

    activation_df = pd.DataFrame(activation_rows)

    dist_df.to_csv(OUT_DIR / "combined_route_score_distribution.csv", index=False)
    activation_df.to_csv(OUT_DIR / "combined_route_threshold_activation.csv", index=False)

    print("\n=== Route score distribution ===")
    print(dist_df.to_string(index=False))

    print("\n=== Route threshold activation ===")
    print(activation_df.to_string(index=False))

    # ---------- Plot ----------
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 160,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    # Panel A: route score distribution
    ax = axes[0]
    width = 0.36
    x = np.arange(len(score_values))

    eval_counts = [
        dist_df[(dist_df["split"] == "Eval") & (dist_df["route_score"] == s)]["count"].iloc[0]
        for s in score_values
    ]
    adv_counts = [
        dist_df[(dist_df["split"] == "Adv") & (dist_df["route_score"] == s)]["count"].iloc[0]
        for s in score_values
    ]

    ax.bar(x - width / 2, eval_counts, width, label="CounselBench-Eval")
    ax.bar(x + width / 2, adv_counts, width, label="CounselBench-Adv")

    ax.axvline(x=3.5, linestyle="--", linewidth=1.2)
    ax.text(3.55, max(max(eval_counts), max(adv_counts)) * 0.92,
            r"$\gamma=4$", rotation=90, va="top")

    ax.set_title("(a) Distribution of route utility score")
    ax.set_xlabel(r"Route score $\max(u_{\mathrm{info}}, u_{\mathrm{cope}})$")
    ax.set_ylabel("Number of examples")
    ax.set_xticks(x)
    ax.set_xticklabels(score_values)
    ax.legend(frameon=False)

    # Panel B: threshold activation
    ax = axes[1]
    x = np.arange(len(gammas))

    eval_act = [
        activation_df[(activation_df["split"] == "Eval") & (activation_df["gamma"] == g)]["high_axis_count"].iloc[0]
        for g in gammas
    ]
    adv_act = [
        activation_df[(activation_df["split"] == "Adv") & (activation_df["gamma"] == g)]["high_axis_count"].iloc[0]
        for g in gammas
    ]

    ax.bar(x - width / 2, eval_act, width, label="CounselBench-Eval")
    ax.bar(x + width / 2, adv_act, width, label="CounselBench-Adv")

    # Annotate gamma=4 bars.
    if 4 in gammas:
        g4_idx = gammas.index(4)
        ax.annotate(
            r"$\gamma=4$",
            xy=(g4_idx, max(eval_act[g4_idx], adv_act[g4_idx])),
            xytext=(g4_idx, max(max(eval_act), max(adv_act)) * 0.82 if max(max(eval_act), max(adv_act)) > 0 else 1),
            ha="center",
            arrowprops=dict(arrowstyle="->", linewidth=1.0),
        )

    ax.set_title("(b) Activation by route threshold")
    ax.set_xlabel(r"Route threshold $\gamma$")
    ax.set_ylabel(r"Count with $\max(u_{\mathrm{info}},u_{\mathrm{cope}})\geq\gamma$")
    ax.set_xticks(x)
    ax.set_xticklabels(gammas)
    ax.legend(frameon=False)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)

    fig.tight_layout()

    out_png = OUT_DIR / "combined_route_threshold_calibration.png"
    out_pdf = OUT_DIR / "combined_route_threshold_calibration.pdf"

    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    print(f"\n[OK] saved: {out_png}")
    print(f"[OK] saved: {out_pdf}")
    print(f"[OK] saved: {OUT_DIR / 'combined_route_score_distribution.csv'}")
    print(f"[OK] saved: {OUT_DIR / 'combined_route_threshold_activation.csv'}")


if __name__ == "__main__":
    main()
