import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.gate import decide_retrieval


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)

    for col in ["u_info", "u_cope", "u_spec", "mean_need", "r_safe"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "mean_need" not in df.columns or df["mean_need"].isna().all():
        df["mean_need"] = df[["u_info", "u_cope", "u_spec"]].mean(axis=1)

    if "retrieve" not in df.columns:
        df["retrieve"] = False

    if "route" not in df.columns:
        df["route"] = "none"

    df["retrieve"] = df["retrieve"].astype(bool)
    return df


def is_safety(row) -> bool:
    r_safe = row.get("r_safe", 0)
    cats = row.get("safety_matched_categories", [])
    pats = row.get("safety_matched_patterns", [])
    return bool(r_safe == 1 or len(cats) > 0 or len(pats) > 0)


def simulated_route(row, tau: float, route_tau: float) -> str:
    u_info = row.get("u_info", 0)
    u_cope = row.get("u_cope", 0)
    u_spec = row.get("u_spec", 0)

    if not is_safety(row) and any(pd.isna(score) for score in (u_info, u_cope, u_spec)):
        return "none"

    return decide_retrieval(
        r_safe=int(is_safety(row)),
        gate_scores={"u_info": u_info, "u_cope": u_cope, "u_spec": u_spec},
        mean_threshold=tau,
        route_threshold=route_tau,
    )["route"]


def add_simulation_columns(df: pd.DataFrame, tau: float, route_tau: float) -> pd.DataFrame:
    df = df.copy()
    df["sim_route"] = df.apply(lambda row: simulated_route(row, tau, route_tau), axis=1)
    df["sim_retrieve"] = df["sim_route"] != "none"
    df["hard_safety"] = df.apply(is_safety, axis=1)
    df["max_info_cope"] = df[["u_info", "u_cope"]].max(axis=1)
    df["soft_retrieve"] = (
        (df["max_info_cope"] >= route_tau) | (df["mean_need"] >= tau)
    )
    return df


def plot_mean_need_hist(df: pd.DataFrame, name: str, out_dir: Path, tau: float):
    plt.figure(figsize=(7, 4.5))
    bins = np.arange(0, 5.25, 0.25)
    plt.hist(df["mean_need"].dropna(), bins=bins, edgecolor="black")
    plt.axvline(tau, linestyle="--", linewidth=2, label=f"tau = {tau}")
    plt.xlim(0, 5)
    plt.xlabel("Mean retrieval-need score")
    plt.ylabel("Number of questions")
    plt.title(f"{name}: Gate score distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_mean_need_hist.png", dpi=300)
    plt.close()


def plot_need_score_bars(df: pd.DataFrame, name: str, out_dir: Path):
    score_values = list(range(0, 6))
    counts = {}

    for col in ["u_info", "u_cope", "u_spec"]:
        rounded = df[col].round().astype("Int64")
        counts[col] = [int((rounded == s).sum()) for s in score_values]

    x = np.arange(len(score_values))
    width = 0.25

    plt.figure(figsize=(7.5, 4.5))
    plt.bar(x - width, counts["u_info"], width, label="u_info")
    plt.bar(x, counts["u_cope"], width, label="u_cope")
    plt.bar(x + width, counts["u_spec"], width, label="u_spec")

    plt.xticks(x, score_values)
    plt.xlim(-0.75, len(score_values) - 0.25)
    plt.xlabel("Gate score")
    plt.ylabel("Number of questions")
    plt.title(f"{name}: Distribution of gate dimensions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_need_score_barplot.png", dpi=300)
    plt.close()


def plot_activation_curve(
    df: pd.DataFrame, name: str, out_dir: Path, route_tau: float
):
    max_score = max(5.0, float(df["mean_need"].max()))
    thresholds = np.round(np.arange(0, max_score + 0.25, 0.25), 2)

    rows = []
    safety = df.apply(is_safety, axis=1)
    high_axis = df[["u_info", "u_cope"]].max(axis=1) >= route_tau
    for tau in thresholds:
        mean_gate = df["mean_need"] >= tau
        soft_retrieve = high_axis | mean_gate
        total_retrieve = safety | soft_retrieve
        rows.append(
            {
                "dataset": name,
                "tau": tau,
                "route_tau": route_tau,
                "soft_activation_rate": float(soft_retrieve.mean()),
                "total_activation_rate_with_safety": float(total_retrieve.mean()),
                "soft_count": int(soft_retrieve.sum()),
                "total_count_with_safety": int(total_retrieve.sum()),
                "n": len(df),
            }
        )

    sweep_df = pd.DataFrame(rows)

    plt.figure(figsize=(7, 4.5))
    plt.plot(sweep_df["tau"], sweep_df["soft_activation_rate"], marker="o", label="Soft gate only")
    plt.plot(sweep_df["tau"], sweep_df["total_activation_rate_with_safety"], marker="o", label="With hard safety")
    plt.xlim(0, max_score)
    plt.ylim(0, 1.05)
    plt.xlabel("Threshold tau")
    plt.ylabel("Retrieval activation rate")
    plt.title(f"{name}: Activation rate by threshold (route_tau={route_tau})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_activation_by_threshold.png", dpi=300)
    plt.close()

    return sweep_df


def plot_route_counts(df: pd.DataFrame, name: str, out_dir: Path, tau: float, route_tau: float):
    sim_df = add_simulation_columns(df, tau, route_tau)

    order = ["none", "safety", "psychoeducation", "coping", "all_non_safety"]
    counts = sim_df["sim_route"].value_counts().reindex(order, fill_value=0)

    plt.figure(figsize=(7, 4.5))
    plt.bar(counts.index, counts.values, edgecolor="black")
    plt.xlabel("Simulated route")
    plt.ylabel("Number of questions")
    plt.title(f"{name}: Routing distribution at tau={tau}, route_tau={route_tau}")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_route_counts_tau{tau}_route{route_tau}.png", dpi=300)
    plt.close()

    return sim_df


def summarize(df: pd.DataFrame, name: str, tau: float, route_tau: float):
    sim_df = add_simulation_columns(df, tau, route_tau)

    return {
        "dataset": name,
        "n": len(df),
        "saved_retrieve_count": int(df["retrieve"].sum()),
        "saved_retrieve_rate": float(df["retrieve"].mean()),
        "sim_retrieve_count": int(sim_df["sim_retrieve"].sum()),
        "sim_retrieve_rate": float(sim_df["sim_retrieve"].mean()),
        "hard_safety_count": int(sim_df["hard_safety"].sum()),
        "hard_safety_rate": float(sim_df["hard_safety"].mean()),
        "mean_need_mean": float(df["mean_need"].mean()),
        "mean_need_median": float(df["mean_need"].median()),
        "u_info_mean": float(df["u_info"].mean()),
        "u_cope_mean": float(df["u_cope"].mean()),
        "u_spec_mean": float(df["u_spec"].mean()),
        "tau": tau,
        "route_tau": route_tau,
    }


def process_one(path: Path, name: str, out_dir: Path, tau: float, route_tau: float):
    df = load_jsonl(path)

    plot_mean_need_hist(df, name, out_dir, tau)
    plot_need_score_bars(df, name, out_dir)
    sweep_df = plot_activation_curve(df, name, out_dir, route_tau)
    sim_df = plot_route_counts(df, name, out_dir, tau, route_tau)

    sim_df.to_csv(out_dir / f"{name}_calibration_rows.csv", index=False)

    return summarize(df, name, tau, route_tau), sweep_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", type=str, default="outputs/eval/gated_retrieval.jsonl")
    parser.add_argument("--adv_path", type=str, default="outputs/adv/gated_retrieval.jsonl")
    parser.add_argument("--out_dir", type=str, default="outputs/figures")
    parser.add_argument("--tau", type=float, default=3.25)
    parser.add_argument("--route_tau", type=float, default=4.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    sweeps = []

    eval_path = Path(args.eval_path)
    adv_path = Path(args.adv_path)

    if eval_path.exists():
        summary, sweep = process_one(eval_path, "eval", out_dir, args.tau, args.route_tau)
        summaries.append(summary)
        sweeps.append(sweep)
    else:
        print(f"[WARN] missing eval file: {eval_path}")

    if adv_path.exists():
        summary, sweep = process_one(adv_path, "adv", out_dir, args.tau, args.route_tau)
        summaries.append(summary)
        sweeps.append(sweep)
    else:
        print(f"[WARN] missing adv file: {adv_path}")

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(out_dir / "calibration_summary.csv", index=False)
        print("\n=== Calibration Summary ===")
        print(summary_df.to_string(index=False))

    if sweeps:
        sweep_df = pd.concat(sweeps, ignore_index=True)
        sweep_df.to_csv(out_dir / "threshold_sweep.csv", index=False)

    print(f"\nSaved figures and csv files to: {out_dir}")


if __name__ == "__main__":
    main()
