import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("outputs/figures_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 원래 calibration 결과 CSV
csv_candidates = [
    Path("outputs/figures_final/threshold_sweep.csv"),
    Path("outputs/figures/threshold_sweep.csv"),
    Path("outputs/figures_tau2p25/threshold_sweep.csv"),
]

csv_path = None
for p in csv_candidates:
    if p.exists():
        csv_path = p
        break

if csv_path is None:
    raise FileNotFoundError("threshold_sweep.csv not found in outputs/figures_final or outputs/figures")

df = pd.read_csv(csv_path)
print("[INFO] using:", csv_path)
print("[INFO] columns:", list(df.columns))

# column 이름 자동 대응
def pick_col(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

tau_col = pick_col(["tau", "threshold", "mean_tau"])
dataset_col = pick_col(["dataset", "split", "name"])
rate_col = pick_col([
    "total_activation_rate_with_safety",
    "retrieval_activation_rate",
    "activation_rate",
    "total_retrieval_activation_rate",
    "retrieve_rate",
    "retrieval_rate",
    "rate",
])

if tau_col is None or dataset_col is None or rate_col is None:
    raise ValueError(
        f"Cannot infer columns. tau_col={tau_col}, dataset_col={dataset_col}, rate_col={rate_col}. "
        f"Columns={list(df.columns)}"
    )

# main figure에 쓸 threshold만
taus = [2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50]
df = df[df[tau_col].round(2).isin(taus)].copy()
df[tau_col] = df[tau_col].round(2)

# dataset 이름 정리
def norm_dataset(x):
    s = str(x).lower()
    if "eval" in s:
        return "Eval"
    if "adv" in s:
        return "Adv"
    return str(x)

df["dataset_norm"] = df[dataset_col].map(norm_dataset)

eval_rates = []
adv_rates = []
for tau in taus:
    e = df[(df[tau_col] == tau) & (df["dataset_norm"] == "Eval")][rate_col]
    a = df[(df[tau_col] == tau) & (df["dataset_norm"] == "Adv")][rate_col]

    if len(e) == 0 or len(a) == 0:
        raise ValueError(f"Missing Eval/Adv row for tau={tau}")

    eval_rates.append(float(e.iloc[0]))
    adv_rates.append(float(a.iloc[0]))

# 혹시 percentage로 저장돼 있으면 0~1로 변환
if max(eval_rates + adv_rates) > 1.5:
    eval_rates = [x / 100.0 for x in eval_rates]
    adv_rates = [x / 100.0 for x in adv_rates]

fig, ax = plt.subplots(figsize=(8.6, 4.8))

x = range(len(taus))
width = 0.36

ax.bar([i - width / 2 for i in x], eval_rates, width, label="Eval")
ax.bar([i + width / 2 for i in x], adv_rates, width, label="Adv")

# 선택 threshold 표시
tau_labels = [f"{t:.2f}" for t in taus]
ax.axvline(taus.index(2.25), linestyle="--", linewidth=1)
ax.text(
    taus.index(2.25) + 0.03,
    max(max(eval_rates), max(adv_rates)) * 0.96,
    r"$\tau$=2.25",
    rotation=90,
    va="top",
    fontsize=10,
)

ax.axvline(taus.index(3.25), linestyle="--", linewidth=1)
ax.text(
    taus.index(3.25) + 0.03,
    max(max(eval_rates), max(adv_rates)) * 0.96,
    r"$\tau$=3.25",
    rotation=90,
    va="top",
    fontsize=10,
)

ax.set_title("Threshold sweep: retrieval activation rate")
ax.set_xlabel(r"Threshold $\tau$")
ax.set_ylabel("Total retrieval activation rate")
ax.set_xticks(list(x))
ax.set_xticklabels(tau_labels)
ax.legend()

# 핵심 수정: 위에 살짝 여유
y_max = max(max(eval_rates), max(adv_rates))
ax.set_ylim(0, max(0.56, y_max * 1.10))

fig.tight_layout()

out_png = OUT_DIR / "tau_activation_main.png"
out_pdf = OUT_DIR / "tau_activation_main.pdf"

fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")

print("[OK] saved:", out_png)
print("[OK] saved:", out_pdf)
print("[INFO] y_max:", y_max)
print("[INFO] ylim upper:", max(0.56, y_max * 1.10))
