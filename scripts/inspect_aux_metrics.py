import argparse
import json
from collections import Counter
from pathlib import Path
import pandas as pd

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def get_scores(row):
    # support both possible field names
    s = row.get("judge_scores") or row.get("scores") or row.get("parsed") or {}
    return s

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Format: METHOD=path.jsonl"
    )
    parser.add_argument("--out", default="outputs/eval/aux_metric_sanity.csv")
    args = parser.parse_args()

    records = []

    for item in args.inputs:
        method, path = item.split("=", 1)
        rows = load_jsonl(path)

        fc_vals = []
        tox_vals = []
        fc_counter = Counter()
        tox_counter = Counter()
        n_ok = 0

        for row in rows:
            s = get_scores(row)
            if not s:
                continue

            fc = str(s.get("factual_consistency", "")).strip()
            tox = s.get("toxicity", None)

            if fc:
                fc_counter[fc] += 1
                if fc in {"1", "2", "3", "4"}:
                    fc_vals.append(int(fc))

            if tox is not None:
                try:
                    tox = int(tox)
                    tox_vals.append(tox)
                    tox_counter[str(tox)] += 1
                except Exception:
                    pass

            n_ok += 1

        records.append({
            "method": method,
            "n": len(rows),
            "n_scored": n_ok,
            "factual_mean_excl_unsure": sum(fc_vals) / len(fc_vals) if fc_vals else None,
            "factual_4_rate": fc_counter["4"] / max(1, len(rows)),
            "factual_unsure_rate": fc_counter["I am not sure"] / max(1, len(rows)),
            "toxicity_mean": sum(tox_vals) / len(tox_vals) if tox_vals else None,
            "toxicity_1_rate": tox_counter["1"] / max(1, len(rows)),
            "fc_dist": dict(fc_counter),
            "tox_dist": dict(tox_counter),
        })

    df = pd.DataFrame(records)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"\n[DONE] wrote {args.out}")

if __name__ == "__main__":
    main()
