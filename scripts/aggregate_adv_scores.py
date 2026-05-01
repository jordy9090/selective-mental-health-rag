import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.in_file, encoding="utf-8")]

    mode_yes = defaultdict(int)
    mode_total = defaultdict(int)

    valid_yes = 0
    valid_total = 0

    for row in rows:
        s = row.get("judge_scores")
        if not s:
            continue

        mode = row["target_failure_mode"]

        if s["target_failure_present"] != "I am not sure":
            mode_total[mode] += 1
            if s["target_failure_present"] == "Yes":
                mode_yes[mode] += 1

        if s["valid_response"] != "I am not sure":
            valid_total += 1
            if s["valid_response"] == "Yes":
                valid_yes += 1

    result = {}
    rates = []

    for mode in sorted(mode_total.keys()):
        rate = mode_yes[mode] / mode_total[mode] if mode_total[mode] else None
        result[f"{mode}_failure_rate"] = round(rate, 4) if rate is not None else None
        if rate is not None:
            rates.append(rate)

    result["macro_avg_failure_rate"] = round(sum(rates) / len(rates), 4) if rates else None
    result["valid_response_rate"] = round(valid_yes / valid_total, 4) if valid_total else None
    result["n_scored"] = sum(1 for r in rows if r.get("judge_scores") is not None)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
