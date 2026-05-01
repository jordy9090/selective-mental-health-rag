import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.in_file, encoding="utf-8")]

    sums = defaultdict(float)
    counts = defaultdict(int)

    medical_yes = 0
    medical_total = 0

    for row in rows:
        s = row.get("judge_scores")
        if not s:
            continue

        for k in ["overall", "empathy", "specificity", "toxicity"]:
            sums[k] += s[k]
            counts[k] += 1

        fc = s["factual_consistency"]
        if fc != "I am not sure":
            sums["factual_consistency"] += int(fc)
            counts["factual_consistency"] += 1

        if s["medical_advice"] != "I am not sure":
            medical_total += 1
            if s["medical_advice"] == "Yes":
                medical_yes += 1

    result = {}
    for k in ["overall", "empathy", "specificity", "toxicity", "factual_consistency"]:
        result[k + "_mean"] = round(sums[k] / counts[k], 4) if counts[k] else None

    result["medical_advice_yes_rate"] = round(medical_yes / medical_total, 4) if medical_total else None
    result["n_scored"] = sum(1 for r in rows if r.get("judge_scores") is not None)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
