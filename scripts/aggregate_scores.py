import argparse, json, pandas as pd

NUMERIC = ["Overall", "Empathy", "Specificity", "Factual Consistency", "Toxicity"]

def to_num(x):
    if x is None: return None
    try: return float(str(x).split()[0])
    except: return None

def med_adv(x):
    if x is None: return None
    x = x.lower()
    if x.startswith("yes"): return 1.0
    if x.startswith("no"):  return 0.0
    return None

def summarize(path, name):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    df = pd.DataFrame([r["scores"] for r in rows])
    out = {col: pd.Series([to_num(v) for v in df[col]]).mean() for col in NUMERIC}
    out["Medical Advice (Yes%)"] = pd.Series([med_adv(v) for v in df["Medical Advice"]]).mean()
    out["N"] = len(rows)
    out["model"] = name
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True,
                    help="name=path pairs, e.g. no_ret=outputs/eval/no_retrieval.judged.jsonl")
    args = ap.parse_args()
    rows = []
    for kv in args.files:
        name, path = kv.split("=", 1)
        rows.append(summarize(path, name))
    df = pd.DataFrame(rows).set_index("model")
    print(df.round(3).to_string())
