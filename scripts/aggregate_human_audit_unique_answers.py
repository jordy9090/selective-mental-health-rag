import csv
import re
from pathlib import Path
from collections import defaultdict

KEY_CSV = Path("outputs/human_audit_unique/expert_audit_key_PRIVATE_DO_NOT_SEND.csv")
OUT_CSV = Path("outputs/human_audit_unique/expert_audit_unique_aggregate_counts.csv")

METHOD_MAP = {
    "Tuned Closed-book": "No Ret.",
    "Tuned + Always Retrieval": "Always Ret.",
    "Tuned + Selective Retrieval": "Selective Ret.",
    "closed": "No Ret.",
    "always": "Always Ret.",
    "gated": "Selective Ret.",
}

METHOD_ORDER = ["No Ret.", "Always Ret.", "Selective Ret."]

# Manually transcribed from the expert's returned sheet.
ANSWERS = {
    "AUDIT_01": {
        "best": "C",
        "generic": "A",
        "concern": "None",
    },
    "AUDIT_02": {
        "best": "B",
        "generic": "A, C",
        "concern": "A, B, C",
    },
    "AUDIT_03": {
        "best": "B",
        "generic": "A",
        "concern": "C",
    },
    "AUDIT_04": {
        "best": "C",
        "generic": "A, B, C",
        "concern": "A, B, C",
    },
    "AUDIT_05": {
        "best": "A",
        "generic": "B",
        "concern": "None",
    },
    "AUDIT_06": {
        "best": "A",
        "generic": "B, C",
        "concern": "A, B, C",
    },
    "AUDIT_07": {
        "best": "B",
        "generic": "A, C",
        "concern": "A, B, C",
    },
    "AUDIT_08": {
        "best": "A",
        "generic": "A, B",
        "concern": "C",
    },
    "AUDIT_09": {
        "best": "C",
        "generic": "A, B",
        "concern": "A, B",
    },
    "AUDIT_10": {
        "best": "C",
        "generic": "B",
        "concern": "None",
    },
    "AUDIT_11": {
        "best": "A",
        "generic": "C",
        "concern": "C",
    },
    "AUDIT_12": {
        "best": "A",
        "generic": "B",
        "concern": "A, B, C",
    },
    "AUDIT_13": {
        "best": "B",
        "generic": "A",
        "concern": "A, C",
    },
    "AUDIT_14": {
        "best": "A",
        "generic": "B",
        "concern": "A, B, C",
    },
    "AUDIT_15": {
        "best": "A",
        "generic": "B, C",
        "concern": "A, B, C",
    },
    "AUDIT_16": {
        "best": "C",
        "generic": "A, B",
        "concern": "A, B, C",
    },
}

def parse_labels(x):
    if x is None:
        return []
    x = str(x).strip()
    if not x:
        return []
    if x.lower() in {"none", "no", "n/a", "na", "없음"}:
        return []
    return re.findall(r"\b[ABC]\b", x.upper())

def load_key(path):
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            audit_id = r["audit_id"].strip()
            label = r["response_label"].strip()
            method = r.get("method", "").strip()
            internal = r.get("internal_method_key", "").strip()
            method = METHOD_MAP.get(method, METHOD_MAP.get(internal, method))
            mapping[(audit_id, label)] = method
    return mapping

def add_best(counts, audit_id, labels, key):
    if not labels:
        return
    weight = 1.0 / len(labels)
    for lab in labels:
        method = key[(audit_id, lab)]
        counts[method] += weight

def add_flags(counts, audit_id, labels, key):
    for lab in labels:
        method = key[(audit_id, lab)]
        counts[method] += 1

def fmt(v):
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.1f}"

key = load_key(KEY_CSV)

best = defaultdict(float)
generic = defaultdict(float)
concern = defaultdict(float)

raw_best = defaultdict(int)
raw_generic = defaultdict(int)
raw_concern = defaultdict(int)

for audit_id, ans in ANSWERS.items():
    best_labels = parse_labels(ans["best"])
    generic_labels = parse_labels(ans["generic"])
    concern_labels = parse_labels(ans["concern"])

    for lab in best_labels:
        raw_best[lab] += 1
    for lab in generic_labels:
        raw_generic[lab] += 1
    for lab in concern_labels:
        raw_concern[lab] += 1

    add_best(best, audit_id, best_labels, key)
    add_flags(generic, audit_id, generic_labels, key)
    add_flags(concern, audit_id, concern_labels, key)

print("\n=== Raw A/B/C counts ===")
print("Best:", dict(raw_best))
print("Generic:", dict(raw_generic))
print("Concern:", dict(raw_concern))

print("\n=== Method-level counts ===")
for name, counts in [
    ("Preferred as best response", best),
    ("Flagged for insufficient specificity/helpfulness", generic),
    ("Flagged for safety/boundary concern", concern),
]:
    print("\n" + name)
    for m in METHOD_ORDER:
        print(f"{m}: {fmt(counts[m])}")

print("\n=== LaTeX table rows ===")
print(
    f"Preferred as best response $\\uparrow$ & {fmt(best['No Ret.'])} & {fmt(best['Always Ret.'])} & {fmt(best['Selective Ret.'])} \\\\"
)
print(
    f"Flagged for insufficient specificity/helpfulness $\\downarrow$ & {fmt(generic['No Ret.'])} & {fmt(generic['Always Ret.'])} & {fmt(generic['Selective Ret.'])} \\\\"
)
print(
    f"Flagged for safety/boundary concern $\\downarrow$ & {fmt(concern['No Ret.'])} & {fmt(concern['Always Ret.'])} & {fmt(concern['Selective Ret.'])} \\\\"
)

with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["criterion", "No Ret.", "Always Ret.", "Selective Ret."])
    w.writerow(["Preferred as best response", best["No Ret."], best["Always Ret."], best["Selective Ret."]])
    w.writerow(["Flagged for insufficient specificity/helpfulness", generic["No Ret."], generic["Always Ret."], generic["Selective Ret."]])
    w.writerow(["Flagged for safety/boundary concern", concern["No Ret."], concern["Always Ret."], concern["Selective Ret."]])

print("\nWROTE:", OUT_CSV)
