import json, csv, re
from pathlib import Path

OUT = Path("outputs/case_studies")
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "eval_no": Path("outputs/eval/no_retrieval.judged.v2.jsonl"),
    "eval_always": Path("outputs/eval/always_retrieval.judged.v2.jsonl"),
    "eval_gated": Path("outputs/eval/gated_retrieval.judged.v2.jsonl"),
    "adv_no": Path("outputs/adv/no_retrieval.judged.jsonl"),
    "adv_always": Path("outputs/adv/always_retrieval.judged.jsonl"),
    "adv_gated": Path("outputs/adv/gated_retrieval.judged.v2.jsonl"),
}

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def by_uid(rows):
    return {str(r.get("uid", r.get("idx"))): r for r in rows}

def short(s, n=240):
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s[:n] + ("..." if len(s) > n else "")

def flatten(x, prefix=""):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            nk = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten(v, nk))
    else:
        out[prefix] = x
    return out

def norm_label(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    s = str(v).strip().lower()
    if not s:
        return ""
    if "not sure" in s or "unsure" in s or "unclear" in s:
        return "not_sure"
    if re.search(r"\byes\b", s) or "present" in s or s == "true":
        return "yes"
    if re.search(r"\bno\b", s) or "absent" in s or s == "false":
        return "no"
    return s

def find_values(record, must_tokens):
    scores = flatten(record.get("judge_scores", {}))
    hits = []
    for k, v in scores.items():
        kl = k.lower()
        if all(t in kl for t in must_tokens):
            hits.append((k, v))
    return hits

def first_metric(record, must_tokens):
    vals = find_values(record, must_tokens)
    if not vals:
        return ""
    # prefer non-rationale fields
    vals = sorted(vals, key=lambda kv: ("reason" in kv[0].lower() or "rationale" in kv[0].lower(), len(kv[0])))
    return vals[0][1]

def parse_medical_label(record):
    # 1) judge_scores 안에서 medical + advice key 찾기
    vals = find_values(record, ["medical", "advice"])
    for k, v in vals:
        lab = norm_label(v)
        if lab in {"yes", "no", "not_sure"}:
            return lab

    # 2) judge_raw 안에서 json-ish 패턴 찾기
    raw = str(record.get("judge_raw", ""))
    pats = [
        r'"medical[_\s-]*advice"[^:\n]*:\s*"?(yes|no|not sure|unsure)',
        r'medical advice[^:\n]*:\s*(yes|no|not sure|unsure)',
        r'Unauthorized medical advice[^:\n]*:\s*(yes|no|not sure|unsure)',
    ]
    for p in pats:
        m = re.search(p, raw, flags=re.I)
        if m:
            return norm_label(m.group(1))
    return ""

def parse_failure_label(record):
    vals = flatten(record.get("judge_scores", {}))
    preferred = []
    for k, v in vals.items():
        kl = k.lower()
        if any(t in kl for t in ["failure", "present", "label", "judgment", "answer"]):
            preferred.append((k, v))
    for k, v in preferred:
        lab = norm_label(v)
        if lab in {"yes", "no", "not_sure"}:
            return lab

    raw = str(record.get("judge_raw", ""))
    pats = [
        r'"failure[_\s-]*present"[^:\n]*:\s*"?(yes|no|not sure|unsure)',
        r'"present"[^:\n]*:\s*"?(yes|no|not sure|unsure)',
        r'"answer"[^:\n]*:\s*"?(yes|no|not sure|unsure)',
        r'target failure[^:\n]*:\s*(yes|no|not sure|unsure)',
    ]
    for p in pats:
        m = re.search(p, raw, flags=re.I)
        if m:
            return norm_label(m.group(1))
    return ""

def metric(record, name):
    return first_metric(record, [name])

def evidence_summary(record):
    ev = record.get("evidence") or []
    parts = []
    for e in ev[:3]:
        parts.append(f"{e.get('source_family','?')}:{e.get('doc_id','?')}#{e.get('chunk_id','?')}")
    return " | ".join(parts)

def write_md_case(fp, title, uid, question, blocks, interpretation_hint):
    fp.write(f"\n\n## {title}\n\n")
    fp.write(f"**UID:** `{uid}`\n\n")
    fp.write(f"**Question:** {short(question, 900)}\n\n")
    for name, r in blocks:
        if not r:
            continue
        fp.write(f"### {name}\n\n")
        fp.write(f"- medical_advice: `{parse_medical_label(r)}`\n")
        fp.write(f"- failure_label: `{parse_failure_label(r)}`\n")
        fp.write(f"- overall: `{metric(r, 'overall')}` | empathy: `{metric(r, 'empathy')}` | specificity: `{metric(r, 'specificity')}`\n")
        if "retrieve" in r:
            fp.write(f"- retrieve: `{r.get('retrieve')}` | route: `{r.get('route')}` | mean_need: `{r.get('mean_need')}` | r_safe: `{r.get('r_safe')}`\n")
        fp.write(f"- evidence: {evidence_summary(r)}\n\n")
        fp.write("**Response excerpt:**\n\n")
        fp.write(short(r.get("response"), 1500) + "\n\n")
        ev = r.get("evidence") or []
        if ev:
            fp.write("**Top evidence excerpt:**\n\n")
            for e in ev[:2]:
                fp.write(f"- `{e.get('source_family')}:{e.get('doc_id')}#{e.get('chunk_id')}`: {short(e.get('text'), 500)}\n")
            fp.write("\n")
    fp.write(f"**Interpretation hint:** {interpretation_hint}\n\n")
    fp.write("---\n")

def main():
    data = {k: load_jsonl(p) for k, p in FILES.items()}
    E_no = by_uid(data["eval_no"])
    E_always = by_uid(data["eval_always"])
    E_gated = by_uid(data["eval_gated"])

    A_no = by_uid(data["adv_no"])
    A_always = by_uid(data["adv_always"])
    A_gated = by_uid(data["adv_gated"])

    # =========================
    # Case 1: Always retrieval medical-advice YES
    # =========================
    med_rows = []
    for uid, ar in E_always.items():
        nr = E_no.get(uid)
        gr = E_gated.get(uid)
        a_med = parse_medical_label(ar)
        n_med = parse_medical_label(nr) if nr else ""
        g_med = parse_medical_label(gr) if gr else ""
        if a_med == "yes":
            med_rows.append({
                "uid": uid,
                "question": short(ar.get("question"), 220),
                "no_med": n_med,
                "always_med": a_med,
                "gated_med": g_med,
                "no_overall": metric(nr, "overall") if nr else "",
                "always_overall": metric(ar, "overall"),
                "gated_overall": metric(gr, "overall") if gr else "",
                "no_specificity": metric(nr, "specificity") if nr else "",
                "always_specificity": metric(ar, "specificity"),
                "gated_specificity": metric(gr, "specificity") if gr else "",
                "always_evidence": evidence_summary(ar),
                "always_response": short(ar.get("response"), 420),
            })

    med_csv = OUT / "medical_advice_candidates.csv"
    with open(med_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["uid","question","no_med","always_med","gated_med","no_overall","always_overall","gated_overall","no_specificity","always_specificity","gated_specificity","always_evidence","always_response"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(med_rows)

    med_md = OUT / "medical_advice_cases.md"
    with open(med_md, "w", encoding="utf-8") as fp:
        fp.write("# Candidate Case Study 1: Always Retrieval Medical-Advice Risk\n")
        fp.write(f"\nTotal candidates where always retrieval is judged as medical_advice=YES: **{len(med_rows)}**\n")
        selected = med_rows[:10]
        for row in selected:
            uid = row["uid"]
            write_md_case(
                fp,
                "Always Retrieval increases medical-advice risk",
                uid,
                E_always[uid].get("question"),
                [("No Retrieval", E_no.get(uid)), ("Always Retrieval", E_always.get(uid)), ("Gated Retrieval", E_gated.get(uid))],
                "Use this case if always retrieval becomes more concrete/specific but crosses a professional-boundary or medical-advice line."
            )

    # =========================
    # Case 2: Gated retrieval symptoms failure
    # =========================
    sym_rows = []
    for uid, gr in A_gated.items():
        if str(gr.get("target_failure_mode", "")).lower() != "symptoms":
            continue
        nr = A_no.get(uid)
        ar = A_always.get(uid)
        g_lab = parse_failure_label(gr)
        n_lab = parse_failure_label(nr) if nr else ""
        a_lab = parse_failure_label(ar) if ar else ""
        if g_lab == "yes":
            sym_rows.append({
                "uid": uid,
                "question": short(gr.get("question"), 260),
                "target_failure_mode": gr.get("target_failure_mode"),
                "no_failure": n_lab,
                "always_failure": a_lab,
                "gated_failure": g_lab,
                "retrieve": gr.get("retrieve"),
                "route": gr.get("route"),
                "mean_need": gr.get("mean_need"),
                "r_safe": gr.get("r_safe"),
                "gated_evidence": evidence_summary(gr),
                "gated_response": short(gr.get("response"), 460),
            })

    sym_csv = OUT / "symptoms_failure_candidates.csv"
    with open(sym_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["uid","question","target_failure_mode","no_failure","always_failure","gated_failure","retrieve","route","mean_need","r_safe","gated_evidence","gated_response"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sym_rows)

    sym_md = OUT / "symptoms_failure_cases.md"
    with open(sym_md, "w", encoding="utf-8") as fp:
        fp.write("# Candidate Case Study 2: Symptoms Failure under Gated Retrieval\n")
        fp.write(f"\nTotal candidates where target=symptoms and gated retrieval is judged failure=YES: **{len(sym_rows)}**\n")
        selected = sym_rows[:10]
        for row in selected:
            uid = row["uid"]
            write_md_case(
                fp,
                "Gated Retrieval still triggers symptoms failure",
                uid,
                A_gated[uid].get("question"),
                [("No Retrieval", A_no.get(uid)), ("Always Retrieval", A_always.get(uid)), ("Gated Retrieval", A_gated.get(uid))],
                "Use this case if the model speculates about symptoms/conditions from an underspecified single-turn query. This supports the limitation: gating controls retrieval exposure but cannot fully solve clinical uncertainty."
            )

    print("\nDONE.")
    print(f"- {med_csv}")
    print(f"- {med_md}")
    print(f"- {sym_csv}")
    print(f"- {sym_md}")
    print("\nOpen the .md files first. Pick ONE clean case from each, not the first random goblin that appears.")
    
if __name__ == "__main__":
    main()
