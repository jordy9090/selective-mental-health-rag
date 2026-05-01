import argparse
import json
import os
import re
import sys
from pathlib import Path

from tqdm import tqdm
from datasets import load_dataset, load_dataset_builder

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.generator import load_model, generate
from src.retriever import BM25Retriever
from src.gate import (
    detect_hard_safety,
    build_gate_prompt,
    parse_gate_output,
    decide_retrieval,
    allowed_families_for_route,
)


SYS_BASE = (
    "You are a professional therapist providing mental health support online. "
    "Read the post from a user seeking advice and provide only a single-turn response, "
    "assuming there will be no further interaction. Be empathetic, specific, and safe. "
    "Do not provide medication advice or clinical diagnosis."
)

SYS_RAG = SYS_BASE + (
    " You are given evidence excerpts from therapy manuals, psychoeducation materials, "
    "and safety guidelines. Use them only when relevant. Do not copy verbatim."
)

SYS_SAFETY_RAG = SYS_BASE + (
    " This is a safety-sensitive mental health question. "
    "Use the provided safety guideline excerpts only when relevant. "
    "Prioritize immediate safety, crisis support, trusted-person help-seeking, and professional help. "
    "Do not diagnose. Do not provide medication instructions. "
    "If there is immediate danger, encourage contacting local emergency services or a crisis hotline."
)

SYS_GATE = (
    "You are a careful retrieval gate evaluator for mental-health QA. "
    "Return JSON only. Do not include markdown or extra text."
)

ADV_FAILURE_KEYS = [
    "apathetic",
    "assumptions",
    "symptoms",
    "judgmental",
    "medication",
    "therapy",
]


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def format_user_rag(question, evidence):
    blocks = [
        f"[Evidence {i} | {c.get('source_family', 'unknown')}]\n{c.get('text', '')}"
        for i, (c, _) in enumerate(evidence, 1)
    ]

    if not blocks:
        return f"Evidence:\nNone\n\nUser question:\n{question}"

    return "Evidence:\n" + "\n\n".join(blocks) + f"\n\nUser question:\n{question}"


def extract_question(ex):
    q_text = clean_text(ex.get("questionText", ""))
    q_title = clean_text(ex.get("questionTitle", ""))

    if q_text:
        if q_title and q_title.lower() not in q_text.lower():
            return f"{q_title}\n{q_text}"
        return q_text
    if q_title:
        return q_title

    for k in [
        "question",
        "query",
        "patient_question",
        "post",
        "text",
        "input",
        "Question",
        "prompt",
    ]:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return clean_text(v)

    raise KeyError(f"no question-like field in {list(ex.keys())}")


def extract_uid(ex, question):
    for k in ["questionID", "question_id", "qid", "id", "ID"]:
        v = ex.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return question


def build_unique_rows(ds):
    cols = set(ds.column_names)

    # CounselBench-Adv: wide format
    if set(ADV_FAILURE_KEYS).issubset(cols):
        unique_rows = []
        for row_idx, ex in enumerate(ds):
            for failure_mode in ADV_FAILURE_KEYS:
                q = clean_text(ex.get(failure_mode, ""))
                if not q:
                    continue

                unique_rows.append({
                    "uid": f"adv_{row_idx}_{failure_mode}",
                    "question": q,
                    "raw": {
                        "source_row": row_idx,
                        "failure_mode": failure_mode,
                        "original_row": dict(ex),
                    },
                })
        return unique_rows

    seen, unique_rows = set(), []
    for ex in ds:
        try:
            q = extract_question(ex)
        except KeyError:
            continue

        uid = extract_uid(ex, q)
        if uid in seen:
            continue
        seen.add(uid)

        unique_rows.append({
            "uid": uid,
            "question": q,
            "raw": dict(ex),
        })

    return unique_rows


def choose_split(dataset_name, requested_split=None):
    builder = load_dataset_builder(dataset_name)
    split_names = list(builder.info.splits.keys())

    if not split_names:
        raise ValueError(f"No splits found for dataset: {dataset_name}")

    if requested_split is not None:
        if requested_split in split_names:
            return requested_split, split_names
        print(f"[WARN] split='{requested_split}' not found. available={split_names}")

    if "test" in split_names:
        return "test", split_names
    if "train" in split_names:
        return "train", split_names
    return split_names[0], split_names


def serialize_evidence(ev):
    return [
        {
            "doc_id": c.get("doc_id"),
            "chunk_id": c.get("chunk_id"),
            "source_family": c.get("source_family"),
            "score": s,
            "text": c.get("text", "")[:600],
        }
        for c, s in ev
    ]


def run_no_retrieval(model, tok, question, max_new):
    response = generate(
        model,
        tok,
        SYS_BASE,
        question,
        max_new=max_new,
        do_sample=True,
        temperature=0.7,
        top_p=1.0,
    )
    return response, []


def run_always_retrieval(model, tok, retriever, question, top_k, max_new):
    ev = retriever.retrieve(question, top_k=top_k)
    user_p = format_user_rag(question, ev)

    response = generate(
        model,
        tok,
        SYS_RAG,
        user_p,
        max_new=max_new,
        do_sample=True,
        temperature=0.7,
        top_p=1.0,
    )
    return response, ev


def run_gated_retrieval(model, tok, retriever, question, top_k, max_new, gate_max_new):
    """
    Gated retrieval flow:
    1. query-only hard safety trigger
    2. if safety: retrieve from safety corpus + safety-first generation
    3. else: generate closed-book draft
    4. soft utility gate on (question, draft)
    5. retrieve if decision says yes
    6. if retrieve: regenerate with evidence
       else: use draft as final response
    """

    # 1) hard safety first
    hard = detect_hard_safety(question)
    r_safe = int(hard.get("r_safe", 0))

    draft = None
    gate_raw = None
    gate_scores = {
        "u_info": None,
        "u_cope": None,
        "u_spec": None,
        "dominant_route": None,
        "brief_reason": None,
        "raw_gate_text": None,
    }

    # 2) safety override
    if r_safe == 1:
        decision = decide_retrieval(r_safe=1, gate_scores={})
        route = decision["route"]
        allowed = allowed_families_for_route(route)

        ev = retriever.retrieve(
            question,
            top_k=top_k,
            allowed_families=allowed,
        )

        user_p = format_user_rag(question, ev)
        response = generate(
            model,
            tok,
            SYS_SAFETY_RAG,
            user_p,
            max_new=max_new,
            do_sample=True,
            temperature=0.7,
            top_p=1.0,
        )

        return {
            "response": response,
            "draft": draft,
            "evidence": ev,
            "r_safe": r_safe,
            "safety_matched_categories": hard.get("matched_categories", []),
            "safety_matched_patterns": hard.get("matched_patterns", []),
            "u_info": None,
            "u_cope": None,
            "u_spec": None,
            "mean_need": None,
            "retrieve": True,
            "route": route,
            "allowed_families": allowed,
            "gate_reason": "hard safety trigger",
            "gate_raw": None,
        }

    # 3) closed-book draft
    draft = generate(
        model,
        tok,
        SYS_BASE,
        question,
        max_new=max_new,
        do_sample=True,
        temperature=0.7,
        top_p=1.0,
    )

    # 4) soft gate, deterministic
    gate_user = build_gate_prompt(question, draft)
    gate_raw = generate(
        model,
        tok,
        SYS_GATE,
        gate_user,
        max_new=gate_max_new,
        do_sample=False,
    )

    gate_scores = parse_gate_output(gate_raw)

    # 5) decision
    decision = decide_retrieval(r_safe=0, gate_scores=gate_scores)
    retrieve = bool(decision["retrieve"])
    route = decision["route"]
    allowed = allowed_families_for_route(route)

    # 6) final response
    if retrieve:
        ev = retriever.retrieve(
            question,
            top_k=top_k,
            allowed_families=allowed,
        )

        user_p = format_user_rag(question, ev)
        response = generate(
            model,
            tok,
            SYS_RAG,
            user_p,
            max_new=max_new,
            do_sample=True,
            temperature=0.7,
            top_p=1.0,
        )
    else:
        ev = []
        response = draft

    return {
        "response": response,
        "draft": draft,
        "evidence": ev,
        "r_safe": r_safe,
        "safety_matched_categories": hard.get("matched_categories", []),
        "safety_matched_patterns": hard.get("matched_patterns", []),
        "u_info": gate_scores.get("u_info"),
        "u_cope": gate_scores.get("u_cope"),
        "u_spec": gate_scores.get("u_spec"),
        "mean_need": decision.get("mean_need"),
        "retrieve": retrieve,
        "route": route,
        "allowed_families": allowed,
        "gate_reason": gate_scores.get("brief_reason"),
        "gate_raw": gate_scores.get("raw_gate_text"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["no_retrieval", "always_retrieval", "gated_retrieval"],
        required=True,
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--base", default="google/gemma-4-E4B-it")
    ap.add_argument("--adapter", default="./outputs/gemma4-e4b-it-mh16k")
    ap.add_argument("--chunks", default="data/processed/chunks.jsonl")
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_new", type=int, default=400)
    ap.add_argument("--gate_max_new", type=int, default=180)
    args = ap.parse_args()

    os.makedirs(Path(args.out).parent, exist_ok=True)

    chosen_split, split_names = choose_split(args.dataset, args.split)
    print(f"[INFO] available splits: {split_names}")
    print(f"[INFO] using split: {chosen_split}")

    ds = load_dataset(args.dataset, split=chosen_split)
    print(f"[INFO] loaded rows: {len(ds)}")
    print(f"[INFO] columns: {ds.column_names}")
    if len(ds) > 0:
        print(f"[DEBUG] first row keys: {list(ds[0].keys())}")
        print(f"[DEBUG] first row sample: {ds[0]}")

    unique_rows = build_unique_rows(ds)

    if args.limit is not None:
        unique_rows = unique_rows[:args.limit]

    print(f"[INFO] unique questions: {len(unique_rows)}")

    needs_retriever = args.mode in ["always_retrieval", "gated_retrieval"]
    retriever = BM25Retriever(args.chunks) if needs_retriever else None

    model, tok = load_model(args.base, args.adapter)

    with open(args.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(tqdm(unique_rows)):
            q = row["question"]

            if args.mode == "no_retrieval":
                response, ev = run_no_retrieval(
                    model=model,
                    tok=tok,
                    question=q,
                    max_new=args.max_new,
                )
                extra = {
                    "draft": None,
                    "r_safe": None,
                    "safety_matched_categories": [],
                    "safety_matched_patterns": [],
                    "u_info": None,
                    "u_cope": None,
                    "u_spec": None,
                    "mean_need": None,
                    "retrieve": False,
                    "route": "none",
                    "allowed_families": [],
                    "gate_reason": None,
                    "gate_raw": None,
                }

            elif args.mode == "always_retrieval":
                response, ev = run_always_retrieval(
                    model=model,
                    tok=tok,
                    retriever=retriever,
                    question=q,
                    top_k=args.top_k,
                    max_new=args.max_new,
                )
                extra = {
                    "draft": None,
                    "r_safe": None,
                    "safety_matched_categories": [],
                    "safety_matched_patterns": [],
                    "u_info": None,
                    "u_cope": None,
                    "u_spec": None,
                    "mean_need": None,
                    "retrieve": True,
                    "route": "full_corpus",
                    "allowed_families": None,
                    "gate_reason": None,
                    "gate_raw": None,
                }

            else:
                result = run_gated_retrieval(
                    model=model,
                    tok=tok,
                    retriever=retriever,
                    question=q,
                    top_k=args.top_k,
                    max_new=args.max_new,
                    gate_max_new=args.gate_max_new,
                )
                response = result["response"]
                ev = result["evidence"]
                extra = {
                    "draft": result["draft"],
                    "r_safe": result["r_safe"],
                    "safety_matched_categories": result["safety_matched_categories"],
                    "safety_matched_patterns": result["safety_matched_patterns"],
                    "u_info": result["u_info"],
                    "u_cope": result["u_cope"],
                    "u_spec": result["u_spec"],
                    "mean_need": result["mean_need"],
                    "retrieve": result["retrieve"],
                    "route": result["route"],
                    "allowed_families": result["allowed_families"],
                    "gate_reason": result["gate_reason"],
                    "gate_raw": result["gate_raw"],
                }

            rec = {
                "idx": i,
                "uid": row["uid"],
                "question": q,
                "mode": args.mode,
                "dataset": args.dataset,
                "split": chosen_split,
                "response": response,
                "evidence": serialize_evidence(ev),
                "raw_meta": row["raw"],
            }
            rec.update(extra)

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    print(f"[DONE] wrote: {args.out}")


if __name__ == "__main__":
    main()
