import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator import load_model, generate
from src.gate import build_gate_prompt, parse_gate_output


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def has_soft_scores(row):
    return (
        row.get("u_info") is not None
        and row.get("u_cope") is not None
        and row.get("u_spec") is not None
    )


def get_original_gate_system():
    """
    Load GATE_SYSTEM from scripts/generate_responses.py if it exists.
    This keeps the retrospective scoring aligned with the original pipeline.
    """
    path = ROOT / "scripts" / "generate_responses.py"
    spec = importlib.util.spec_from_file_location("generate_responses_original", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(
        mod,
        "GATE_SYSTEM",
        "You are a careful retrieval gate evaluator for mental-health QA."
    )


@torch.inference_mode()
def call_generate(model, tok, gate_system, gate_user, gate_max_new):
    """
    Try the same call shape used in generate_responses.py:
    generate(model, tok, GATE_SYSTEM, gate_user, max_new=gate_max_new)
    Fallbacks are only for signature compatibility.
    """
    trials = [
        lambda: generate(model, tok, gate_system, gate_user, max_new=gate_max_new),
        lambda: generate(model, tok, gate_system, gate_user, max_new=gate_max_new, do_sample=False),
        lambda: generate(model, tok, gate_user, max_new=gate_max_new),
        lambda: generate(model, tok, gate_user, max_new=gate_max_new, do_sample=False),
    ]

    last_err = None
    for fn in trials:
        try:
            return fn()
        except TypeError as e:
            last_err = e
            continue

    raise RuntimeError(f"Could not call generate with known signatures. Last error: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gated", required=True)
    ap.add_argument("--closed", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate_max_new", type=int, default=180)
    args = ap.parse_args()

    gated_rows = load_jsonl(args.gated)
    closed_rows = load_jsonl(args.closed)

    assert len(gated_rows) == len(closed_rows), (
        f"Line mismatch: gated={len(gated_rows)} closed={len(closed_rows)}"
    )

    missing = [i for i, r in enumerate(gated_rows) if not has_soft_scores(r)]
    print(f"[INFO] missing soft-score rows: {len(missing)} -> {missing}")

    if not missing:
        save_jsonl(gated_rows, args.out)
        print("[DONE] nothing to fill")
        return

    gate_system = get_original_gate_system()
    print("[INFO] using original gate prompt/parser:")
    print("       src.gate.build_gate_prompt")
    print("       src.gate.parse_gate_output")
    print("[INFO] using original generator:")
    print("       src.generator.load_model")
    print("       src.generator.generate")
    print("[INFO] loading model...")

    model, tok = load_model(args.base, args.adapter)

    filled = 0
    failed = 0

    for i in missing:
        g = gated_rows[i]
        c = closed_rows[i]

        question = g.get("question") or c.get("question")
        draft = g.get("draft") or c.get("response") or c.get("draft") or c.get("answer")

        if not question or not draft:
            print(f"[WARN] idx={i}: missing question/draft. skipped.")
            failed += 1
            continue

        gate_user = build_gate_prompt(question, draft)
        gate_raw = call_generate(model, tok, gate_system, gate_user, args.gate_max_new)
        gate_scores = parse_gate_output(gate_raw)

        u_info = gate_scores.get("u_info")
        u_cope = gate_scores.get("u_cope")
        u_spec = gate_scores.get("u_spec")

        if u_info is None or u_cope is None or u_spec is None:
            print(f"[WARN] idx={i}: parsed scores missing. raw={gate_raw[:300]}")
            failed += 1
            continue

        g["u_info"] = u_info
        g["u_cope"] = u_cope
        g["u_spec"] = u_spec
        g["mean_need"] = (u_info + u_cope + u_spec) / 3.0
        g["gate_reason"] = gate_scores.get("brief_reason")
        g["gate_raw"] = gate_scores.get("raw_gate_text", gate_raw)

        # diagnostic-only marker
        g["soft_score_filled_for_analysis"] = True
        g["soft_score_fill_basis"] = "original_gate_prompt_on_closed_book_draft"
        g["retrieve_decision_unchanged"] = True

        filled += 1
        print(
            f"[OK] idx={i} "
            f"u_info={u_info} u_cope={u_cope} u_spec={u_spec} "
            f"mean={g['mean_need']:.2f}"
        )

    save_jsonl(gated_rows, args.out)
    print(f"\n[DONE] filled={filled}, failed={failed}, saved={args.out}")


if __name__ == "__main__":
    main()
