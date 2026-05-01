import json
import re
from typing import Any, Dict, List, Optional

# High-recall safety pattern seed:
# - 목적: 위험 질문을 놓치지 않는 것
# - false positive 약간 허용, false negative 최소화
SAFETY_PATTERN_SETS = {
    "self_harm_suicide": [
        r"\bsuicide\b",
        r"\bsuicidal\b",
        r"kill myself",
        r"want to die",
        r"end my life",
        r"self[- ]?harm",
        r"hurt myself",
        r"cut myself",
        r"\boverdose\b",
        r"can't go on",
        r"cannot go on",
        r"can't stay safe",
    ],
    "harm_to_others": [
        r"hurt someone",
        r"kill him",
        r"kill her",
        r"kill them",
        r"harm someone",
    ],
    "acute_crisis": [
        r"hearing voices",
        r"seeing things",
        r"\bpsychosis\b",
        r"\bparanoid\b",
        r"out of control",
        r"haven't slept for days",
        r"haven’t slept for days",
    ],
    "abuse_immediate_danger": [
        r"unsafe at home",
        r"\babuse\b",
        r"being abused",
        r"\bviolence\b",
        r"\bassault\b",
        r"\bemergency\b",
    ],
    "urgent_medical_boundary": [
        r"should i stop my meds",
        r"should i stop my medication",
        r"change my medication",
        r"change dosage",
        r"mix alcohol with meds",
        r"diagnose me",
    ],
}

COMPILED_PATTERNS = {
    cat: [re.compile(pat, flags=re.IGNORECASE) for pat in patterns]
    for cat, patterns in SAFETY_PATTERN_SETS.items()
}


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_hard_safety(question: str) -> Dict[str, Any]:
    """
    Query-only hard safety trigger.
    Returns:
    {
        "r_safe": 0 or 1,
        "route": "safety" or "none",
        "matched_categories": [...],
        "matched_patterns": [...]
    }
    """
    q = normalize_text(question).lower()
    matched_categories: List[str] = []
    matched_patterns: List[str] = []

    for category, patterns in COMPILED_PATTERNS.items():
        for pat in patterns:
            if pat.search(q):
                matched_categories.append(category)
                matched_patterns.append(pat.pattern)
                break  # category별 첫 매치만 저장

    r_safe = 1 if matched_categories else 0
    return {
        "r_safe": r_safe,
        "route": "safety" if r_safe == 1 else "none",
        "matched_categories": matched_categories,
        "matched_patterns": matched_patterns,
    }


def build_gate_prompt(question: str, draft: str) -> str:
    """
    Soft utility gate prompt (1-5 scale).
    JSON only를 강하게 요구.
    """
    q = normalize_text(question)
    d0 = normalize_text(draft)

    return f"""
You are evaluating whether external guideline retrieval would improve a single-turn mental-health response.

Rate each item from 1 to 5.
Return JSON only. Do not include markdown or extra text.

[User Question]
{q}

[Closed-book Draft]
{d0}

Evaluate:

1. info_need
Does the user question require psychoeducation, explanatory grounding, or factual clarification that the draft does not sufficiently provide?
1 = explanation already sufficient
5 = important explanation is missing or likely inaccurate

2. coping_need
Does the user need concrete, actionable coping steps or next-step guidance that the draft does not sufficiently provide?
1 = coping guidance already sufficient
5 = actionable coping guidance is clearly missing

3. specificity_need
Is the draft too generic or insufficiently tailored to the user's specific concern?
1 = clearly tailored and specific
5 = highly generic / could apply to almost any user

4. dominant_route
Choose exactly one:
- "coping"
- "psychoeducation"
- "all_non_safety"
- "none"

Guidance:
- choose "coping" if the biggest gap is actionable next-step guidance
- choose "psychoeducation" if the biggest gap is explanatory grounding
- choose "all_non_safety" if the draft is mainly too generic or multiple non-safety gaps coexist
- choose "none" only if retrieval is not likely to help

Return exactly:
{{
  "info_need": 1-5,
  "coping_need": 1-5,
  "specificity_need": 1-5,
  "dominant_route": "coping|psychoeducation|all_non_safety|none",
  "brief_reason": "one short sentence"
}}
""".strip()


def _extract_first_json_block(text: str) -> Optional[str]:
    """
    모델이 쓸데없는 말을 섞어도 첫 JSON object만 뽑아냄.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _clamp_score(x: Any, default: int = 3) -> int:
    try:
        val = int(round(float(x)))
    except Exception:
        val = default
    return max(1, min(5, val))


def parse_gate_output(raw_text: str) -> Dict[str, Any]:
    """
    gate model raw output -> normalized dict
    """
    block = _extract_first_json_block(raw_text or "")
    if block is None:
        # fallback: 파싱 실패 시 중립값
        return {
            "u_info": 3,
            "u_cope": 3,
            "u_spec": 3,
            "dominant_route": "all_non_safety",
            "brief_reason": "Failed to parse JSON; using neutral fallback.",
            "raw_gate_text": raw_text,
        }

    try:
        obj = json.loads(block)
    except Exception:
        return {
            "u_info": 3,
            "u_cope": 3,
            "u_spec": 3,
            "dominant_route": "all_non_safety",
            "brief_reason": "Invalid JSON; using neutral fallback.",
            "raw_gate_text": raw_text,
        }

    route = str(obj.get("dominant_route", "all_non_safety")).strip().lower()
    if route not in {"coping", "psychoeducation", "all_non_safety", "none"}:
        route = "all_non_safety"

    return {
        "u_info": _clamp_score(obj.get("info_need", 3)),
        "u_cope": _clamp_score(obj.get("coping_need", 3)),
        "u_spec": _clamp_score(obj.get("specificity_need", 3)),
        "dominant_route": route,
        "brief_reason": str(obj.get("brief_reason", "")).strip(),
        "raw_gate_text": raw_text,
    }


def decide_retrieval(r_safe: int, gate_scores: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decision rule:
    - if hard safety -> retrieve safety
    - elif max(u_info, u_cope) >= 4 -> retrieve
    - elif mean(u_info, u_cope, u_spec) >= 2.75 -> retrieve
    - else no retrieval

    routing:
    - safety -> safety
    - else use dominant_route, but u_spec alone should not force a specific family
    """
    if r_safe == 1:
        return {
            "retrieve": True,
            "route": "safety",
            "mean_need": None,
        }

    u_info = int(gate_scores["u_info"])
    u_cope = int(gate_scores["u_cope"])
    u_spec = int(gate_scores["u_spec"])
    mean_need = (u_info + u_cope + u_spec) / 3.0

    retrieve = False
    if max(u_info, u_cope) >= 4:
        retrieve = True
    elif mean_need >= 2.75:
        retrieve = True

    if not retrieve:
        return {
            "retrieve": False,
            "route": "none",
            "mean_need": round(mean_need, 4),
        }

    # route 결정
    route = gate_scores.get("dominant_route", "all_non_safety")
    if route not in {"coping", "psychoeducation", "all_non_safety"}:
        # fallback: score 기반
        if u_cope >= u_info and u_cope >= 4:
            route = "coping"
        elif u_info > u_cope and u_info >= 4:
            route = "psychoeducation"
        else:
            route = "all_non_safety"

    return {
        "retrieve": True,
        "route": route,
        "mean_need": round(mean_need, 4),
    }


def allowed_families_for_route(route: str) -> List[str]:
    route = (route or "").strip().lower()
    if route == "safety":
        return ["safety"]
    if route == "coping":
        return ["coping"]
    if route == "psychoeducation":
        return ["psychoeducation"]
    if route == "all_non_safety":
        return ["coping", "psychoeducation"]
    return []
