"""LLM-as-judge for generated answers (one call per answer).

The judge returns three things:
- claims: the answer split into atomic claims, each marked supported / unsupported
  BY THE RETRIEVED CONTEXT (not by the judge's own knowledge). Faithfulness =
  supported / total. This claim-decomposition approach follows the RAGAS paper.
- relevance (1-5): does the answer address the question, regardless of correctness.
- correctness vs the golden reference answer: correct / partial / incorrect.

The judge is a different model family (Gemini) from the generator (gpt-oss) to
avoid self-preference bias. Its verdicts are spot-checked by hand with
scripts/audit_judge.py.
"""
from __future__ import annotations

from rag_eval.llm_utils import extract_json
from rag_eval.llm_calls import cached_chat

JUDGE_PROMPTS = {
    "judge-v1": (
        "You are a strict evaluator of answers produced by a question-answering system over "
        "machine-learning research papers. The system was given the numbered CONTEXT passages "
        "below and had to answer using only them.\n\n"
        "Do three things:\n"
        "1. CLAIMS: split the SYSTEM ANSWER into atomic factual claims (ignore citation markers "
        "like [1]). For each claim, set \"supported\" to true only if the claim can be directly "
        "inferred from the CONTEXT. Use only the CONTEXT: not your own knowledge, and not the "
        "reference answer.\n"
        "2. RELEVANCE: an integer 1-5 for how directly and completely the SYSTEM ANSWER addresses "
        "the QUESTION, regardless of whether it is correct (5 = fully addresses it, 1 = off-topic).\n"
        "3. CORRECTNESS: compare the SYSTEM ANSWER with the REFERENCE ANSWER. \"correct\" if it "
        "conveys the reference's key facts without contradicting it (extra correct detail is fine), "
        "\"partial\" if it gets some key facts but misses or garbles others, \"incorrect\" if it "
        "misses the key facts or contradicts the reference.\n\n"
        "Reply with JSON only, in this exact shape:\n"
        "{{\"claims\": [{{\"claim\": \"...\", \"supported\": true}}], \"relevance\": 5, "
        "\"correctness\": \"correct\", \"notes\": \"one short sentence\"}}\n\n"
        "QUESTION:\n{question}\n\n"
        "REFERENCE ANSWER:\n{reference}\n\n"
        "CONTEXT:\n{context}\n\n"
        "SYSTEM ANSWER:\n{answer}\n"
    ),
}

CORRECTNESS_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
RETRY_SUFFIX = "\n\nYour previous reply was not valid JSON. Reply with the JSON object only."


def _parse(data: dict) -> dict:
    claims = []
    for c in data.get("claims") or []:
        if isinstance(c, dict) and str(c.get("claim", "")).strip():
            sup = c.get("supported")
            sup = sup if isinstance(sup, bool) else str(sup).strip().lower() in {"true", "yes", "1"}
            claims.append({"claim": str(c["claim"]).strip(), "supported": sup})
    rel = int(round(float(data.get("relevance"))))
    rel = min(5, max(1, rel))
    corr = str(data.get("correctness", "")).strip().lower()
    if corr not in CORRECTNESS_SCORE:
        raise ValueError(f"bad correctness label {corr!r}")
    return {
        "claims": claims,
        "faithfulness": (sum(c["supported"] for c in claims) / len(claims)) if claims else None,
        "relevance": (rel - 1) / 4,          # 1..5 -> 0..1
        "relevance_raw": rel,
        "correctness": CORRECTNESS_SCORE[corr],
        "correctness_label": corr,
        "judge_notes": str(data.get("notes", ""))[:300],
    }


def judge_answer(cfg: dict, question: str, reference: str, context: str, answer: str) -> dict:
    prompt = JUDGE_PROMPTS[cfg["judge"]["prompt_version"]].format(
        question=question, reference=reference, context=context, answer=answer)
    last_error = ""
    for attempt_prompt in (prompt, prompt + RETRY_SUFFIX):
        text, _, _ = cached_chat(cfg, "judge", [("user", attempt_prompt)], what="judge")
        try:
            return {**_parse(extract_json(text)), "judge_error": None}
        except (ValueError, TypeError, KeyError) as e:
            last_error = f"{type(e).__name__}: {e}"[:200]
    return {"claims": [], "faithfulness": None, "relevance": None, "relevance_raw": None,
            "correctness": None, "correctness_label": None, "judge_notes": "", "judge_error": last_error}
