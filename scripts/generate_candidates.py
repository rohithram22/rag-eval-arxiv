"""Step 5a: draft golden-set candidates with the question-generation LLM.

For up to ~80 papers (all seed papers first, then a seeded random order of the
rest) it picks one prose passage, asks the LLM for a question / answer /
verbatim evidence quote, then checks that the evidence really is in the paper.

Output: eval/golden_candidates.jsonl (committed: it's the provenance of the golden set)
Resumable: papers already in the file are skipped. Stops cleanly at a daily token limit.

Usage:
    python scripts/generate_candidates.py --limit 3     # quick trial: 3 LLM calls
    python scripts/generate_candidates.py               # full run (~25-30 min)
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter

from llama_index.core.llms import ChatMessage, MessageRole

from rag_eval.chunking import load_papers
from rag_eval.config import load_config, resolve_path
from rag_eval.golden import (QG_PROMPTS, QTYPE_DESC, UNANSWERABLE_CANDIDATES, append_jsonl,
                             find_evidence, quality_flags, read_jsonl, sample_passage)
from rag_eval.llm_utils import DailyLimitError, call_with_backoff, extract_json
from rag_eval.llms import get_llm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="max LLM calls this run (for a trial)")
    args = parser.parse_args()

    cfg = load_config()
    g = cfg["golden"]
    out_path = resolve_path(cfg, "eval_dir") / "golden_candidates.jsonl"
    existing = read_jsonl(out_path)

    # Hand-written unanswerable candidates are added once.
    if not any(r["type"] == "unanswerable" for r in existing):
        for i, q in enumerate(UNANSWERABLE_CANDIDATES):
            row = {"id": f"u{i:02d}", "type": "unanswerable", "answerable": False, "question": q,
                   "answer": "I don't know based on the provided papers.", "status": "pending",
                   "source": "manual", "flags": []}
            append_jsonl(out_path, row)
            existing.append(row)

    llm_rows = [r for r in existing if r.get("source") == "llm"]
    done_papers = {r["base_id"] for r in llm_rows}
    n_valid = sum(r["status"] != "auto_rejected" for r in llm_rows)
    n_calls_before = len(llm_rows)

    papers = load_papers(cfg)
    rng = random.Random(cfg["project"]["seed"] + n_calls_before)  # new passages on resume
    seeds = [p for p in papers if p["seed"]]
    others = [p for p in papers if not p["seed"]]
    random.Random(cfg["project"]["seed"]).shuffle(others)       # same paper order every run
    order = [p for p in seeds + others if p["base_id"] not in done_papers]

    llm = get_llm(cfg["qgen"])
    prompt = QG_PROMPTS[g["prompt_version"]]
    qtypes = g["question_types"]
    budget = min(g["max_calls"] - n_calls_before, args.limit or 10**9)
    print(f"{n_valid} valid candidates so far; target {g['n_candidates']}; "
          f"up to {budget} calls this run ({g['min_interval_s']}s apart)")

    stats: Counter = Counter()
    calls = 0
    for p in order:
        if n_valid >= g["n_candidates"] or calls >= budget:
            break
        span = sample_passage(p["text"], rng, g["passage_min_chars"], g["passage_max_chars"])
        if span is None:
            stats["no_good_passage"] += 1
            continue
        ps, pe = span
        passage = p["text"][ps:pe]
        qtype = qtypes[(n_calls_before + calls) % len(qtypes)]
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=prompt["system"]),
            ChatMessage(role=MessageRole.USER, content=prompt["user"].format(
                title=p["title"], passage=passage, qtype=qtype, qtype_desc=QTYPE_DESC[qtype])),
        ]
        cid = f"c{n_calls_before + calls:03d}"
        try:
            reply = call_with_backoff(lambda: llm.chat(messages), f"{cid} {p['base_id']}")
        except DailyLimitError as e:
            print(f"\nSTOPPED: {e}")
            break
        calls += 1

        row = {"id": cid, "type": qtype, "answerable": True, "base_id": p["base_id"],
               "arxiv_id": p["arxiv_id"], "title": p["title"], "passage_start": ps, "passage_end": pe,
               "source": "llm", "generator": cfg["qgen"]["model"], "prompt_version": g["prompt_version"]}
        try:
            data = extract_json(reply.message.content or "")
        except ValueError as e:
            row.update(status="auto_rejected", reject_reason=f"bad_json: {e}")
            stats["bad_json"] += 1
        else:
            if data.get("skip"):
                row.update(status="auto_rejected", reject_reason=f"llm_skip: {data.get('reason', '')}")
                stats["llm_skipped_passage"] += 1
            else:
                row.update(question=str(data.get("question", "")).strip(),
                           answer=str(data.get("answer", "")).strip(),
                           evidence=str(data.get("evidence", "")).strip())
                found = find_evidence(passage, row["evidence"])
                if not found:
                    row.update(status="auto_rejected", reject_reason="evidence_not_in_passage")
                    stats["evidence_not_found"] += 1
                else:
                    s, e, how = found
                    row.update(evidence=p["text"][ps + s:ps + e],  # store the VERBATIM paper text
                               evidence_start=ps + s, evidence_end=ps + e, evidence_match=how,
                               status="pending")
                    row["flags"] = quality_flags(row)
                    n_valid += 1
                    stats["valid"] += 1
        append_jsonl(out_path, row)
        mark = "+" if row["status"] == "pending" else "-"
        print(f"  [{mark}] {cid} {qtype:<10} {p['base_id']:<12} "
              f"{row.get('question', row.get('reject_reason', ''))[:90]}")
        time.sleep(g["min_interval_s"])

    print(f"\nThis run: {calls} calls -> " + ", ".join(f"{k}={v}" for k, v in stats.most_common()))
    print(f"Valid candidates now: {n_valid} (target {g['n_candidates']}). File: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
