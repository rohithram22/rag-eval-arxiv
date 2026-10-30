"""Spot-check the LLM judge against your own judgment.

Shows judged answers one at a time (seeded random sample). For each, you say
whether you agree with (1) the correctness verdict and (2) the claim-by-claim
faithfulness verdicts. Results go to eval/judge_audit.jsonl (committed), and the
agreement rate is what you report for the judge's reliability.

Usage:
    python scripts/audit_judge.py c512-dense            # audit 10 answers (resumes where you stopped)
    python scripts/audit_judge.py c512-dense --n 15
    python scripts/audit_judge.py c512-dense --report   # agreement summary only
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap

from rag_eval.config import load_config, resolve_path
from rag_eval.golden import append_jsonl, read_jsonl
from rag_eval.indexing import build_or_load_index

W = 100


def wrap(s: str, indent: str = "    ") -> str:
    return textwrap.indent(textwrap.fill(" ".join(str(s).split()), W), indent)


def ask(prompt: str) -> str | None:
    while True:
        a = input(prompt).strip().lower()
        if a in {"y", "n", "s", "q"}:
            return a


def report(audit: list[dict], label: str) -> None:
    mine = [a for a in audit if a["label"] == label]
    print(f"\n=== judge audit for {label}: {len(mine)} answers ===")
    for key in ("agree_correctness", "agree_claims"):
        vals = [a[key] for a in mine if a[key] is not None]
        if vals:
            print(f"  {key:<18} {sum(vals)}/{len(vals)} = {sum(vals) / len(vals):.0%}")
    notes = [a for a in mine if a.get("note")]
    for a in notes:
        print(f"  note [{a['id']}]: {a['note']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    base = load_config()
    run_dir = resolve_path(base, "results_dir") / args.label
    audit_path = resolve_path(base, "eval_dir") / "judge_audit.jsonl"
    audit = read_jsonl(audit_path)
    if args.report:
        report(audit, args.label)
        return 0

    summary = json.loads((run_dir / "summary.json").read_text())
    if summary["mode"] != "full":
        print("This run has no judged answers (retrieval-only).")
        return 1
    cfg = summary["config"]
    rows = read_jsonl(run_dir / "per_question.jsonl")
    golden = {g["id"]: g for g in read_jsonl(resolve_path(cfg, "eval_dir") / cfg["eval"]["golden_file"])}
    done = {a["id"] for a in audit if a["label"] == args.label}
    judged = [r for r in rows if r["answerable"] and not r.get("abstained") and not r.get("judge_error")]
    sample = random.Random(cfg["project"]["seed"]).sample(judged, k=min(args.n, len(judged)))
    todo = [r for r in sample if r["id"] not in done]
    if not todo:
        print("All sampled answers already audited.")
        report(audit, args.label)
        return 0

    _, nodes, _ = build_or_load_index(cfg, verbose=False)
    text_of = {n.node_id: n.text for n in nodes}
    print("Judge the JUDGE: read the context yourself, then say whether its verdicts are right.")
    for i, r in enumerate(todo, start=1):
        g = golden[r["id"]]
        print("\n" + "=" * W)
        print(f"[{i}/{len(todo)}] {r['id']}\nQ: {r['question']}")
        print("REFERENCE:"); print(wrap(g["answer"]))
        print("CONTEXT given to the system (each chunk truncated):")
        for x in r["retrieved"]:
            print(wrap(f"[{x['rank']}] {text_of.get(x['id'], '')[:700]} ...", "    "))
        print("SYSTEM ANSWER:"); print(wrap(r["answer"]))
        print(f"JUDGE  correctness: {r['correctness_label']}   relevance: {r['relevance_raw']}/5   "
              f"faithfulness: {r['faithfulness'] if r['faithfulness'] is None else round(r['faithfulness'], 2)}")
        for c in r["claims"]:
            print(wrap(f"{'SUPPORTED  ' if c['supported'] else 'UNSUPPORTED'}  {c['claim']}"))
        a1 = ask("\nAgree with the CORRECTNESS verdict? [y/n/s=skip/q=quit] > ")
        if a1 == "q":
            break
        a2 = ask("Agree with ALL the claim verdicts? [y/n/s/q] > ")
        if a2 == "q":
            break
        note = input("Optional note (Enter to skip): ").strip()
        append_jsonl(audit_path, {
            "label": args.label, "id": r["id"], "judge_model": cfg["judge"]["model"],
            "judge_prompt": cfg["judge"]["prompt_version"],
            "agree_correctness": None if a1 == "s" else a1 == "y",
            "agree_claims": None if a2 == "s" else a2 == "y", "note": note,
        })
    report(read_jsonl(audit_path), args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
