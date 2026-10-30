"""Inspect one evaluation run: per-type breakdown and the failure cases.

Usage:
    python scripts/inspect_run.py c512-dense                    # breakdown + all failure kinds
    python scripts/inspect_run.py c512-dense --what misses      # retrieval misses only
    python scripts/inspect_run.py c512-dense --what unfaithful --n 5
Failure kinds: misses | unfaithful | incorrect | unanswerable | all
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import defaultdict
from statistics import mean

from rag_eval.config import load_config, resolve_path
from rag_eval.golden import read_jsonl

W = 100


def wrap(s: str, indent: str = "      ") -> str:
    return textwrap.indent(textwrap.fill(" ".join(str(s).split()), W), indent)


def avg(vals):
    vals = [v for v in vals if v is not None]
    return f"{mean(vals):.2f}" if vals else "  - "


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--what", default="all", choices=["misses", "unfaithful", "incorrect", "unanswerable", "all"])
    parser.add_argument("--n", type=int, default=8, help="max examples per failure kind")
    args = parser.parse_args()

    cfg = load_config()
    run_dir = resolve_path(cfg, "results_dir") / args.label
    rows = read_jsonl(run_dir / "per_question.jsonl")
    if not rows:
        print(f"No results in {run_dir}")
        return 1
    summary = json.loads((run_dir / "summary.json").read_text())
    golden = {g["id"]: g for g in read_jsonl(resolve_path(cfg, "eval_dir") / summary["config"]["eval"]["golden_file"])}
    titles = {g["base_id"]: g["title"] for g in golden.values() if g.get("base_id")}
    full = summary["mode"] == "full"

    # ---- per question type
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    print(f"\n=== {args.label} ({summary['mode']}) : breakdown by question type ===")
    print(f"{'type':<13}{'n':>3}  {'hit@5':>6}  {'mrr':>5}" + (f"  {'correct':>7}  {'faithful':>8}  {'abstain':>7}" if full else ""))
    for t, rs in sorted(by_type.items()):
        line = f"{t:<13}{len(rs):>3}  {avg([r.get('hit@5') for r in rs]):>6}  {avg([r.get('mrr') for r in rs]):>5}"
        if full:
            line += (f"  {avg([r.get('correctness') for r in rs]):>7}  {avg([r.get('faithfulness') for r in rs]):>8}"
                     f"  {avg([float(r['abstained']) for r in rs if 'abstained' in r]):>7}")
        print(line)

    def section(name, items, show):
        if args.what not in (name, "all"):
            return
        print(f"\n--- {name}: {len(items)} ---")
        for r in items[:args.n]:
            show(r)
        if len(items) > args.n:
            print(f"   ... {len(items) - args.n} more (use --n)")

    def show_miss(r):
        g = golden[r["id"]]
        papers = [x["id"].split("#")[0] for x in r["retrieved"]]
        print(f"\n[{r['id']}] {r['question']}")
        print(f"   expected: {g['base_id']}  {g['title'][:70]}")
        print(f"   right paper retrieved: {'yes, but not the evidence passage' if r['paper_hit'] else 'NO'}")
        print("   got: " + ", ".join(f"{x['id']}({x['score']:.2f})" for x in r["retrieved"]))
        for pid in dict.fromkeys(papers):
            if pid != g["base_id"]:
                print(f"        {pid}: {titles.get(pid, '(paper without a golden question)')[:70]}")

    def show_unfaithful(r):
        print(f"\n[{r['id']}] faithfulness={r['faithfulness']:.2f}  {r['question']}")
        print(wrap(f"ANSWER: {r['answer']}"))
        for c in r["claims"]:
            if not c["supported"]:
                print(wrap(f"UNSUPPORTED: {c['claim']}"))

    def show_incorrect(r):
        g = golden[r["id"]]
        print(f"\n[{r['id']}] {r.get('correctness_label') or 'abstained'}  (retrieval rank: {r.get('first_relevant_rank') or 'miss'})")
        print(wrap(f"Q: {r['question']}"))
        print(wrap(f"REFERENCE: {g['answer']}"))
        print(wrap(f"ANSWER: {r['answer']}"))
        if r.get("judge_notes"):
            print(wrap(f"JUDGE: {r['judge_notes']}"))

    def show_unans(r):
        print(f"\n[{r['id']}] {r['question']}")
        print(wrap(f"ANSWER (should have abstained): {r['answer']}"))

    ans = [r for r in rows if r["answerable"]]
    section("misses", [r for r in ans if r.get("first_relevant_rank") is None], show_miss)
    if full:
        section("unfaithful", [r for r in ans if r.get("faithfulness") is not None and r["faithfulness"] < 1], show_unfaithful)
        section("incorrect", [r for r in ans if r.get("correctness") is not None and r["correctness"] < 1], show_incorrect)
        section("unanswerable", [r for r in rows if not r["answerable"] and not r.get("abstained")], show_unans)
    return 0


if __name__ == "__main__":
    sys.exit(main())
