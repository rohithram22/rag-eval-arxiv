"""Step 5b: review golden-set candidates, add your own questions, export the final set.

Usage:
    python scripts/review_golden.py review     # walk through pending candidates
    python scripts/review_golden.py add        # add a hand-written question (with evidence)
    python scripts/review_golden.py stats      # counts by status / type / flags
    python scripts/review_golden.py export     # accepted -> eval/golden.jsonl

Review rule: judge the QUESTION, never whether the system answers it correctly.
Rejecting questions the system fails would bias the evaluation in its favour.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from collections import Counter
from statistics import mean

from rag_eval.chunking import load_papers
from rag_eval.config import load_config, resolve_path
from rag_eval.golden import find_evidence, lexical_overlap, quality_flags, read_jsonl, write_jsonl

W = 100


def wrap(s: str, indent: str = "    ") -> str:
    return textwrap.indent(textwrap.fill(" ".join(s.split()), W), indent)


def paths(cfg):
    d = resolve_path(cfg, "eval_dir")
    return d / "golden_candidates.jsonl", d / "golden.jsonl"


def show(item: dict, papers: dict, i: int, n: int) -> None:
    print("\n" + "=" * W)
    head = f"[{i}/{n}] {item['id']}  type={item['type']}  "
    head += f"paper={item['arxiv_id']}  {item['title'][:60]}" if item["answerable"] else "UNANSWERABLE"
    print(head)
    if item.get("flags"):
        print(f"FLAGS: {', '.join(item['flags'])}")
    print(f"\nQ: {item['question']}")
    if item["answerable"]:
        print(f"A: {item['answer']}")
        print("EVIDENCE:")
        print(wrap(item["evidence"]))
        text = papers[item["base_id"]]["text"]
        es, ee = item["evidence_start"], item["evidence_end"]
        ps, pe = item.get("passage_start", es), item.get("passage_end", ee)
        ctx = text[ps:es] + " >>>" + text[es:ee] + "<<< " + text[ee:pe]
        print("PASSAGE (evidence between >>> <<<):")
        print(wrap(ctx[:2600] + (" ..." if len(ctx) > 2600 else "")))
    else:
        print("Use [p]review to check that NO retrieved chunk answers this, then accept or reject.")


def edit(item: dict, papers: dict) -> None:
    for field in ("question", "answer"):
        new = input(f"New {field} (Enter = keep): ").strip()
        if new:
            item.setdefault("original", {})[field] = item[field]
            item[field] = new
            item["edited"] = True
    if item["answerable"]:
        new_ev = input("New evidence quote, copied from the paper (Enter = keep): ").strip()
        if new_ev:
            text = papers[item["base_id"]]["text"]
            found = find_evidence(text, new_ev)
            if not found:
                print("  Evidence not found in the paper text; keeping the old evidence.")
            else:
                s, e, how = found
                item.setdefault("original", {})["evidence"] = item["evidence"]
                item.update(evidence=text[s:e], evidence_start=s, evidence_end=e,
                            evidence_match=how, edited=True)
    item["flags"] = quality_flags(item)


def preview(cfg: dict, question: str) -> None:
    from rag_eval.pipeline import RAGPipeline  # lazy: loads the embedding model only when used

    rag = RAGPipeline(cfg, verbose=False)
    for rank, r in enumerate(rag.retrieve(question), start=1):
        print(f"  [{rank}] {r.score:.3f} {r.node.node_id:<16} {r.node.metadata['title'][:60]}")
        print(wrap(r.node.text[:300] + " ...", "        "))


def cmd_review(cfg) -> None:
    cand_path, _ = paths(cfg)
    items = read_jsonl(cand_path)
    papers = {p["base_id"]: p for p in load_papers(cfg)}
    pending = [it for it in items if it["status"] == "pending"]
    print(f"{len(pending)} pending. Accept if: self-contained, unambiguous across 150 papers, answer "
          f"fully supported by the evidence, realistic, not general knowledge.")
    for i, item in enumerate(pending, start=1):
        show(item, papers, i, len(pending))
        while True:
            choice = input("\n[a]ccept [e]dit [r]eject [s]kip [p]review(unanswerable) [q]uit > ").strip().lower()
            if choice == "a":
                item["status"] = "accepted"
            elif choice == "e":
                edit(item, papers)
                show(item, papers, i, len(pending))
                continue
            elif choice == "r":
                item["status"] = "rejected"
                item["reject_reason"] = input("Reason (short): ").strip() or "reviewer"
            elif choice == "p":
                if item["answerable"]:
                    print("  Preview is for unanswerable items only: judging answerable questions by "
                          "how the system does would bias the evaluation.")
                else:
                    preview(cfg, item["question"])
                continue
            elif choice == "q":
                write_jsonl(cand_path, items)
                print("Saved. Run `review` again to continue where you left off.")
                return
            elif choice != "s":
                continue
            write_jsonl(cand_path, items)  # save after every decision
            break
    print("\nNo more pending candidates.")
    cmd_stats(cfg)


def cmd_add(cfg) -> None:
    cand_path, _ = paths(cfg)
    items = read_jsonl(cand_path)
    papers = load_papers(cfg)
    by_id = {p["base_id"]: p for p in papers} | {p["arxiv_id"]: p for p in papers}
    answerable = input("Answerable from the corpus? [Y/n] ").strip().lower() != "n"
    question = input("Question: ").strip()
    n_manual = sum(1 for it in items if it["id"].startswith("m"))
    row = {"id": f"m{n_manual:02d}", "question": question, "source": "manual",
           "status": "accepted", "answerable": answerable}
    if not answerable:
        row.update(type="unanswerable", answer="I don't know based on the provided papers.", flags=[])
    else:
        p = by_id.get(input("arXiv id of the paper (e.g. 2309.15217): ").strip())
        if not p:
            print("Unknown arXiv id (it must be in the corpus).")
            return
        row.update(type=input(f"Type {cfg['golden']['question_types']}: ").strip() or "factual",
                   answer=input("Reference answer: ").strip(), base_id=p["base_id"],
                   arxiv_id=p["arxiv_id"], title=p["title"])
        ev = input("Evidence quote (copy 1-3 sentences from the paper): ").strip()
        found = find_evidence(p["text"], ev)
        if not found:
            print("Evidence not found in the parsed paper text. Copy it from the parsed text "
                  "(see the command in the Step 5 notes), then retry.")
            return
        s, e, how = found
        row.update(evidence=p["text"][s:e], evidence_start=s, evidence_end=e, evidence_match=how)
        row["flags"] = quality_flags(row)
    items.append(row)
    write_jsonl(cand_path, items)
    print(f"Added {row['id']} (accepted). Flags: {row.get('flags') or 'none'}")


def cmd_stats(cfg) -> None:
    cand_path, _ = paths(cfg)
    items = read_jsonl(cand_path)
    acc = [it for it in items if it["status"] == "accepted"]
    print("\n=========== GOLDEN SET STATS ===========")
    print("status:      " + ", ".join(f"{k}={v}" for k, v in Counter(it["status"] for it in items).most_common()))
    reasons = Counter(it.get("reject_reason", "").split(":")[0] for it in items if it["status"] == "auto_rejected")
    if reasons:
        print("auto-reject: " + ", ".join(f"{k}={v}" for k, v in reasons.most_common()))
    if acc:
        ans = [it for it in acc if it["answerable"]]
        print(f"accepted:    {len(acc)} ({len(ans)} answerable, {len(acc) - len(ans)} unanswerable)")
        print("types:       " + ", ".join(f"{k}={v}" for k, v in Counter(it["type"] for it in acc).most_common()))
        print(f"papers:      {len({it['base_id'] for it in ans})} distinct")
        print("sources:     " + ", ".join(f"{k}={v}" for k, v in Counter(it["source"] for it in acc).most_common()))
        if ans:
            ov = mean(lexical_overlap(it["question"], it["evidence"]) for it in ans)
            print(f"overlap:     mean question/evidence word overlap = {ov:.2f}")
        flagged = [it["id"] for it in acc if it.get("flags")]
        print(f"flagged:     {len(flagged)} accepted items still flagged {flagged[:8]}")
    print("========================================")


def cmd_export(cfg) -> None:
    cand_path, out_path = paths(cfg)
    keep = ("id", "question", "answer", "type", "answerable", "base_id", "arxiv_id", "title",
            "evidence", "evidence_start", "evidence_end", "source", "edited")
    acc = [{k: it[k] for k in keep if k in it} for it in read_jsonl(cand_path) if it["status"] == "accepted"]
    write_jsonl(out_path, acc)
    print(f"Exported {len(acc)} accepted items -> {out_path}")
    cmd_stats(cfg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["review", "add", "stats", "export"])
    args = parser.parse_args()
    cfg = load_config()
    {"review": cmd_review, "add": cmd_add, "stats": cmd_stats, "export": cmd_export}[args.command](cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
