"""Step 3a: parse every PDF in the manifest into data/processed/papers.jsonl.

Usage:
    python scripts/parse_corpus.py
"""
from __future__ import annotations

import json
import random
import sys
from statistics import mean, median

from tqdm import tqdm

from rag_eval.config import REPO_ROOT, load_config, resolve_path
from rag_eval.parsing import parse_pdf

MANIFEST = REPO_ROOT / "data" / "manifest.jsonl"
LOW_TEXT_CHARS = 8000  # a 6+ page paper with less clean text than this probably parsed badly


def main() -> int:
    cfg = load_config()
    raw_dir = resolve_path(cfg, "raw_dir")
    out_dir = resolve_path(cfg, "processed_dir")
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    with open(MANIFEST, encoding="utf-8") as f:
        manifest = [json.loads(line) for line in f if line.strip()]

    p_cfg = cfg["parsing"]
    papers, failures = [], []
    for rec in tqdm(manifest, desc="parsing", unit="paper"):
        pdf = raw_dir / f"{rec['arxiv_id'].replace('/', '_')}.pdf"
        try:
            parsed = parse_pdf(pdf, margin_frac=p_cfg["header_footer_margin"],
                               drop_references=p_cfg["drop_references"])
        except Exception as e:  # noqa: BLE001
            failures.append(f"{rec['arxiv_id']}: {type(e).__name__}: {e}")
            continue
        papers.append({
            "arxiv_id": rec["arxiv_id"],
            "base_id": rec["base_id"],
            "title": rec["title"],
            "published": rec["published"],
            "seed": rec["seed"],
            "n_pages": parsed["n_pages"],
            "raw_chars": parsed["raw_chars"],
            "n_chars": len(parsed["text"]),
            "refs_removed": parsed["refs_removed"],
            "text": parsed["text"],
        })

    out_path = out_dir / "papers.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Samples to read by eye: one seed paper + one random paper.
    rng = random.Random(cfg["project"]["seed"])
    seeds = [p for p in papers if p["seed"]]
    others = [p for p in papers if not p["seed"]]
    samples = ([seeds[0]] if seeds else []) + ([rng.choice(others)] if others else [])
    for p in samples:
        (samples_dir / f"{p['base_id']}.txt").write_text(
            f"{p['title']}\n{'=' * len(p['title'])}\n\n{p['text']}", encoding="utf-8")

    # ---- stats
    n = len(papers)
    chars = [p["n_chars"] for p in papers]
    kept_ratio = sum(chars) / max(1, sum(p["raw_chars"] for p in papers))
    no_refs = [p for p in papers if not p["refs_removed"]]
    low_text = [p for p in papers if p["n_chars"] < LOW_TEXT_CHARS and p["n_pages"] >= 6]

    print("\n=========== PARSING SUMMARY ===========")
    print(f"parsed:            {n}/{len(manifest)}  (failures: {len(failures)})")
    print(f"clean chars:       total={sum(chars):,}  mean={mean(chars):,.0f}  median={median(chars):,.0f}")
    print(f"kept vs raw text:  {kept_ratio:.0%}  (the rest = references, headers, page numbers, noise)")
    print(f"references cut:    {n - len(no_refs)}/{n} papers ({(n - len(no_refs)) / max(1, n):.0%})")
    if no_refs:
        print("  no References heading found in: " + ", ".join(p["arxiv_id"] for p in no_refs[:10])
              + (" ..." if len(no_refs) > 10 else ""))
    print(f"low-text papers:   {len(low_text)}" + (
        "  -> " + ", ".join(f"{p['arxiv_id']} ({p['n_chars']} chars)" for p in low_text) if low_text else ""))
    for f_ in failures:
        print(f"  FAILED {f_}")
    print(f"wrote:             {out_path.relative_to(REPO_ROOT)}")
    print(f"samples to read:   " + ", ".join(
        str((samples_dir / f"{p['base_id']}.txt").relative_to(REPO_ROOT)) for p in samples))
    print("=======================================")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
