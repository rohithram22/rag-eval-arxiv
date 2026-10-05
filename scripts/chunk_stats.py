"""Step 3b: chunk the corpus at several sizes and report statistics.

Usage:
    python scripts/chunk_stats.py                 # sizes 128 256 512
    python scripts/chunk_stats.py --sizes 256     # one size
Writes data/processed/chunks_<size>.jsonl for inspection (gitignored).
"""
from __future__ import annotations

import argparse
import json
import sys
from statistics import median

from rag_eval.chunking import chunk_corpus, count_tokens, get_tokenizer, load_papers, node_to_record
from rag_eval.config import REPO_ROOT, load_config, resolve_path

EMBED_LIMIT = 510  # bge-small reads 512 tokens, minus [CLS] and [SEP]


def percentile(values: list[int], q: float) -> int:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    args = parser.parse_args()

    base = load_config()
    papers = load_papers(base)
    tokenizer = get_tokenizer(base["embedding"]["model_name"])
    out_dir = resolve_path(base, "processed_dir")

    header = f"{'size':>5} {'overlap':>7} {'chunks':>7} {'per_paper':>9} {'med_tok':>7} {'p95_tok':>7} " \
             f"{'max_tok':>7} {'over_limit':>10} {'overlap_ok':>10}"
    rows, example = [], None
    for size in args.sizes:
        cfg = load_config(overrides={"chunking.chunk_size": size, "chunking.chunk_overlap": size // 8})
        nodes = chunk_corpus(cfg, papers)
        records = [node_to_record(n) for n in nodes]

        embed_tokens = [count_tokens(tokenizer, r["embed_text"]) for r in records]
        over_limit = sum(t > EMBED_LIMIT for t in embed_tokens)

        # Overlap check: consecutive chunks of the same paper should share text.
        pairs = ok = 0
        for a, b in zip(records, records[1:]):
            if a["arxiv_id"] == b["arxiv_id"] and a["end_char"] is not None and b["start_char"] is not None:
                pairs += 1
                ok += b["start_char"] < a["end_char"]
        overlap_ok = ok / pairs if pairs else 0.0

        with open(out_dir / f"chunks_{size}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        rows.append(f"{size:>5} {size // 8:>7} {len(records):>7} {len(records) / len(papers):>9.1f} "
                    f"{median(embed_tokens):>7.0f} {percentile(embed_tokens, 0.95):>7} {max(embed_tokens):>7} "
                    f"{over_limit:>10} {overlap_ok:>10.0%}")
        if example is None or size == 256:
            mid = [r for r in records if r["arxiv_id"].startswith("2004.04906")] or records
            example = (size, mid[len(mid) // 3])

    print("\nToken counts below are for the EMBEDDED text (title header + chunk), bge tokenizer.")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(r)
    print(f"\nover_limit should be 0 (no chunk above {EMBED_LIMIT} tokens -> nothing truncated by bge-small).")
    print("overlap_ok should be close to 100% (consecutive chunks share text).")

    size, rec = example
    print(f"\n--- example chunk (size {size}): {rec['id']} ---")
    print(rec["embed_text"][:1200] + (" ..." if len(rec["embed_text"]) > 1200 else ""))
    print(f"\nWrote data/processed/chunks_<size>.jsonl for: {', '.join(map(str, args.sizes))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
