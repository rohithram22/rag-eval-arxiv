"""Step 4b: ask the RAG system a question from the terminal.

Usage:
    python scripts/ask.py "How does DPR train its encoders?"
    python scripts/ask.py "..." --k 3 --chunk-size 256
    python scripts/ask.py "..." --retrieve-only         # no LLM call
"""
from __future__ import annotations

import argparse
import sys
import textwrap

from rag_eval.config import load_config
from rag_eval.pipeline import RAGPipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k", type=int, help="number of chunks to retrieve")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--retrieve-only", action="store_true")
    args = parser.parse_args()

    overrides = {}
    if args.k:
        overrides["retrieval.top_k"] = args.k
    if args.chunk_size:
        overrides.update({"chunking.chunk_size": args.chunk_size, "chunking.chunk_overlap": args.chunk_size // 8})
    cfg = load_config(overrides=overrides)

    rag = RAGPipeline(cfg)
    results = rag.retrieve(args.question)

    print(f"\nQ: {args.question}\n")
    print("Retrieved chunks:")
    for rank, r in enumerate(results, start=1):
        m = r.node.metadata
        snippet = " ".join(r.node.text.split())[:220]
        print(f"  [{rank}] score={r.score:.3f}  {r.node.node_id}  {m['title'][:70]}")
        print(textwrap.indent(textwrap.fill(snippet + " ...", 100), "        "))

    if args.retrieve_only:
        return 0

    out = rag.answer(args.question, results=results)
    print("\nAnswer:")
    print(textwrap.indent(textwrap.fill(out["answer"], 100), "  "))
    lat = out["latency_s"]
    print(f"\n(retrieve {lat['retrieve']:.2f}s, generate {lat['generate']:.2f}s, "
          f"model {cfg['generator']['model']}, prompt {out['prompt_version']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
