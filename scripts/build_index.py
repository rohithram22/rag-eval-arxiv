"""Step 4a: build (or load) the FAISS index for a chunk size and sanity-check it.

Usage:
    python scripts/build_index.py                     # chunk_size from base.yaml (512)
    python scripts/build_index.py --chunk-size 256    # overlap = size // 8
    python scripts/build_index.py --rebuild
"""
from __future__ import annotations

import argparse
import random
import sys
import time

from rag_eval.config import load_config, resolve_path
from rag_eval.indexing import build_or_load_index, embed_texts
from rag_eval.chunking import node_to_record


def dir_size_mb(path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    overrides = {}
    if args.chunk_size:
        overrides = {"chunking.chunk_size": args.chunk_size, "chunking.chunk_overlap": args.chunk_size // 8}
    cfg = load_config(overrides=overrides)

    index, nodes, meta = build_or_load_index(cfg, rebuild=args.rebuild)
    folder = resolve_path(cfg, "index_dir") / meta["key"]
    print(f"index folder: {folder}  ({dir_size_mb(folder):.1f} MB on disk)")

    # Alignment check: re-embed 20 random chunks and search with their own vectors.
    # Each must come back as its own top-1 hit; otherwise vectors and chunks are misaligned.
    rng = random.Random(cfg["project"]["seed"])
    sample = rng.sample(range(len(nodes)), k=min(20, len(nodes)))
    texts = [node_to_record(nodes[i])["embed_text"] for i in sample]
    t0 = time.time()
    vecs = embed_texts(cfg, texts, show_progress=False)
    scores, ids = index.search(vecs, 1)
    search_ms = (time.time() - t0) * 1000 / len(sample)
    hits = sum(int(ids[j][0]) == i for j, i in enumerate(sample))
    print(f"alignment check: {hits}/{len(sample)} chunks retrieved themselves at rank 1 "
          f"(top score avg {scores[:, 0].mean():.3f}; ~{search_ms:.0f} ms per embed+search)")
    if hits < len(sample) - 1:  # allow one exact-duplicate chunk
        print("WARNING: vectors and chunks look misaligned. Rebuild with --rebuild.")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
