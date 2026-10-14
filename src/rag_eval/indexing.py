"""Embed chunks and build / cache an exact FAISS index.

Cache key = hash of everything the vectors depend on: parsing + chunking +
embedding config, plus a fingerprint of papers.jsonl (so re-parsing with new
code also invalidates the cache). Different retrieval/reranker/LLM settings
reuse the same index.

Layout: indexes/<key>/
  index.faiss   exact inner-product index (vectors are L2-normalized -> cosine)
  nodes.jsonl   chunk records, line i <-> vector i
  meta.json     what was built, from which config, how long it took
"""
from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from llama_index.core.schema import TextNode

from rag_eval.chunking import chunk_corpus, node_to_record, record_to_node
from rag_eval.config import config_hash, resolve_path


@lru_cache(maxsize=2)
def _embed_model_cached(model_name: str, max_length: int, query_instruction: str, batch_size: int):
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=model_name,
        max_length=max_length,
        query_instruction=query_instruction,
        normalize=True,
        embed_batch_size=batch_size,
        device="cpu",
    )


def get_embed_model(cfg: dict):
    e = cfg["embedding"]
    return _embed_model_cached(e["model_name"], e["max_length"], e["query_instruction"], e["batch_size"])


def _file_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]


def index_key(cfg: dict) -> str:
    papers = resolve_path(cfg, "processed_dir") / "papers.jsonl"
    fingerprint = _file_fingerprint(papers)
    sections = {s: cfg[s] for s in ("parsing", "chunking", "embedding")}
    return config_hash({"sections": sections, "papers": fingerprint}, length=12)


def embed_texts(cfg: dict, texts: list[str], show_progress: bool = True) -> np.ndarray:
    vecs = get_embed_model(cfg).get_text_embedding_batch(texts, show_progress=show_progress)
    arr = np.asarray(vecs, dtype="float32")
    faiss.normalize_L2(arr)  # already normalized by the model; harmless and guarantees cosine
    return arr


def embed_query(cfg: dict, question: str) -> np.ndarray:
    vec = np.asarray([get_embed_model(cfg).get_query_embedding(question)], dtype="float32")
    faiss.normalize_L2(vec)
    return vec


def build_or_load_index(cfg: dict, rebuild: bool = False, verbose: bool = True
                        ) -> tuple[faiss.Index, list[TextNode], dict]:
    key = index_key(cfg)
    out = resolve_path(cfg, "index_dir") / key
    paths = {n: out / n for n in ("index.faiss", "nodes.jsonl", "meta.json")}
    include_title = cfg["chunking"]["include_title"]

    if not rebuild and all(p.exists() for p in paths.values()):
        index = faiss.read_index(str(paths["index.faiss"]))
        with open(paths["nodes.jsonl"], encoding="utf-8") as f:
            nodes = [record_to_node(json.loads(line), include_title) for line in f]
        meta = json.loads(paths["meta.json"].read_text())
        if index.ntotal != len(nodes):
            raise RuntimeError(f"Corrupt index cache {out}: {index.ntotal} vectors vs {len(nodes)} nodes. "
                               f"Delete the folder or pass rebuild=True.")
        if verbose:
            print(f"Loaded cached index {key}: {index.ntotal} vectors (built in {meta['build_seconds']:.0f}s)")
        return index, nodes, meta

    if verbose:
        c = cfg["chunking"]
        print(f"Building index {key} (chunk_size={c['chunk_size']}, overlap={c['chunk_overlap']})...")
    t0 = time.time()
    nodes = chunk_corpus(cfg)
    records = [node_to_record(n) for n in nodes]
    vectors = embed_texts(cfg, [r["embed_text"] for r in records], show_progress=verbose)

    index = faiss.IndexFlatIP(vectors.shape[1])  # exact search: no approximation noise in experiments
    index.add(vectors)

    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(paths["index.faiss"]))
    with open(paths["nodes.jsonl"], "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "key": key,
        "n_vectors": int(index.ntotal),
        "dim": int(vectors.shape[1]),
        "build_seconds": round(time.time() - t0, 1),
        "config": {s: cfg[s] for s in ("parsing", "chunking", "embedding")},
    }
    paths["meta.json"].write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"Built index {key}: {meta['n_vectors']} vectors x {meta['dim']} dims in {meta['build_seconds']:.0f}s")
    return index, nodes, meta
