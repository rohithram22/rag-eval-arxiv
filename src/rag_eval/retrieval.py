"""Retrievers: dense (FAISS), sparse (BM25), hybrid (reciprocal rank fusion),
optionally followed by a cross-encoder reranker.

Every configuration works in two stages:
  1. CANDIDATES: fetch retrieval.candidate_k chunks (dense, BM25, or both fused).
  2. FINAL: keep the first retrieval.top_k candidates, or, with the reranker
     enabled, let the cross-encoder re-order all candidates and keep the best top_k.

The candidate list is kept on the retriever (`last_candidates`), so the evaluation
can report the reranker's ceiling: a reranker can only promote evidence that is
already among the candidates.
"""
from __future__ import annotations

from functools import lru_cache

import faiss
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle, TextNode

from rag_eval.indexing import embed_query

MODES = ("dense", "bm25", "hybrid")


def _embed_text(node: TextNode) -> str:
    """The same text the dense index embedded: title header + chunk text."""
    return node.get_content(metadata_mode=MetadataMode.EMBED)


# ----------------------------------------------------------------- dense
class DenseSearcher:
    def __init__(self, cfg: dict, index: faiss.Index):
        self.cfg, self.index = cfg, index

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores, ids = self.index.search(embed_query(self.cfg, query), k)
        return [(int(i), float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]


# ----------------------------------------------------------------- sparse
class BM25Searcher:
    """Okapi BM25 (bm25s, Lucene variant, k1=1.5, b=0.75) over the same title+chunk
    text the dense index uses. English stopwords removed, Snowball stemming."""

    def __init__(self, nodes: list[TextNode], stemming: bool = True):
        import bm25s

        self._bm25s = bm25s
        self._stemmer = None
        if stemming:
            import Stemmer

            self._stemmer = Stemmer.Stemmer("english")
        tokens = bm25s.tokenize([_embed_text(n) for n in nodes], stopwords="en",
                                stemmer=self._stemmer, show_progress=False)
        self._index = bm25s.BM25(k1=1.5, b=0.75, method="lucene")
        self._index.index(tokens, show_progress=False)
        self._n = len(nodes)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self._bm25s.tokenize([query], stopwords="en", stemmer=self._stemmer, show_progress=False)
        ids, scores = self._index.retrieve(q, k=min(k, self._n), show_progress=False)
        # Chunks sharing no term with the query score 0: they are not matches, drop them.
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if s > 0]


# ----------------------------------------------------------------- fusion
def rrf_fuse(rankings: list[list[tuple[int, float]]], k: int, rrf_k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(chunk) = sum over lists of 1 / (rrf_k + rank).

    Uses ranks, not raw scores, because cosine similarities (~0-1) and BM25 scores
    (unbounded) are on incomparable scales. Ties: better best-rank first, then index.
    """
    fused: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    for ranking in rankings:
        for rank, (i, _) in enumerate(ranking, start=1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + rank)
            best_rank[i] = min(best_rank.get(i, rank), rank)
    order = sorted(fused, key=lambda i: (-fused[i], best_rank[i], i))
    return [(i, fused[i]) for i in order[:k]]


# ----------------------------------------------------------------- reranker
@lru_cache(maxsize=2)
def _cross_encoder(model_name: str, max_length: int):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, max_length=max_length, device="cpu")


class CrossEncoderReranker:
    """Scores (query, chunk) pairs jointly. Inputs longer than max_length tokens
    (query + chunk) are truncated from the end of the chunk."""

    def __init__(self, cfg: dict):
        r = cfg["reranker"]
        self.model = _cross_encoder(r["model_name"], r["max_length"])
        self.batch_size = r["batch_size"]

    def rerank(self, query: str, candidates: list[NodeWithScore], k: int) -> list[NodeWithScore]:
        if not candidates:
            return []
        pairs = [(query, _embed_text(c.node)) for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        order = sorted(range(len(candidates)), key=lambda j: (-float(scores[j]), j))
        return [NodeWithScore(node=candidates[j].node, score=float(scores[j])) for j in order[:k]]


# ----------------------------------------------------------------- the configured retriever
class ConfiguredRetriever(BaseRetriever):
    """retrieval.mode (dense | bm25 | hybrid) + optional reranker, as one LlamaIndex retriever."""

    def __init__(self, cfg: dict, index: faiss.Index, nodes: list[TextNode]):
        super().__init__()
        r = cfg["retrieval"]
        self.mode, self.top_k, self.candidate_k, self.rrf_k = r["mode"], r["top_k"], r["candidate_k"], r["rrf_k"]
        if self.mode not in MODES:
            raise ValueError(f"retrieval.mode must be one of {MODES}, got '{self.mode}'")
        if self.candidate_k < self.top_k:
            raise ValueError("retrieval.candidate_k must be >= retrieval.top_k")
        self.nodes = nodes
        self.dense = DenseSearcher(cfg, index) if self.mode in ("dense", "hybrid") else None
        self.bm25 = BM25Searcher(nodes, r["bm25_stemming"]) if self.mode in ("bm25", "hybrid") else None
        self.reranker = CrossEncoderReranker(cfg) if cfg["reranker"]["enabled"] else None
        self.last_candidates: list[NodeWithScore] = []

    def candidates(self, query: str) -> list[NodeWithScore]:
        k = self.candidate_k
        if self.mode == "dense":
            ranked = self.dense.search(query, k)
        elif self.mode == "bm25":
            ranked = self.bm25.search(query, k)
        else:
            ranked = rrf_fuse([self.dense.search(query, k), self.bm25.search(query, k)], k, self.rrf_k)
        return [NodeWithScore(node=self.nodes[i], score=s) for i, s in ranked]

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query = query_bundle.query_str
        self.last_candidates = self.candidates(query)
        if self.reranker:
            return self.reranker.rerank(query, self.last_candidates, self.top_k)
        return self.last_candidates[: self.top_k]


def build_retriever(cfg: dict, index: faiss.Index, nodes: list[TextNode]) -> ConfiguredRetriever:
    return ConfiguredRetriever(cfg, index, nodes)
