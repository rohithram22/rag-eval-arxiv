"""Retrievers. Step 4: dense (FAISS). Step 7 adds BM25, hybrid fusion and reranking.

Our retrievers implement LlamaIndex's BaseRetriever interface, so LlamaIndex's
fusion retriever and rerankers can be plugged in on top of them later.
"""
from __future__ import annotations

import faiss
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from rag_eval.indexing import embed_query


class FaissDenseRetriever(BaseRetriever):
    """Embed the query (with bge's query instruction) and return the top-k chunks by cosine."""

    def __init__(self, cfg: dict, index: faiss.Index, nodes: list[TextNode], top_k: int):
        super().__init__()
        self._cfg = cfg
        self._index = index
        self._nodes = nodes
        self._top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        qvec = embed_query(self._cfg, query_bundle.query_str)
        scores, ids = self._index.search(qvec, self._top_k)
        return [
            NodeWithScore(node=self._nodes[i], score=float(s))
            for s, i in zip(scores[0], ids[0])
            if i != -1
        ]


def build_retriever(cfg: dict, index: faiss.Index, nodes: list[TextNode]) -> BaseRetriever:
    mode = cfg["retrieval"]["mode"]
    top_k = cfg["retrieval"]["top_k"]
    if cfg["reranker"]["enabled"]:
        raise NotImplementedError("Reranking is added in Step 7.")
    if mode == "dense":
        return FaissDenseRetriever(cfg, index, nodes, top_k=top_k)
    raise NotImplementedError(f"retrieval.mode='{mode}' is added in Step 7 (only 'dense' for now).")
