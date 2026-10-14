"""The RAG pipeline: retrieve -> build prompt -> generate.

    from rag_eval.config import load_config
    from rag_eval.pipeline import RAGPipeline
    rag = RAGPipeline(load_config(overrides={"chunking.chunk_size": 256}))
    result = rag.answer("How does DPR train its encoders?")
"""
from __future__ import annotations

import time

from llama_index.core.llms import ChatMessage, MessageRole

from rag_eval.indexing import build_or_load_index
from rag_eval.llms import get_llm
from rag_eval.prompts import build_messages
from rag_eval.retrieval import build_retriever


class RAGPipeline:
    def __init__(self, cfg: dict, verbose: bool = True):
        self.cfg = cfg
        self.index, self.nodes, self.index_meta = build_or_load_index(cfg, verbose=verbose)
        self.retriever = build_retriever(cfg, self.index, self.nodes)
        self._llm = None  # created lazily: retrieval-only runs need no API key

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm(self.cfg["generator"])
        return self._llm

    def retrieve(self, question: str):
        return self.retriever.retrieve(question)

    def answer(self, question: str, results=None) -> dict:
        t0 = time.time()
        results = results if results is not None else self.retrieve(question)
        t_retrieve = time.time() - t0

        system, user = build_messages(self.cfg["generator"]["prompt_version"], question, results)
        t1 = time.time()
        response = self.llm.chat([
            ChatMessage(role=MessageRole.SYSTEM, content=system),
            ChatMessage(role=MessageRole.USER, content=user),
        ])
        t_generate = time.time() - t1

        return {
            "question": question,
            "answer": (response.message.content or "").strip(),
            "contexts": [
                {
                    "rank": rank,
                    "id": r.node.node_id,
                    "arxiv_id": r.node.metadata["arxiv_id"],
                    "title": r.node.metadata["title"],
                    "score": round(r.score, 4),
                    "text": r.node.text,
                }
                for rank, r in enumerate(results, start=1)
            ],
            "latency_s": {"retrieve": round(t_retrieve, 3), "generate": round(t_generate, 3)},
            "prompt_version": self.cfg["generator"]["prompt_version"],
        }
