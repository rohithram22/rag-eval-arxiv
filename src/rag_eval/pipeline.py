"""The RAG pipeline: retrieve -> build prompt -> generate.

    from rag_eval.config import load_config
    from rag_eval.pipeline import RAGPipeline
    rag = RAGPipeline(load_config(overrides={"chunking.chunk_size": 256}))
    result = rag.answer("How does DPR train its encoders?")

Step 6 change: generation goes through llm_calls.cached_chat (disk cache,
pacing, rate-limit backoff), so the demo, ask.py and the evaluation all
produce answers the same way.
"""
from __future__ import annotations

import time

from rag_eval.indexing import build_or_load_index
from rag_eval.llm_calls import cached_chat
from rag_eval.prompts import build_messages
from rag_eval.retrieval import build_retriever


class RAGPipeline:
    def __init__(self, cfg: dict, verbose: bool = True):
        self.cfg = cfg
        self.index, self.nodes, self.index_meta = build_or_load_index(cfg, verbose=verbose)
        self.retriever = build_retriever(cfg, self.index, self.nodes)

    def retrieve(self, question: str):
        return self.retriever.retrieve(question)

    def generate(self, question: str, results) -> tuple[str, float, bool]:
        """Return (answer, api_seconds, from_cache) for already-retrieved results.

        api_seconds is the LLM request time only (pacing pauses excluded). For a cached answer
        it is the time the original request took (None for older cache entries).
        """
        system, user = build_messages(self.cfg["generator"]["prompt_version"], question, results)
        answer, from_cache, api_s = cached_chat(self.cfg, "generator", [("system", system), ("user", user)],
                                                what="generate")
        return answer, api_s, from_cache

    def answer(self, question: str, results=None) -> dict:
        t0 = time.time()
        results = results if results is not None else self.retrieve(question)
        t_retrieve = time.time() - t0
        answer, t_generate, from_cache = self.generate(question, results)
        return {
            "question": question,
            "answer": answer,
            "contexts": [
                {"rank": rank, "id": r.node.node_id, "arxiv_id": r.node.metadata["arxiv_id"],
                 "title": r.node.metadata["title"], "score": round(r.score, 4), "text": r.node.text}
                for rank, r in enumerate(results, start=1)
            ],
            "latency_s": {"retrieve": round(t_retrieve, 3),
                          "generate": None if t_generate is None else round(t_generate, 3)},
            "from_cache": from_cache,
            "prompt_version": self.cfg["generator"]["prompt_version"],
        }
