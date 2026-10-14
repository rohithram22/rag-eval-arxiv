"""Versioned prompts. A prompt is part of the experiment: if it changes, the
version changes, and results record which version produced them."""
from __future__ import annotations

from llama_index.core.schema import NodeWithScore

PROMPTS = {
    "v1": {
        "system": (
            "You are a research assistant answering questions about machine-learning papers.\n"
            "Rules:\n"
            "1. Use ONLY the numbered context passages below. Do not use outside knowledge.\n"
            "2. Cite the passages that support each claim with their numbers, e.g. [1] or [2][3].\n"
            "3. If the passages do not contain the answer, reply exactly: "
            "\"I don't know based on the provided papers.\"\n"
            "4. Be concise: at most 5 sentences."
        ),
        "user": "Context passages:\n\n{context}\n\nQuestion: {question}\n\nAnswer:",
    },
}

ABSTAIN = "I don't know based on the provided papers."


def format_context(results: list[NodeWithScore]) -> str:
    """Number passages [1]..[k] in rank order, each labelled with its paper."""
    blocks = []
    for rank, r in enumerate(results, start=1):
        meta = r.node.metadata
        blocks.append(f"[{rank}] ({meta['title']}, arXiv:{meta['arxiv_id']})\n{r.node.text}")
    return "\n\n".join(blocks)


def build_messages(version: str, question: str, results: list[NodeWithScore]) -> tuple[str, str]:
    p = PROMPTS[version]
    return p["system"], p["user"].format(context=format_context(results), question=question)
