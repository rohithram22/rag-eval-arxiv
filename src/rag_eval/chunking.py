"""Parsed papers -> chunks (LlamaIndex TextNodes), with a sentence-packing chunker.

Why a custom chunker instead of LlamaIndex's SentenceSplitter: SentenceSplitter
carries overlap only in whole units that FIT the overlap budget. Academic
sentences average ~25-30 tokens, so with a 16-token overlap (chunk size 128)
almost nothing fits and most chunks silently get zero overlap. That would make
the chunk-size experiment unfair (small chunks would also mean less overlap).

Policy implemented here, identical for every chunk size:
- Chunks are packs of whole sentences; a sentence is never cut, unless it alone
  exceeds the chunk budget (flattened tables do), then it's split by tokens.
- Overlap = the last sentences of the previous chunk, up to chunk_overlap tokens,
  but ALWAYS at least one sentence (unless that sentence is over half the budget).
- chunk_size is the FULL embedding window: title header + chunk text + the 2
  special tokens ([CLS], [SEP]) all fit inside it, so bge never truncates.
- Tokens are counted with the embedding model's own tokenizer.
- Deterministic IDs: "<arxiv base id>#<chunk index>".
"""
from __future__ import annotations

import json
from functools import lru_cache

from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
from llama_index.core.schema import MetadataMode, TextNode

from rag_eval.config import resolve_path

SPECIAL_TOKENS = 2   # [CLS] + [SEP] added by the embedding model
SAFETY_TOKENS = 4    # sum of per-sentence counts can differ slightly from the joined text


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str):
    """The embedding model's own (fast) tokenizer."""
    from transformers import AutoTokenizer
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()  # silence "sequence longer than max length" warnings
    return AutoTokenizer.from_pretrained(model_name)


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


@lru_cache(maxsize=1)
def _sentence_splitter():
    return split_by_sentence_tokenizer()  # NLTK punkt; pieces concatenate back to the input


def load_papers(cfg: dict) -> list[dict]:
    path = resolve_path(cfg, "processed_dir") / "papers.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/parse_corpus.py")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _sentences_with_spans(text: str, tokenizer, budget: int) -> list[tuple[int, int, int]]:
    """(start_char, end_char, n_tokens) per sentence; over-long sentences are split by tokens."""
    spans = []
    pos = 0
    for piece in _sentence_splitter()(text):
        start, end = pos, pos + len(piece)
        pos = end
        if not piece.strip():
            continue
        n = count_tokens(tokenizer, piece)
        if n <= budget:
            spans.append((start, end, n))
            continue
        # Over-long "sentence" (e.g. a flattened table): cut into budget-sized token windows.
        enc = tokenizer(piece, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        for k in range(0, len(offsets), budget):
            window = offsets[k:k + budget]
            spans.append((start + window[0][0], start + window[-1][1], len(window)))
    return spans


def _pack(spans: list[tuple[int, int, int]], budget: int, overlap: int) -> list[tuple[int, int]]:
    """Greedily pack sentences into chunks of <= budget tokens, with sentence-level overlap."""
    chunks = []
    i, n = 0, len(spans)
    while i < n:
        j, used = i, 0
        while j < n and used + spans[j][2] <= budget:
            used += spans[j][2]
            j += 1
        if j == i:          # cannot happen (spans are <= budget), but never loop forever
            j = i + 1
        chunks.append((spans[i][0], spans[j - 1][1]))
        if j >= n:
            break
        # Step back k sentences for overlap: at least one, up to `overlap` tokens,
        # never the whole chunk (we must make progress).
        k, ov = 0, 0
        while k < (j - i) - 1:
            s_tokens = spans[j - 1 - k][2]
            if k == 0:
                if s_tokens > budget // 2:   # one huge sentence: don't duplicate it
                    break
            elif ov + s_tokens > overlap:
                break
            ov += s_tokens
            k += 1
        i = j - k
    return chunks


def chunk_corpus(cfg: dict, papers: list[dict] | None = None) -> list[TextNode]:
    """Chunk the whole corpus according to cfg['chunking']."""
    papers = papers if papers is not None else load_papers(cfg)
    c = cfg["chunking"]
    tokenizer = get_tokenizer(cfg["embedding"]["model_name"])
    hidden = ["arxiv_id"] + ([] if c["include_title"] else ["title"])

    nodes: list[TextNode] = []
    for p in papers:
        metadata = {"arxiv_id": p["arxiv_id"], "title": p["title"]}
        header_tokens = count_tokens(tokenizer, f"title: {p['title']}") if c["include_title"] else 0
        budget = c["chunk_size"] - SPECIAL_TOKENS - SAFETY_TOKENS - header_tokens
        if budget < 32:
            raise ValueError(f"chunk_size {c['chunk_size']} too small for title of {p['arxiv_id']}")

        text = p["text"]
        spans = _sentences_with_spans(text, tokenizer, budget)
        for idx, (start, end) in enumerate(_pack(spans, budget, c["chunk_overlap"])):
            nodes.append(TextNode(
                id_=f"{p['base_id']}#{idx}",
                text=text[start:end].strip(),
                metadata=dict(metadata),
                excluded_embed_metadata_keys=list(hidden),
                excluded_llm_metadata_keys=list(hidden),
                start_char_idx=start,
                end_char_idx=end,
            ))
    return nodes


def node_to_record(node: TextNode) -> dict:
    """Plain-dict view of a chunk (for saving / inspection)."""
    return {
        "id": node.node_id,
        "arxiv_id": node.metadata["arxiv_id"],
        "title": node.metadata["title"],
        "start_char": node.start_char_idx,
        "end_char": node.end_char_idx,
        "text": node.text,
        "embed_text": node.get_content(metadata_mode=MetadataMode.EMBED),
    }
