"""Retrieval metrics computed from paper-anchored evidence offsets, plus abstention detection.

A retrieved chunk is RELEVANT to a golden item when it comes from the item's paper
and overlaps the evidence span enough:

    relevance = overlap_chars / min(len(evidence), len(chunk))   >= threshold

Dividing by the SMALLER of the two keeps the rule fair across chunk sizes: a small
chunk counts if most of it is evidence; a large chunk counts if it contains most of
the evidence. (Requiring the whole evidence would penalise small chunks; accepting
any overlap at all would reward large chunks for merely touching it.)
"""
from __future__ import annotations

import re

ABSTAIN_RE = re.compile(r"\bi (?:don['’]t|do not) know\b", re.I)


def is_abstention(answer: str) -> bool:
    """True if the answer declines (the prompt's exact sentence, or close variants)."""
    return bool(ABSTAIN_RE.search(answer[:150]))


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def chunk_relevance(chunk: dict, item: dict) -> float:
    if chunk["base_id"] != item["base_id"]:
        return 0.0
    es, ee = item["evidence_start"], item["evidence_end"]
    cs, ce = chunk["start"], chunk["end"]
    ov = _overlap(cs, ce, es, ee)
    return ov / max(1, min(ee - es, ce - cs))


def evidence_coverage(chunks: list[dict], item: dict) -> float:
    """Share of the evidence characters covered by the union of retrieved chunks."""
    es, ee = item["evidence_start"], item["evidence_end"]
    spans = sorted((max(c["start"], es), min(c["end"], ee))
                   for c in chunks if c["base_id"] == item["base_id"])
    covered, cur_s, cur_e = 0, None, None
    for s, e in spans:
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            covered += (cur_e - cur_s) if cur_e is not None else 0
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    covered += (cur_e - cur_s) if cur_e is not None else 0
    return covered / max(1, ee - es)


def retrieval_metrics(chunks: list[dict], item: dict, ks: list[int], threshold: float) -> dict:
    """chunks: ranked list of {'base_id', 'start', 'end', ...}. Returns per-question metrics."""
    rels = [chunk_relevance(c, item) for c in chunks]
    first = next((rank for rank, r in enumerate(rels, start=1) if r >= threshold), None)
    out = {f"hit@{k}": float(first is not None and first <= k) for k in ks}
    out.update({
        "mrr": 1.0 / first if first else 0.0,
        "first_relevant_rank": first,
        "coverage": evidence_coverage(chunks, item),
        "paper_hit": float(any(c["base_id"] == item["base_id"] for c in chunks)),
        "max_relevance": round(max(rels), 3) if rels else 0.0,
    })
    return out
