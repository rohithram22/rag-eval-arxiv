"""PDF -> clean text for one paper.

Pipeline per page: text blocks (in the PDF's drawing order, which for LaTeX
papers is column by column) -> drop images, headers/footers, page numbers
-> clean each block -> join blocks as paragraphs. Then cut the References
section (but keep appendices that follow it), verified by checking that the
blocks after the heading actually look like reference entries.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pymupdf

# Vertical arXiv stamp on page 1, e.g. "arXiv:2005.11401v4 [cs.CL] 12 Apr 2021"
ARXIV_STAMP = re.compile(
    r"arXiv:\s?\d{4}\.\d{4,5}(v\d+)?\s*\[[^\]]+\]\s*\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4}"
)
# Case-sensitive on purpose: "References ..." starts a section, "references are ..." is prose.
# Also matches a heading merged with the first entry: "7. REFERENCES [1] Maximum ...".
REF_HEADING = re.compile(
    r"^(\d+\.?\s*)?(References|REFERENCES|Bibliography|BIBLIOGRAPHY|Reference List)(\s*:)?(\s|$)"
)
# What reference entries contain: a year, "et al.", or a "[12]" marker.
REF_ENTRY_HINT = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b|\bet al\.|\[\d+\]")
REF_WINDOW_CHARS = 2500   # text inspected after a candidate heading
REF_MIN_HINTS = 6         # a reference list has ~1 year per 150-250 chars; prose has far fewer
APPENDIX_HEADING = re.compile(
    r"^\s*(appendix|appendices|supplementary material|"
    r"[A-H](\.\d+)*\.?\s+[A-Z][A-Za-z].{0,80})\s*$"
)
YEAR = re.compile(r"\b(19|20)\d{2}\b")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u00ad\ufffd]")


def clean_block(raw: str) -> str:
    """Normalize one text block into a single clean paragraph string."""
    t = unicodedata.normalize("NFKC", raw)          # ligatures: "ﬁ" -> "fi"
    t = ARXIV_STAMP.sub(" ", t)
    t = re.sub(r"(\w)-\n\s*(?=[a-z])", r"\1", t)   # "retrie-\nval" -> "retrieval"
    t = re.sub(r"\s*\n\s*", " ", t)                 # join wrapped lines
    t = CONTROL_CHARS.sub("", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def _page_blocks(page: pymupdf.Page, margin_frac: float) -> list[str]:
    """Raw text blocks of one page, minus images and header/footer noise."""
    height = page.rect.height
    blocks = []
    for x0, y0, x1, y1, text, _block_no, block_type in page.get_text("blocks"):
        if block_type != 0:          # 1 = image block
            continue
        raw = text.strip()
        if not raw:
            continue
        in_margin = y1 < height * margin_frac or y0 > height * (1 - margin_frac)
        if in_margin and len(raw) < 120:   # running headers, page numbers, footers
            continue
        blocks.append(raw)
    return blocks


def _looks_like_reference_list(blocks: list[str], i: int) -> bool:
    """Is the text starting at block i dense with bibliography markers?

    Measured over characters, not blocks: some PDFs extract an entire reference
    list as one or two giant blocks, so counting blocks would miss it.
    """
    window, n = [], 0
    for b in blocks[i:]:
        window.append(b)
        n += len(b)
        if n >= REF_WINDOW_CHARS:
            break
    text = " ".join(window)[:REF_WINDOW_CHARS]
    return sum(1 for _ in REF_ENTRY_HINT.finditer(text)) >= REF_MIN_HINTS


def _drop_references(blocks: list[str]) -> tuple[list[str], bool]:
    """Remove blocks from a References heading up to an appendix heading (or the end).

    A heading only counts if the text right after it is dense with reference
    markers (years, "et al.", [n]). This content check replaces a position-based guard, which failed on
    papers whose appendices are longer than their main text (e.g. RETRO).
    """
    out: list[str] = []
    skipping = found = False
    for i, text in enumerate(blocks):
        if not skipping and not found and REF_HEADING.match(text) and _looks_like_reference_list(blocks, i):
            skipping = found = True
            continue
        if skipping and len(text) <= 100 and APPENDIX_HEADING.match(text) and not YEAR.search(text):
            skipping = False
        if not skipping:
            out.append(text)
    return out, found


def parse_pdf(path: str | Path, margin_frac: float = 0.06, drop_references: bool = True) -> dict:
    """Return {'text', 'n_pages', 'raw_chars', 'n_blocks', 'refs_removed'}."""
    raw_blocks: list[str] = []
    raw_chars = 0
    with pymupdf.open(path) as doc:
        n_pages = doc.page_count
        for page in doc:
            raw_chars += len(page.get_text())
            raw_blocks.extend(_page_blocks(page, margin_frac))

    cleaned = [clean_block(b) for b in raw_blocks]
    cleaned = [c for c in cleaned if len(c) >= 3 and not c.isdigit()]  # stray page numbers

    refs_removed = False
    if drop_references:
        cleaned, refs_removed = _drop_references(cleaned)

    text = "\n\n".join(cleaned)
    return {
        "text": text,
        "n_pages": n_pages,
        "raw_chars": raw_chars,
        "n_blocks": len(cleaned),
        "refs_removed": refs_removed,
    }