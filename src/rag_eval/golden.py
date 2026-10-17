"""Golden evaluation set: passage sampling, question-generation prompt, evidence
location, quality flags, and JSONL storage.

Ground truth is anchored to the PAPER TEXT (papers.jsonl), not to chunks:
each item stores its evidence quote plus character offsets into the paper.
That keeps the golden set valid for every chunk size we test in Step 7.
"""
from __future__ import annotations

import json
import random
import re
from difflib import SequenceMatcher
from pathlib import Path


# ----------------------------------------------------------------- storage
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------- passage sampling
def _letter_ratio(s: str) -> float:
    return sum(c.isalpha() for c in s) / max(1, len(s))


def sample_passage(text: str, rng: random.Random, min_chars: int, max_chars: int,
                   tries: int = 25) -> tuple[int, int] | None:
    """Pick a prose passage (consecutive paragraphs) and return its (start, end) offsets.

    Skips the first blocks (title/authors), tables (low letter ratio) and fragments.
    """
    blocks, pos = [], 0
    for block in text.split("\n\n"):
        blocks.append((pos, pos + len(block)))
        pos += len(block) + 2
    if len(blocks) < 6:
        return None
    for _ in range(tries):
        i = rng.randrange(3, len(blocks) - 1)
        s, e = blocks[i]
        j = i
        while e - s < min_chars and j + 1 < len(blocks):
            j += 1
            e = blocks[j][1]
        if not (min_chars <= e - s <= max_chars):
            continue
        passage = text[s:e]
        if _letter_ratio(passage) < 0.65 or passage.count(". ") < 4:
            continue
        return s, e
    return None


# ----------------------------------------------------------------- prompt
QG_PROMPTS = {
    "qg-v1": {
        "system": (
            "You write evaluation questions for a question-answering system over a collection of "
            "about 150 machine-learning research papers on retrieval-augmented generation (RAG).\n"
            "Given ONE passage from ONE paper, write ONE question with a reference answer.\n\n"
            "Rules:\n"
            "1. The passage alone must fully support the answer.\n"
            "2. Self-contained: someone who has not seen the passage must know what the question is "
            "about. Name the specific method, model, dataset or paper topic. NEVER say 'the authors', "
            "'this paper', 'this work', 'the proposed method' or 'the study' without naming it; "
            "with 150 papers in the collection, such questions are ambiguous.\n"
            "3. Phrase it like a real user would: paraphrase. Do not copy distinctive phrases from the "
            "passage, except names of methods, models, datasets and metrics.\n"
            "4. It must require this paper: not answerable from general ML knowledge alone.\n"
            "5. Do not ask about author names, affiliations, figure/table numbers or section numbers.\n"
            "6. If the passage is unsuitable (mostly a table, references, boilerplate, too fragmentary), "
            "reply {\"skip\": true, \"reason\": \"...\"}.\n\n"
            "Otherwise reply with JSON only:\n"
            "{\"question\": \"...\", \"answer\": \"1-2 sentence reference answer\", "
            "\"evidence\": \"1-3 consecutive sentences copied VERBATIM from the passage that support the answer\"}"
        ),
        "user": "Paper title: {title}\n\nPassage:\n<<<\n{passage}\n>>>\n\nQuestion type: {qtype} ({qtype_desc})",
    }
}

QTYPE_DESC = {
    "factual": "a specific fact stated in the passage",
    "method": "how something works or is done",
    "numeric": "a specific number, score or quantity reported in the passage",
    "why": "the reason or motivation for a design choice or finding",
    "definition": "what a term, component or concept means in this paper",
}

# Hand-written candidates for questions the corpus should NOT answer (abstention test).
# The reviewer confirms with a retrieval preview that no chunk actually answers them.
UNANSWERABLE_CANDIDATES = [
    "What learning rate schedule was used to train the original ResNet-50 on ImageNet?",
    "How many parameters does the U-Net in Stable Diffusion v1.5 have?",
    "What input image resolution does the YOLOv3 object detector use by default?",
    "What is the capital of Australia?",
    "How does AlphaGo Zero's Monte Carlo tree search choose moves during self-play?",
    "What dataset was the WaveNet speech synthesis model originally trained on?",
    "How does the Mamba architecture's selective state space layer differ from S4?",
    "What batch size was used to pre-train the original BERT-Large model?",
]


# ----------------------------------------------------------------- evidence location
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                         "–": "-", "—": "-", "−": "-"})


def _normalize_with_map(s: str) -> tuple[str, list[int]]:
    """Lowercase, unify quotes/dashes, collapse whitespace; keep a map norm_idx -> orig_idx."""
    out, idx_map, prev_space = [], [], False
    for i, ch in enumerate(s.translate(_QUOTES)):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            prev_space = True
        else:
            out.append(ch.lower())
            prev_space = False
        idx_map.append(i)
    return "".join(out), idx_map


def _sentence_spans(s: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+", s):
        spans.append((start, m.start()))
        start = m.end()
    if start < len(s):
        spans.append((start, len(s)))
    return spans


def find_evidence(text: str, evidence: str, fuzzy_threshold: float = 0.85) -> tuple[int, int, str] | None:
    """Locate `evidence` in `text`. Returns (start, end, how) with offsets into `text`.

    1) exact match after normalization (case, whitespace, quote styles)
    2) fuzzy: best window of 1-4 consecutive sentences with similarity >= threshold
    """
    ev_norm, _ = _normalize_with_map(evidence.strip())
    if len(ev_norm) < 20:
        return None
    t_norm, t_map = _normalize_with_map(text)
    k = t_norm.find(ev_norm)
    if k != -1:
        return t_map[k], t_map[k + len(ev_norm) - 1] + 1, "exact"

    spans = _sentence_spans(text)
    best = (0.0, None)
    for i in range(len(spans)):
        for n in range(1, 5):
            if i + n > len(spans):
                break
            s, e = spans[i][0], spans[i + n - 1][1]
            cand, _ = _normalize_with_map(text[s:e])
            if abs(len(cand) - len(ev_norm)) > 0.5 * len(ev_norm):
                continue
            r = SequenceMatcher(None, cand, ev_norm, autojunk=False).ratio()
            if r > best[0]:
                best = (r, (s, e))
    if best[1] and best[0] >= fuzzy_threshold:
        return best[1][0], best[1][1], f"fuzzy({best[0]:.2f})"
    return None


# ----------------------------------------------------------------- quality flags
STOPWORDS = set("""a an the of in on for to and or is are was were be been by with from as at that this
which what how why when where who whose does do did it its their there these those than then into over
under between about can could would should may might use used using based paper method model""".split())
AMBIGUOUS = re.compile(r"\b(the authors|this paper|this work|this study|the study|the proposed|"
                       r"our method|our approach|the paper|in this approach)\b", re.I)


def content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9\-]+", s.lower()) if w not in STOPWORDS}


def lexical_overlap(question: str, evidence: str) -> float:
    """Share of the question's content words that also appear in the evidence.

    High overlap = the question copies the passage's wording, which makes keyword
    search (BM25) look better than it would on real user questions.
    """
    q = content_words(question)
    return len(q & content_words(evidence)) / len(q) if q else 0.0


def quality_flags(item: dict) -> list[str]:
    flags = []
    q = item.get("question", "")
    if AMBIGUOUS.search(q):
        flags.append("ambiguous_reference")
    n_words = len(q.split())
    if n_words < 6 or n_words > 40:
        flags.append(f"length({n_words} words)")
    if item.get("answerable", True):
        ov = lexical_overlap(q, item.get("evidence", ""))
        if ov > 0.7:
            flags.append(f"high_overlap({ov:.2f})")
    return flags
