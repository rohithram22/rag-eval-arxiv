"""Streamlit demo: ask questions about 150 RAG papers and see the cited answer,
the retrieved passages, and how retrieval configurations differ.

Run:
    streamlit run app/app.py
"""
from __future__ import annotations

import json
import os
import re
import time

import streamlit as st

from rag_eval.config import load_config, resolve_path
from rag_eval.golden import read_jsonl
from rag_eval.llm_utils import DailyLimitError
from rag_eval.metrics import is_abstention
from rag_eval.pipeline import RAGPipeline

# Configurations evaluated end to end in Step 7 (all with 512-token chunks).
CONFIGS = {
    "Hybrid + reranker (best ranking)": "c512-hybrid-rerank",
    "BM25 (fastest)": "c512-bm25",
    "Dense + reranker": "c512-dense-rerank",
    "Dense (baseline)": "c512-dense",
}
OVERRIDES = {
    "c512-hybrid-rerank": {"retrieval.mode": "hybrid", "reranker.enabled": True},
    "c512-bm25": {"retrieval.mode": "bm25", "reranker.enabled": False},
    "c512-dense-rerank": {"retrieval.mode": "dense", "reranker.enabled": True},
    "c512-dense": {"retrieval.mode": "dense", "reranker.enabled": False},
}
CITATION = re.compile(r"[\[【](\d+)[\]】]")   # [1] or the full-width 【1】 some models emit
MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>#|~$])")


def escape_md(text: str) -> str:
    """Show paper text literally (no accidental bold, LaTeX, links...)."""
    return MD_SPECIAL.sub(r"\\\1", text)


@st.cache_resource(show_spinner="Loading the index and models (first time only)...")
def get_pipeline(label: str) -> RAGPipeline:
    # Interactive use: no pacing pause between generator calls (rate-limit backoff still applies).
    overrides = {**OVERRIDES[label], "eval.generator_min_interval_s": 0}
    return RAGPipeline(load_config(overrides=overrides), verbose=False)


@st.cache_data
def load_summary(label: str) -> dict | None:
    path = resolve_path(load_config(), "results_dir") / label / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_golden() -> list[dict]:
    cfg = load_config()
    return read_jsonl(resolve_path(cfg, "eval_dir") / cfg["eval"]["golden_file"])


def run_query(label: str, question: str) -> dict:
    rag = get_pipeline(label)
    t0 = time.time()
    results = rag.retrieve(question)
    t_retrieve = time.time() - t0
    answer, t_generate, cached = rag.generate(question, results)
    return {
        "label": label, "answer": answer, "cached": cached,
        "t_retrieve": t_retrieve, "t_generate": t_generate,
        "sources": [{"rank": i, "id": r.node.node_id, "title": r.node.metadata["title"],
                     "arxiv_id": r.node.metadata["arxiv_id"], "score": float(r.score), "text": r.node.text}
                    for i, r in enumerate(results, start=1)],
    }


def render_result(res: dict) -> None:
    answer = res["answer"]
    cited = {int(n) for n in CITATION.findall(answer)}
    if is_abstention(answer):
        st.info(answer)
    else:
        st.markdown(answer)

    gen = "cached" if res["cached"] else f"{res['t_generate']:.1f}s"
    c1, c2, c3 = st.columns(3)
    c1.metric("Retrieval", f"{res['t_retrieve']:.2f}s")
    c2.metric("Generation", gen)
    c3.metric("Passages cited", f"{len(cited & set(range(1, len(res['sources']) + 1)))} / {len(res['sources'])}")

    st.markdown("**Retrieved passages** (the only text the model was allowed to use)")
    for s in res["sources"]:
        mark = "  ·  cited" if s["rank"] in cited else ""
        with st.expander(f"[{s['rank']}] {s['title'][:90]}  ·  score {s['score']:.3f}{mark}",
                         expanded=s["rank"] in cited):
            st.markdown(f"[arXiv:{s['arxiv_id']}](https://arxiv.org/abs/{s['arxiv_id']})  ·  chunk `{s['id']}`")
            st.markdown(escape_md(s["text"]))


def sidebar_metrics(label: str) -> None:
    s = load_summary(label)
    if not s or "generation" not in s:
        st.caption("No full evaluation results found for this configuration.")
        return
    r, g = s["retrieval"], s["generation"]
    st.markdown(
        f"| Metric | Value |\n|---|---|\n"
        f"| Answer correctness | {g['correctness']['mean']:.3f} |\n"
        f"| Faithfulness | {g['faithfulness']['mean']:.3f} |\n"
        f"| Retrieval hit@5 | {r['hit@5']['mean']:.3f} |\n"
        f"| Retrieval MRR | {r['mrr']['mean']:.3f} |\n"
        f"| Retrieval latency | {s['latency']['retrieve_median_s']:.2f}s |"
    )
    st.caption(f"Measured on the golden set ({s['n_answerable']} answerable + {s['n_unanswerable']} "
               f"unanswerable questions). See RESULTS.md.")


def main() -> None:
    st.set_page_config(page_title="RAG over ML papers", layout="wide")
    st.title("Ask 150 papers on retrieval-augmented generation")
    st.caption("Answers come only from retrieved passages, with citations. "
               "If the papers don't contain the answer, the system says so.")

    with st.sidebar:
        st.header("Configuration")
        name = st.radio("Retrieval", list(CONFIGS), index=0)
        sidebar_metrics(CONFIGS[name])
        compare = st.checkbox("Compare with another configuration")
        other = None
        if compare:
            others = [n for n in CONFIGS if n != name]
            other = st.selectbox("Compare with", others, index=len(others) - 1)
        st.divider()
        st.caption("Corpus: 150 arXiv papers (2020-2025). Embeddings: bge-small. "
                   "Generator: gpt-oss-20b (Groq). 512-token chunks, top 5 passages.")

    if not os.environ.get("GROQ_API_KEY"):
        st.warning("GROQ_API_KEY is not set: retrieval works, but answers can't be generated.")

    golden = load_golden()
    examples = ["Type your own question below"] + [f"{g['id']} · {g['type']} · {g['question']}" for g in golden]

    def use_example():
        choice = st.session_state["example"]
        if choice != examples[0]:
            st.session_state["question"] = choice.split(" · ", 2)[2]

    st.selectbox("Or pick a question from the evaluation set (answers for these are usually cached)",
                 examples, key="example", on_change=use_example)
    question = st.text_area("Question", key="question", height=80,
                            placeholder="e.g. Why do language models struggle with information in the middle of long contexts?")
    ask = st.button("Ask", type="primary", disabled=not question.strip())

    labels = [CONFIGS[name]] + ([CONFIGS[other]] if other else [])
    if ask:
        results = []
        for label in labels:
            with st.spinner(f"Retrieving and answering with {label}..."):
                try:
                    results.append(run_query(label, question.strip()))
                except DailyLimitError:
                    st.error("The generator's free daily token limit is used up. Try again later "
                             "(questions from the evaluation set may still work: their answers are cached).")
                    return
                except Exception as e:  # noqa: BLE001
                    st.error(f"{type(e).__name__}: {e}")
                    return
        # Keep results across reruns (Streamlit reruns the script on every click, e.g. opening a passage).
        st.session_state["last"] = {"question": question.strip(), "results": results}

    last = st.session_state.get("last")
    if not last:
        return
    st.subheader(last["question"])
    names = {v: k for k, v in CONFIGS.items()}
    if len(last["results"]) == 1:
        render_result(last["results"][0])
    else:
        for col, res in zip(st.columns(len(last["results"])), last["results"]):
            with col:
                st.markdown(f"#### {names[res['label']]}")
                render_result(res)


main()
