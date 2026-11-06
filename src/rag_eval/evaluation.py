"""Run one configuration over the golden set and save per-question rows + a summary.

Two modes:
- retrieval-only: hit@k, MRR, evidence coverage, paper hit. Deterministic, free,
  needs no API key. Used to screen all configurations in Step 7.
- full: also generates answers (generator LLM) and scores them (judge LLM).

Output: results/<label>/{summary.json, per_question.jsonl}
"""
from __future__ import annotations

import json
import time
from statistics import median

from rag_eval.config import config_hash, resolve_path
from rag_eval.golden import read_jsonl, write_jsonl
from rag_eval.judge import judge_answer
from rag_eval.metrics import is_abstention, retrieval_metrics
from rag_eval.pipeline import RAGPipeline
from rag_eval.prompts import format_context
from rag_eval.stats import bootstrap_ci

RETRIEVAL_METRICS = ["hit@1", "hit@3", "hit@5", "mrr", "coverage", "paper_hit", "pool_hit"]
GENERATION_METRICS = ["correctness", "faithfulness", "relevance"]


def default_label(cfg: dict) -> str:
    rr = "-rerank" if cfg["reranker"]["enabled"] else ""
    return f"c{cfg['chunking']['chunk_size']}-{cfg['retrieval']['mode']}{rr}"


def _chunk_view(results) -> list[dict]:
    return [{"rank": rank, "id": r.node.node_id, "base_id": r.node.node_id.split("#")[0],
             "start": r.node.start_char_idx, "end": r.node.end_char_idx, "score": round(float(r.score), 4)}
            for rank, r in enumerate(results, start=1)]


def evaluate_item(rag: RAGPipeline, item: dict, cfg: dict, retrieval_only: bool) -> dict:
    e = cfg["eval"]
    t0 = time.time()
    results = rag.retrieve(item["question"])
    t_retrieve = time.time() - t0
    chunks = _chunk_view(results)

    row = {"id": item["id"], "type": item["type"], "answerable": item["answerable"],
           "question": item["question"], "gold_paper": item.get("base_id"),
           "retrieved": [{k: c[k] for k in ("rank", "id", "score")} for c in chunks],
           "t_retrieve": round(t_retrieve, 3)}
    if item["answerable"]:
        row.update(retrieval_metrics(chunks, item, e["ks"], e["relevance_threshold"]))
        # Candidate pool (before cutting to top_k / reranking): the reranker's ceiling.
        pool = _chunk_view(getattr(rag.retriever, "last_candidates", []) or results)
        pool_rank = retrieval_metrics(pool, item, [], e["relevance_threshold"])["first_relevant_rank"]
        row.update(pool_rank=pool_rank, pool_hit=float(pool_rank is not None), pool_size=len(pool))
    if retrieval_only:
        return row

    answer, t_gen, cached = rag.generate(item["question"], results)
    abstained = is_abstention(answer)
    row.update(answer=answer, abstained=abstained,
               t_generate=None if t_gen is None else round(t_gen, 3), gen_cached=cached)
    if not item["answerable"]:
        return row
    if abstained:
        # Declining an answerable question: wrong answer, but no claims to check.
        row.update(correctness=0.0, faithfulness=None, relevance=None, claims=[], judge_error=None)
        return row
    row.update(judge_answer(cfg, item["question"], item["answer"], format_context(results), answer))
    return row


def summarize(rows: list[dict], cfg: dict, label: str, mode: str, index_key: str) -> dict:
    n_boot, seed = cfg["eval"]["bootstrap_samples"], cfg["project"]["seed"]
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    summary = {
        "label": label, "mode": mode, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_questions": len(rows), "n_answerable": len(ans), "n_unanswerable": len(una),
        "index_key": index_key,
        "config_hash": config_hash(cfg, ["parsing", "chunking", "embedding", "retrieval", "reranker"]),
        "retrieval": {m: bootstrap_ci([r[m] for r in ans], n_boot, seed) for m in RETRIEVAL_METRICS},
        "latency": {"retrieve_median_s": round(median(r["t_retrieve"] for r in rows), 3)},
        "config": cfg,
    }
    if mode == "full":
        gen = {m: bootstrap_ci([r.get(m) for r in ans], n_boot, seed) for m in GENERATION_METRICS}
        gen["false_abstention"] = bootstrap_ci([float(r["abstained"]) for r in ans], n_boot, seed)
        gen["unanswerable_abstention"] = bootstrap_ci([float(r["abstained"]) for r in una], n_boot, seed)
        gen["n_judged"] = sum(1 for r in ans if not r["abstained"] and not r.get("judge_error"))
        gen["judge_errors"] = sum(1 for r in ans if r.get("judge_error"))
        summary["generation"] = gen
        times = [r["t_generate"] for r in rows if r.get("t_generate") is not None]
        summary["latency"]["generate_median_s"] = round(median(times), 3) if times else None
        summary["latency"]["generate_timed_n"] = len(times)
    return summary


def run_evaluation(cfg: dict, label: str, retrieval_only: bool = False, limit: int | None = None,
                   verbose: bool = True) -> dict | None:
    mode = "retrieval" if retrieval_only else "full"
    out_dir = resolve_path(cfg, "results_dir") / label
    existing = out_dir / "summary.json"
    if retrieval_only and existing.exists() and json.loads(existing.read_text())["mode"] == "full":
        print(f"results/{label} already has FULL results (retrieval metrics are identical); not overwriting.")
        return json.loads(existing.read_text())

    golden = read_jsonl(resolve_path(cfg, "eval_dir") / cfg["eval"]["golden_file"])
    if limit:
        golden = golden[:limit]
    rag = RAGPipeline(cfg, verbose=verbose)

    rows = []
    for i, item in enumerate(golden, start=1):
        row = evaluate_item(rag, item, cfg, retrieval_only)
        rows.append(row)
        if verbose:
            parts = []
            if item["answerable"]:
                parts.append(f"rank={row['first_relevant_rank'] or '-'}")
            if not retrieval_only:
                if row["abstained"]:
                    parts.append("ABSTAINED")
                elif item["answerable"]:
                    f = row.get("faithfulness")
                    parts.append(f"corr={row.get('correctness_label') or 'ERR'} "
                                 f"faith={'-' if f is None else f'{f:.2f}'}")
                else:
                    parts.append("ANSWERED (should abstain)")
                if row.get("gen_cached"):
                    parts.append("(cached)")
            print(f"  [{i:>2}/{len(golden)}] {item['id']:<5} {item['type']:<12} {'  '.join(parts)}")

    summary = summarize(rows, cfg, label, mode, rag.index_meta["key"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "per_question.jsonl", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def format_summary(s: dict) -> str:
    def fmt(m):
        if m["mean"] is None:
            return "   n/a"
        return f"{m['mean']:.3f}  [{m['lo']:.2f}, {m['hi']:.2f}]  n={m['n']}"

    lines = [f"\n=========== {s['label']} ({s['mode']}) ===========",
             f"questions: {s['n_questions']} ({s['n_answerable']} answerable, {s['n_unanswerable']} unanswerable)",
             "RETRIEVAL (answerable only)          mean   95% CI"]
    pool_k = s["config"]["retrieval"]["candidate_k"]
    names = {"pool_hit": f"pool_hit@{pool_k} (candidates)"}
    for m, v in s["retrieval"].items():
        lines.append(f"  {names.get(m, m):<32} {fmt(v)}")
    if "generation" in s:
        g = s["generation"]
        lines.append("GENERATION")
        for m in ("correctness", "faithfulness", "relevance", "false_abstention", "unanswerable_abstention"):
            lines.append(f"  {m:<32} {fmt(g[m])}")
        lines.append(f"  judged answers: {g['n_judged']}   judge errors: {g['judge_errors']}")
    lat = s["latency"]
    line = f"LATENCY  retrieve median {lat['retrieve_median_s']}s"
    if "generate_median_s" in lat:
        g = lat["generate_median_s"]
        line += (f" | generate median {g}s (over {lat['generate_timed_n']} timed answers)" if g is not None
                 else " | generate median n/a (no timed answers)")
    lines.append(line)
    lines.append("=" * 50)
    return "\n".join(lines)
