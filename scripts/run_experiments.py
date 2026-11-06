"""Step 7: run the experiment grid.

Stage 1 (screen): retrieval-only evaluation of every configuration. Free and
deterministic (no LLM calls): ~1 min per configuration, a few minutes with the reranker.
Stage 2 (full): generation + judging for the finalists chosen by compare.py.

Usage:
    python scripts/run_experiments.py --list
    python scripts/run_experiments.py --stage screen
    python scripts/run_experiments.py --stage full --labels c256-hybrid-rerank c512-hybrid-rerank
Finished configurations are skipped (use --force to re-run).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from rag_eval.config import load_config, resolve_path
from rag_eval.evaluation import default_label, format_summary, run_evaluation
from rag_eval.llm_utils import DailyLimitError

CHUNK_SIZES = [128, 256, 512]
# (retrieval mode, reranker). 2 x 2 factorial on {dense, hybrid} x {reranker off, on},
# plus BM25 alone as a diagnostic (how good is keyword search by itself?).
VARIANTS = [("dense", False), ("dense", True), ("hybrid", False), ("hybrid", True), ("bm25", False)]


def grid() -> dict[str, dict]:
    configs = {}
    for size in CHUNK_SIZES:
        for mode, rerank in VARIANTS:
            overrides = {"chunking.chunk_size": size, "chunking.chunk_overlap": size // 8,
                         "retrieval.mode": mode, "reranker.enabled": rerank}
            configs[default_label(load_config(overrides=overrides))] = overrides
    return configs


def is_done(label: str, need_full: bool) -> bool:
    path = resolve_path(load_config(), "results_dir") / label / "summary.json"
    if not path.exists():
        return False
    s = json.loads(path.read_text())
    has_pool = "pool_hit" in s["retrieval"]
    return has_pool and (s["mode"] == "full" or not need_full)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["screen", "full"])
    parser.add_argument("--labels", nargs="+", help="configurations for --stage full")
    parser.add_argument("--list", action="store_true", help="show the grid and what is done")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    configs = grid()

    if args.list or not args.stage:
        for label in configs:
            state = "full" if is_done(label, True) else "screened" if is_done(label, False) else "-"
            print(f"  {label:<22} {state}")
        return 0

    if args.stage == "screen":
        todo = list(configs)
    else:
        if not args.labels:
            parser.error("--stage full needs --labels (see compare.py for the finalists)")
        unknown = [l for l in args.labels if l not in configs]
        if unknown:
            parser.error(f"unknown labels {unknown}; run --list")
        todo = args.labels
    need_full = args.stage == "full"

    results_dir = resolve_path(load_config(), "results_dir")
    for n, label in enumerate(todo, start=1):
        if not args.force and is_done(label, need_full):
            print(f"[{n}/{len(todo)}] {label}: already done, skipping")
            continue
        old = results_dir / label / "summary.json"
        if not need_full and old.exists() and json.loads(old.read_text())["mode"] == "full":
            print(f"[{n}/{len(todo)}] {label}: has FULL results from before Step 7 (no pool_hit). "
                  f"Refresh it with the full command (answers come from the cache), then re-run the screen.")
            continue
        print(f"\n[{n}/{len(todo)}] {label} ({args.stage})")
        t0 = time.time()
        try:
            summary = run_evaluation(load_config(overrides=configs[label]), label,
                                     retrieval_only=not need_full, verbose=need_full)
        except DailyLimitError as e:
            print(f"\nSTOPPED: {e}\nRe-run the same command later: finished answers are cached.")
            return 2
        r = summary["retrieval"]
        print(f"   hit@5={r['hit@5']['mean']:.3f}  mrr={r['mrr']['mean']:.3f}  "
              f"pool_hit={r['pool_hit']['mean']:.3f}  ({time.time() - t0:.0f}s)")
        if need_full:
            print(format_summary(summary))
    print("\nDone. Next: python scripts/compare.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
