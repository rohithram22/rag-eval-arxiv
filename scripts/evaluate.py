"""Step 6: evaluate one RAG configuration on the golden set.

Usage:
    python scripts/evaluate.py --retrieval-only               # baseline, retrieval metrics only (~1 min)
    python scripts/evaluate.py --limit 5                      # quick full trial -> results/trial
    python scripts/evaluate.py                                # full baseline -> results/c512-dense
    python scripts/evaluate.py --chunk-size 256 --retrieval-only
    python scripts/evaluate.py --set retrieval.top_k=3 --label c512-dense-k3

Re-running is cheap: every LLM reply is cached in .cache/llm/. If a daily rate
limit stops a run, run the same command later; finished answers are reused.
"""
from __future__ import annotations

import argparse
import sys

import yaml

from rag_eval.config import load_config
from rag_eval.evaluation import default_label, format_summary, run_evaluation
from rag_eval.llm_utils import DailyLimitError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrieval-only", action="store_true", help="no LLM calls; retrieval metrics only")
    parser.add_argument("--chunk-size", type=int, help="sets chunk_size and chunk_overlap = size // 8")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="any config override, e.g. retrieval.mode=hybrid (repeatable)")
    parser.add_argument("--label", help="results folder name (default: derived from the config)")
    parser.add_argument("--limit", type=int, help="only the first N golden questions (trial run)")
    args = parser.parse_args()

    overrides = {}
    if args.chunk_size:
        overrides.update({"chunking.chunk_size": args.chunk_size, "chunking.chunk_overlap": args.chunk_size // 8})
    for kv in args.set:
        key, _, value = kv.partition("=")
        overrides[key.strip()] = yaml.safe_load(value)
    cfg = load_config(overrides=overrides)

    label = args.label or ("trial" if args.limit else default_label(cfg))
    print(f"Evaluating '{label}' ({'retrieval only' if args.retrieval_only else 'full'})")
    try:
        summary = run_evaluation(cfg, label, retrieval_only=args.retrieval_only, limit=args.limit)
    except DailyLimitError as e:
        print(f"\nSTOPPED: {e}\nRun the same command later: cached answers and verdicts are reused.")
        return 2
    print(format_summary(summary))
    print(f"Saved to results/{label}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
