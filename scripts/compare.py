"""Step 7: compare all evaluated configurations.

Prints (and writes results/comparison.md):
  1. the leaderboard, sorted by MRR
  2. one-factor effects: every pair of configurations that differ in exactly ONE
     setting, compared with a paired bootstrap on the same questions
  3. the finalists for full evaluation, chosen by a rule fixed in advance
  4. if full runs exist: end-to-end comparison with the baseline

Usage:
    python scripts/compare.py
    python scripts/compare.py --baseline c512-dense --metric mrr
"""
from __future__ import annotations

import argparse
import json
import sys

from rag_eval.config import load_config, resolve_path
from rag_eval.golden import read_jsonl
from rag_eval.stats import paired_bootstrap

N_FINALISTS = 3
FACTOR_NAMES = {0: "chunk size", 1: "retrieval mode", 2: "reranker"}


def load_runs(results_dir) -> dict[str, dict]:
    runs = {}
    for path in sorted(results_dir.glob("*/summary.json")):
        label = path.parent.name
        if label == "trial":
            continue
        s = json.loads(path.read_text())
        c = s["config"]
        rows = {r["id"]: r for r in read_jsonl(path.parent / "per_question.jsonl")}
        runs[label] = {"summary": s, "rows": rows,
                       "key": (c["chunking"]["chunk_size"], c["retrieval"]["mode"], c["reranker"]["enabled"])}
    return runs


def mean(s: dict, section: str, m: str):
    return s.get(section, {}).get(m, {}).get("mean")


def fmt(v):
    return "  -  " if v is None else f"{v:.3f}"


def per_q(run: dict, metric: str) -> dict:
    return {i: r.get(metric) for i, r in run["rows"].items() if r["answerable"]}


def diff_str(d: dict) -> str:
    if d["diff"] is None:
        return "n/a"
    if d["lo"] == d["hi"] == 0:
        return "0 (identical on every question)"
    sig = "*" if d["lo"] > 0 or d["hi"] < 0 else " "
    return f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]{sig} P(B>A)={d['p_b_better']:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="c512-dense")
    parser.add_argument("--metric", default="mrr", help="retrieval metric for effects/finalists")
    args = parser.parse_args()

    cfg = load_config()
    results_dir = resolve_path(cfg, "results_dir")
    runs = load_runs(results_dir)
    if args.baseline not in runs:
        print(f"Baseline '{args.baseline}' not found in results/.")
        return 1
    if any("pool_hit" not in r["summary"]["retrieval"] for r in runs.values()):
        stale = [l for l, r in runs.items() if "pool_hit" not in r["summary"]["retrieval"]]
        print(f"Re-run these with the Step 7 code first (pool_hit missing): {stale}")
        return 1
    n_boot, seed = cfg["eval"]["bootstrap_samples"], cfg["project"]["seed"]
    md: list[str] = []

    def out(line: str = ""):
        print(line)
        md.append(line)

    # ---- 1. leaderboard
    order = sorted(runs, key=lambda l: (-(mean(runs[l]["summary"], "retrieval", args.metric) or 0),
                                        -(mean(runs[l]["summary"], "retrieval", "hit@5") or 0)))
    n_ans = runs[args.baseline]["summary"]["n_answerable"]
    out(f"## Leaderboard (sorted by {args.metric}; retrieval metrics on {n_ans} answerable questions)\n")
    out("| config | stage | hit@1 | hit@5 | MRR | pool hit@20 | correctness | faithfulness | false abst. | unans. abst. | retrieve s |")
    out("|---|---|---|---|---|---|---|---|---|---|---|")
    for l in order:
        s = runs[l]["summary"]
        g = "generation" in s
        out(f"| {l}{' (baseline)' if l == args.baseline else ''} | {s['mode']} "
            f"| {fmt(mean(s, 'retrieval', 'hit@1'))} | {fmt(mean(s, 'retrieval', 'hit@5'))} "
            f"| {fmt(mean(s, 'retrieval', 'mrr'))} | {fmt(mean(s, 'retrieval', 'pool_hit'))} "
            f"| {fmt(mean(s, 'generation', 'correctness')) if g else '-'} "
            f"| {fmt(mean(s, 'generation', 'faithfulness')) if g else '-'} "
            f"| {fmt(mean(s, 'generation', 'false_abstention')) if g else '-'} "
            f"| {fmt(mean(s, 'generation', 'unanswerable_abstention')) if g else '-'} "
            f"| {s['latency']['retrieve_median_s']} |")

    # ---- 2. one-factor effects
    by_key = {r["key"]: l for l, r in runs.items()}
    out(f"\n## One-factor effects on {args.metric} (B - A, paired bootstrap 95% CI; * = CI excludes 0)\n")
    out("| factor | A | B | difference |")
    out("|---|---|---|---|")
    for factor in (2, 1, 0):
        for ka, la in sorted(by_key.items(), key=lambda x: str(x[0])):
            for kb, lb in sorted(by_key.items(), key=lambda x: str(x[0])):
                diffs = [i for i in range(3) if ka[i] != kb[i]]
                if diffs != [factor]:
                    continue
                if factor == 0 and not ka[0] < kb[0]:
                    continue
                if factor == 1 and not (ka[1] == "dense" and kb[1] in ("hybrid", "bm25")):
                    continue
                if factor == 2 and not (not ka[2] and kb[2]):
                    continue
                d = paired_bootstrap(per_q(runs[la], args.metric), per_q(runs[lb], args.metric), n_boot, seed)
                out(f"| {FACTOR_NAMES[factor]} | {la} | {lb} | {diff_str(d)} |")

    # ---- 3. finalists (rule fixed in advance)
    candidates = [l for l in order if l != args.baseline]
    finalists = candidates[:N_FINALISTS]
    out(f"\n## Finalists (rule: top {N_FINALISTS} by {args.metric}, ties by hit@5, excluding the baseline)\n")
    for l in finalists:
        done = "full results available" if runs[l]["summary"]["mode"] == "full" else "needs a full run"
        out(f"- {l} ({done})")
    todo = [l for l in finalists if runs[l]["summary"]["mode"] != "full"]
    if todo:
        out(f"\nRun: `python scripts/run_experiments.py --stage full --labels {' '.join(todo)}`")

    # ---- 4. end-to-end vs baseline
    base = runs[args.baseline]
    full = [l for l in order if runs[l]["summary"]["mode"] == "full" and l != args.baseline]
    if base["summary"]["mode"] == "full" and full:
        out(f"\n## End-to-end vs baseline {args.baseline} (B - A, paired bootstrap 95% CI)\n")
        out("| config | correctness | faithfulness | hit@5 | MRR |")
        out("|---|---|---|---|---|")
        for l in full:
            cells = [diff_str(paired_bootstrap(per_q(base, m), per_q(runs[l], m), n_boot, seed))
                     for m in ("correctness", "faithfulness", "hit@5", "mrr")]
            out(f"| {l} | " + " | ".join(cells) + " |")

    (results_dir / "comparison.md").write_text("# Experiment comparison\n\n" + "\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWrote results/comparison.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
