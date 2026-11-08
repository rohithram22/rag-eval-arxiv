# Results: which RAG configuration works best, and why

**Question.** For question answering over a fixed corpus of 150 arXiv papers on retrieval-augmented
generation, which retrieval configuration produces the best answers: dense, keyword (BM25) or hybrid
search; with or without a cross-encoder reranker; and at which chunk size?

**Answer.** Retrieval was the bottleneck, and fixing it fixed the answers. Two configurations
**significantly** beat the dense-retrieval baseline on answer correctness:

| Configuration (512-token chunks) | Answer correctness | Δ vs baseline (paired 95% CI) | Retrieval latency |
|---|---|---|---|
| Dense retrieval (baseline) | 0.855 | — | 0.05 s |
| **Hybrid (BM25 + dense, RRF) + reranker** | **0.964** | **+0.109 [+0.027, +0.209]** ✔ | 4.1 s |
| **BM25 alone** | **0.982** | **+0.127 [+0.036, +0.227]** ✔ | **0.001 s** |
| Dense + reranker | 0.927 | +0.073 [+0.000, +0.154] (borderline) | 4.1 s |

✔ = the paired 95% confidence interval excludes zero. All four configurations correctly declined
**8/8** unanswerable questions, so the gains do not come from answering more recklessly.

The two winners are **statistically indistinguishable** from each other (correctness −0.018 [−0.073,
+0.027], MRR +0.042 [−0.070, +0.158]). BM25 gets there at roughly 4,000× lower retrieval latency;
hybrid + reranker ranks the evidence slightly higher on average and is the safer choice for questions
that paraphrase rather than name the method (see [Limitations](#limitations)).

---

## 1. Setup

| Component | Choice |
|---|---|
| Corpus | 150 arXiv papers (cs.CL / cs.IR, 2020–2025; 12 foundational "seed" papers + 138 relevance-ranked RAG papers), 2,030 pages, downloaded once and frozen in `data/manifest.jsonl` (versioned IDs + SHA-256) |
| Parsing | PyMuPDF; ligature/hyphenation repair, header/footer removal; reference lists removed in 145/150 papers |
| Chunking | Custom sentence-packing chunker, token budget counted with the embedding model's tokenizer; overlap ≈ 1/8 of chunk size; paper title prepended to every chunk |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-d), exact FAISS inner-product search |
| Sparse | BM25 (`bm25s`, Lucene variant, k1 = 1.5, b = 0.75, stopwords removed, Snowball stemming) |
| Hybrid | Reciprocal rank fusion of the top 20 from each retriever, k = 60 (standard default, not tuned) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, reranks 20 candidates down to 5 |
| Generator | `openai/gpt-oss-20b` (Groq, reasoning effort low, temperature 0), top 5 chunks, versioned grounded prompt that requires citations and an exact abstention sentence |
| Judge | `gemini-3.5-flash-lite` (a different model family from the generator, to avoid self-preference bias) |
| Hardware / cost | GitHub Codespaces, 2 CPU cores, no GPU; free API tiers only; total cost $0 |

Chunk statistics:

| Chunk size | Chunks | Median tokens | Consecutive chunks sharing text |
|---|---|---|---|
| 128 | 22,227 | 107 | 72% |
| 256 | 8,526 | 233 | 88% |
| 512 | 3,873 | 487 | 93% |

## 2. The golden evaluation set

- **63 questions: 55 answerable + 8 unanswerable**, covering 54 distinct papers (at most one LLM-drafted question per paper).
- Types: method 13, factual 13, why 12, definition 11, numeric 6, unanswerable 8.
- **Construction:** `gpt-oss-120b` drafted 80 valid candidates from single passages (103 calls). 23 more were rejected automatically: 8 passages the model judged unsuitable, and **15 whose "verbatim" evidence quote did not actually occur in the passage**. Every candidate was then reviewed by hand against explicit criteria (self-contained, supported by its evidence, realistic, not general knowledge): 29 rejected, several edited, 4 answerable questions and 8 unanswerable ones written by hand.
- **Ground truth is anchored to the paper text, not to chunks.** Each item stores its evidence quote plus character offsets into the parsed paper, so the same golden set scores every chunk size.
- Mean word overlap between question and evidence: 0.38. Questions paraphrase rather than copy the paper, except for names of methods, datasets and metrics.

## 3. Metrics

**Retrieval** (answerable questions, top 5 chunks). A chunk is *relevant* if it is from the right paper
and `overlap / min(len(evidence), len(chunk)) ≥ 0.5`. Dividing by the smaller span keeps the rule fair
across chunk sizes.

- **hit@k:** is a relevant chunk in the top k?
- **MRR:** 1 / rank of the first relevant chunk.
- **paper hit:** is any chunk from the right paper retrieved?
- **pool hit@20:** is a relevant chunk among the 20 candidates *before* the cut to 5 or reranking? This is the reranker's ceiling.

**Generation**, judged once per answer by the LLM judge:

- **Correctness vs the reference answer** (correct 1 / partial 0.5 / incorrect 0). The headline metric. An abstention on an answerable question counts as 0.
- **Faithfulness:** the answer is split into atomic claims, and each is checked against the retrieved context (RAGAS-style claim decomposition).
- **Answer relevance** (1–5).
- **False-abstention rate** on answerable questions, and **abstention rate** on unanswerable ones.

**Statistics.** 95% bootstrap confidence intervals (2,000 resamples). Configurations are compared with
a **paired** bootstrap over per-question differences, since every configuration answers the same
questions.

**Judge reliability.** On 10 randomly sampled judged answers, the judge's correctness verdict was
confirmed on **10/10** and its claim-level verdicts on **9/10** (cross-checked with a second model,
Claude Sonnet, and reviewed by me). The one disagreement was a supported claim the judge marked
*unsupported*, so faithfulness errs, if anything, on the strict side.

## 4. Experimental protocol

1. **Baseline:** dense retrieval at 512-token chunks, evaluated end to end.
2. **Screen:** 15 configurations (chunk size {128, 256, 512} × {dense, hybrid} × {reranker off, on}, plus BM25 alone at each size), evaluated on retrieval metrics only. This is free and deterministic.
3. **Finalists:** the top 3 by MRR, a rule **fixed before the screen ran**, evaluated end to end.
4. **Effects:** every pair of configurations that differ in exactly one setting is compared with the paired bootstrap.

Every LLM call is cached on disk by a hash of model, settings and prompt, so results are exactly
reproducible. Re-running the baseline reproduced every number with zero API calls.

## 5. Results

### 5.1 Baseline: retrieval is the bottleneck

| hit@1 | hit@5 | MRR | paper hit | pool hit@20 | correctness | faithfulness | false abst. | unanswerable abst. |
|---|---|---|---|---|---|---|---|---|
| 0.436 | 0.691 | 0.527 | 0.964 | 0.891 | 0.855 [0.76, 0.94] | 0.951 | 4/55 | 8/8 |

Cross-tabulating the two stages:

| | Correct | Partial | Incorrect | Abstained |
|---|---|---|---|---|
| Evidence retrieved (38) | **38** | 0 | 0 | 0 |
| Evidence missed (17) | 8 | 2 | 3 | 4 |

**Whenever retrieval found the evidence, the answer was correct (38/38). All 9 errors came from
retrieval misses.** Retrieval almost always found the right *paper* (0.964) but often the wrong
*passage*, so the experiments focus on retrieval. The 4 false abstentions all happened on retrieval
misses: the generator declined rather than guessed, as the prompt requires.

### 5.2 Retrieval screen (15 configurations)

| Configuration | hit@1 | hit@5 | MRR | pool hit@20 | retrieve (s) |
|---|---|---|---|---|---|
| **c512-hybrid-rerank** | **0.618** | **0.909** | **0.732** | 0.982 | 4.10 |
| c512-bm25 | 0.582 | 0.873 | 0.690 | 0.982 | 0.001 |
| c512-dense-rerank | 0.564 | 0.855 | 0.678 | 0.891 | 4.14 |
| c256-hybrid-rerank | 0.545 | 0.855 | 0.664 | 0.927 | 2.00 |
| c512-hybrid | 0.527 | 0.855 | 0.654 | 0.982 | 0.04 |
| c256-dense-rerank | 0.527 | 0.782 | 0.632 | 0.873 | 2.08 |
| c128-hybrid-rerank | 0.527 | 0.800 | 0.631 | 0.927 | 0.96 |
| c256-bm25 | 0.473 | 0.800 | 0.613 | 0.946 | 0.001 |
| c256-hybrid | 0.509 | 0.782 | 0.603 | 0.927 | 0.04 |
| c128-bm25 | 0.509 | 0.764 | 0.602 | 0.909 | 0.001 |
| c128-dense-rerank | 0.509 | 0.746 | 0.595 | 0.800 | 1.00 |
| c128-hybrid | 0.491 | 0.746 | 0.586 | 0.927 | 0.05 |
| c512-dense (baseline) | 0.436 | 0.691 | 0.527 | 0.891 | 0.05 |
| c128-dense | 0.418 | 0.673 | 0.510 | 0.800 | 0.05 |
| c256-dense | 0.382 | 0.636 | 0.488 | 0.873 | 0.04 |

### 5.3 One-factor effects on MRR (paired bootstrap)

| Change | 128 | 256 | 512 |
|---|---|---|---|
| Dense → **hybrid** (no reranker) | **+0.076*** | **+0.115*** | **+0.127*** |
| Dense → **BM25** | +0.093 | **+0.125*** | **+0.163*** |
| **Add reranker** to dense | +0.086 | **+0.144*** | **+0.151*** |
| **Add reranker** to hybrid | +0.045 | +0.061 | +0.078 |
| Dense → hybrid (both with reranker) | +0.036 | +0.032 | +0.054 |

\* = 95% CI excludes 0. Chunk size: **no comparison was significant** in any family, although most
point estimates favor 512 (e.g. hybrid + reranker 128 → 512: +0.101 [−0.027, +0.237]). The full table
of 30 comparisons is in [`results/comparison.md`](results/comparison.md).

### 5.4 End to end (finalists vs baseline)

| Configuration | Correctness | Faithfulness | False abst. | Unans. abst. | Δ correctness | Δ hit@5 | Δ MRR |
|---|---|---|---|---|---|---|---|
| Dense (baseline) | 0.855 | 0.951 | 0.073 | 1.000 | — | — | — |
| Hybrid + reranker | 0.964 | 0.981 | 0.018 | 1.000 | **+0.109*** | **+0.218*** | **+0.205*** |
| BM25 | 0.982 | 0.969 | 0.000 | 1.000 | **+0.127*** | **+0.182*** | **+0.163*** |
| Dense + reranker | 0.927 | 0.976 | 0.054 | 1.000 | +0.073 | **+0.164*** | **+0.151*** |

Faithfulness did not change significantly for any finalist; it was already 0.95–0.98. The gains come
from retrieving the right passages, not from the generator inventing less.

## 6. What won, and why

**1. Hybrid search fixes the first stage.** Without a reranker, hybrid beats dense significantly at
every chunk size. Its candidate pool contains the evidence for 0.982 of questions at 512, against
0.891 for dense: the evidence is missing from the top 20 for only 1 question instead of 6. Dense embeddings
blur rare exact terms (method names, dataset names, metric names), and BM25 matches them.

**2. The reranker fixes ordering, but only within its pool.** On dense candidates, the reranker raises
MRR significantly (+0.15 at 512), but end-to-end correctness improves only borderline (+0.073),
because its pool is still dense's: pool hit@20 stays at 0.891, and the questions dense never surfaces
remain abstentions. On hybrid candidates, it adds less (+0.05 to +0.08, not significant):
**diminishing returns**, because hybrid already fixed many of the misses the reranker would have
fixed.

**3. BM25 alone is surprisingly strong**, which contradicted the hypothesis that it would trail dense.
Two reasons: the corpus is dense with distinctive technical terms, and the golden questions were
required to **name** their method or dataset to avoid ambiguity across 150 similar papers. That is
exactly the kind of query keyword search handles best (see Limitations).

**4. Chunk size did not matter detectably.** With 55 questions, differences of 5–10 MRR points are
inside the noise. The defensible statement is "no measurable effect," not "512 is best."

**5. The cost of quality.** The reranker adds ~4 s per query on a 2-core CPU (100× the cost of
retrieval itself). For this question set, BM25 matches the best configuration at a fraction of the
cost.

**Recommendation.** For this corpus and question style, **BM25** gives the best quality per unit of
cost. **Hybrid + reranker** is the more robust choice when users may describe a method without naming
it, because it still has dense retrieval to fall back on, at the cost of latency (a GPU would remove
most of it). The demo uses hybrid + reranker.

## 7. Worked examples

**c011: "What integration does the Ragas framework provide…?"** One question that shows the whole
story:

| Configuration | Rank of evidence | Why |
|---|---|---|
| Dense | miss (not in top 20) | The rare name "Ragas" is blurred by the embedding |
| BM25 | 1 | Exact matches on "Ragas", "llama-index", "Langchain" |
| Hybrid | 2 | BM25 brings it into the fused list |
| Dense + reranker | **still a miss** | The reranker cannot promote what the first stage never retrieved |
| Hybrid + reranker | 1 | Hybrid puts it in the pool (rank 2); the reranker moves it to 1 |

**m00: "What three properties does RAGAS score…?"** Missed by every configuration, although the
evidence is findable at every chunk size (a chunk with relevance ≥ 0.5 exists at 128, 256 and 512).
The definitions paragraph never says "Ragas"; the name appears only in the title header shared by all
RAGAS chunks. Meanwhile several *citing* papers use "Ragas" in their text next to "evaluation" and
"metrics", so they outrank it. The question also paraphrases the paper ("properties" vs "quality
aspects"). This is classic **vocabulary mismatch with hard negatives**. Hybrid + reranker still answered
it correctly from other retrieved passages, most likely citing papers that describe the metrics accurately.

**c021: a table-parsing failure.** In the baseline, the answer swapped columns of a results table
(faithfulness 0.00). PDF tables are flattened to lines of numbers during parsing, which loses which
number belongs to which column. A parsing limitation turned into a confident wrong answer. With hybrid
+ reranker, the right passage was retrieved at rank 1 and the answer was correct.

**c004: correct answer despite a retrieval "miss."** The answer to the HyDE question appears in
several places in the paper. Retrieval found a different passage than the labeled evidence, and the
answer was still correct. 8 of the baseline's 17 misses were like this.

**Remaining errors of hybrid + reranker (3 of 55).**
- **c001 (partial):** BEIR / Touché-2020; evidence not retrieved.
- **c010 (partial):** RETRO; evidence retrieved at rank 1. Possibly judge strictness: in the audit, the judge marked a supported claim on this question as unsupported.
- **c017 (abstained):** a numeric TrustMus result; evidence not retrieved. BM25 found it at rank 1.

## 8. Limitations

- **Small evaluation set.** 55 answerable questions give 95% intervals of roughly ±10–13 points on single metrics. Paired comparisons are tighter, but effects below ~5 points (including any chunk-size effect) cannot be detected. With ~30 one-factor comparisons, one or two "significant" results could be chance; the conclusions above rest on consistent patterns, not single cells.
- **The question style favors keyword search.** Questions were required to name their method, model or dataset so that they are unambiguous across 150 similar papers, and every chunk carries its paper title. This likely flatters BM25 relative to real users who describe a method without naming it.
- **Single-evidence ground truth.** Each question has one labeled evidence passage, but answers often appear in several places (abstract, introduction, method). Hit rate therefore underestimates retrieval quality: in the baseline, 8 of 17 "misses" were still answered correctly. The end-to-end correctness metric does not have this problem.
- **Passage-local questions.** Questions were drafted from single passages, so the set has no multi-hop questions that need information from several papers.
- **Few numeric questions** (6 of 55); numbers often live in tables, which parsing flattens (c021). Possible question ambiguity in c064 ("improvement opportunities" vs. limitations) was noted but **not** edited after seeing results, to avoid biasing the evaluation.
- **LLM judge.** One judgment per answer from one judge model; spot-checked on only 10 answers.
- **Fixed components.** One embedding model, one generator, top k = 5, RRF k = 60 (not tuned: there is no separate development set, and tuning on the test set would overfit).
- **Reranker truncation.** The cross-encoder reads at most 512 tokens of query + chunk, so the ends of 512-token chunks are cut before reranking. It did not visibly hurt, but it was not isolated.
- **Parsing.** Tables flattened; reference lists remain in 5 of 150 papers; the sentence splitter occasionally breaks at abbreviations such as "et al."
- **Free-tier constraints.** The generator's 200,000-tokens-per-day limit allows about one full evaluation at 512-token chunks per day, so only the 3 finalists (all at 512, by the pre-set rule) were evaluated end to end. Chunk size was therefore compared on retrieval metrics only. Latencies are from a 2-core CPU; a GPU would make the reranker far faster.

## 9. Future work

- **Query rewriting (e.g. HyDE)** to address vocabulary mismatch like m00: search with a hypothetical answer that uses the paper's own terms.
- **Section headings in chunk headers**, to distinguish chunks within one paper.
- **A paraphrased version of the golden set**, with method names removed, to measure how much of BM25's advantage comes from question style.
- **More questions and multi-hop questions**, to detect smaller effects such as chunk size.
- **Weighted fusion tuned on a separate development set**, and table-aware PDF parsing.

## 10. Reproducing these results

```bash
bash scripts/setup.sh && source .venv/bin/activate
python scripts/download_corpus.py         # frozen mode: same 150 papers, checksums verified
python scripts/parse_corpus.py
python scripts/build_index.py --chunk-size 128   # and 256, 512
python scripts/run_experiments.py --stage screen
python scripts/run_experiments.py --stage full --labels c512-hybrid-rerank c512-bm25 c512-dense-rerank
python scripts/evaluate.py                # baseline, end to end
python scripts/compare.py                 # writes results/comparison.md
```

Per-question results for every configuration are in `results/<config>/per_question.jsonl`; the golden
set and its full review history are in `eval/`.