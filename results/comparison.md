# Experiment comparison

## Leaderboard (sorted by mrr; retrieval metrics on 55 answerable questions)

| config | stage | hit@1 | hit@5 | MRR | pool hit@20 | correctness | faithfulness | false abst. | unans. abst. | retrieve s |
|---|---|---|---|---|---|---|---|---|---|---|
| c512-hybrid-rerank | retrieval | 0.618 | 0.909 | 0.732 | 0.982 | - | - | - | - | 4.252 |
| c512-bm25 | retrieval | 0.582 | 0.873 | 0.690 | 0.982 | - | - | - | - | 0.001 |
| c512-dense-rerank | retrieval | 0.564 | 0.855 | 0.678 | 0.891 | - | - | - | - | 4.222 |
| c256-hybrid-rerank | retrieval | 0.545 | 0.855 | 0.664 | 0.927 | - | - | - | - | 1.997 |
| c512-hybrid | retrieval | 0.527 | 0.855 | 0.654 | 0.982 | - | - | - | - | 0.039 |
| c256-dense-rerank | retrieval | 0.527 | 0.782 | 0.632 | 0.873 | - | - | - | - | 2.076 |
| c128-hybrid-rerank | retrieval | 0.527 | 0.800 | 0.631 | 0.927 | - | - | - | - | 0.956 |
| c256-bm25 | retrieval | 0.473 | 0.800 | 0.613 | 0.946 | - | - | - | - | 0.001 |
| c256-hybrid | retrieval | 0.509 | 0.782 | 0.603 | 0.927 | - | - | - | - | 0.042 |
| c128-bm25 | retrieval | 0.509 | 0.764 | 0.602 | 0.909 | - | - | - | - | 0.001 |
| c128-dense-rerank | retrieval | 0.509 | 0.746 | 0.595 | 0.800 | - | - | - | - | 0.995 |
| c128-hybrid | retrieval | 0.491 | 0.746 | 0.586 | 0.927 | - | - | - | - | 0.048 |
| c512-dense (baseline) | full | 0.436 | 0.691 | 0.527 | 0.891 | 0.855 | 0.951 | 0.073 | 1.000 | 0.046 |
| c128-dense | retrieval | 0.418 | 0.673 | 0.510 | 0.800 | - | - | - | - | 0.046 |
| c256-dense | retrieval | 0.382 | 0.636 | 0.488 | 0.873 | - | - | - | - | 0.041 |

## One-factor effects on mrr (B - A, paired bootstrap 95% CI; * = CI excludes 0)

| factor | A | B | difference |
|---|---|---|---|
| reranker | c128-dense | c128-dense-rerank | +0.086 [-0.009, +0.182]  P(B>A)=0.96 |
| reranker | c128-hybrid | c128-hybrid-rerank | +0.045 [-0.066, +0.155]  P(B>A)=0.78 |
| reranker | c256-dense | c256-dense-rerank | +0.144 [+0.035, +0.255]* P(B>A)=1.00 |
| reranker | c256-hybrid | c256-hybrid-rerank | +0.061 [-0.034, +0.158]  P(B>A)=0.89 |
| reranker | c512-dense | c512-dense-rerank | +0.151 [+0.045, +0.257]* P(B>A)=1.00 |
| reranker | c512-hybrid | c512-hybrid-rerank | +0.078 [-0.028, +0.185]  P(B>A)=0.92 |
| retrieval mode | c128-dense | c128-bm25 | +0.093 [-0.024, +0.206]  P(B>A)=0.94 |
| retrieval mode | c128-dense | c128-hybrid | +0.076 [+0.010, +0.147]* P(B>A)=0.99 |
| retrieval mode | c128-dense-rerank | c128-hybrid-rerank | +0.036 [-0.025, +0.105]  P(B>A)=0.86 |
| retrieval mode | c256-dense | c256-bm25 | +0.125 [+0.023, +0.230]* P(B>A)=0.99 |
| retrieval mode | c256-dense | c256-hybrid | +0.115 [+0.045, +0.190]* P(B>A)=1.00 |
| retrieval mode | c256-dense-rerank | c256-hybrid-rerank | +0.032 [-0.003, +0.082]  P(B>A)=0.95 |
| retrieval mode | c512-dense | c512-bm25 | +0.163 [+0.038, +0.276]* P(B>A)=1.00 |
| retrieval mode | c512-dense | c512-hybrid | +0.127 [+0.046, +0.211]* P(B>A)=1.00 |
| retrieval mode | c512-dense-rerank | c512-hybrid-rerank | +0.054 [+0.000, +0.127]  P(B>A)=0.97 |
| chunk size | c128-bm25 | c256-bm25 | +0.010 [-0.051, +0.074]  P(B>A)=0.62 |
| chunk size | c128-bm25 | c512-bm25 | +0.088 [-0.015, +0.196]  P(B>A)=0.95 |
| chunk size | c128-dense | c256-dense | -0.022 [-0.146, +0.105]  P(B>A)=0.38 |
| chunk size | c128-dense | c512-dense | +0.017 [-0.117, +0.147]  P(B>A)=0.61 |
| chunk size | c128-dense-rerank | c256-dense-rerank | +0.037 [-0.065, +0.137]  P(B>A)=0.75 |
| chunk size | c128-dense-rerank | c512-dense-rerank | +0.082 [-0.067, +0.224]  P(B>A)=0.88 |
| chunk size | c128-hybrid | c256-hybrid | +0.017 [-0.092, +0.121]  P(B>A)=0.62 |
| chunk size | c128-hybrid | c512-hybrid | +0.068 [-0.077, +0.212]  P(B>A)=0.83 |
| chunk size | c128-hybrid-rerank | c256-hybrid-rerank | +0.033 [-0.054, +0.124]  P(B>A)=0.78 |
| chunk size | c128-hybrid-rerank | c512-hybrid-rerank | +0.101 [-0.027, +0.237]  P(B>A)=0.94 |
| chunk size | c256-bm25 | c512-bm25 | +0.077 [-0.021, +0.176]  P(B>A)=0.94 |
| chunk size | c256-dense | c512-dense | +0.039 [-0.104, +0.178]  P(B>A)=0.71 |
| chunk size | c256-dense-rerank | c512-dense-rerank | +0.046 [-0.086, +0.170]  P(B>A)=0.75 |
| chunk size | c256-hybrid | c512-hybrid | +0.051 [-0.075, +0.181]  P(B>A)=0.79 |
| chunk size | c256-hybrid-rerank | c512-hybrid-rerank | +0.069 [-0.038, +0.178]  P(B>A)=0.88 |

## Finalists (rule: top 3 by mrr, ties by hit@5, excluding the baseline)

- c512-hybrid-rerank (needs a full run)
- c512-bm25 (needs a full run)
- c512-dense-rerank (needs a full run)

Run: `python scripts/run_experiments.py --stage full --labels c512-hybrid-rerank c512-bm25 c512-dense-rerank`
