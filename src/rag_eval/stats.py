"""Bootstrap confidence intervals.

With ~55 questions, each one moves a rate by ~2 points, so point estimates alone
are misleading. The bootstrap resamples questions with replacement many times and
reads the 2.5th / 97.5th percentiles of the resampled means.

paired_bootstrap() compares two configurations on the SAME questions: it resamples
per-question DIFFERENCES, which cancels out question difficulty and gives much
tighter intervals than comparing two separate CIs (used in Step 7).
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_samples: int = 2000, seed: int = 42, alpha: float = 0.05) -> dict:
    vals = np.array([v for v in values if v is not None], dtype=float)
    if len(vals) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, len(vals), size=(n_samples, len(vals)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": round(float(vals.mean()), 4), "lo": round(float(lo), 4),
            "hi": round(float(hi), 4), "n": int(len(vals))}


def paired_bootstrap(a: dict, b: dict, n_samples: int = 2000, seed: int = 42, alpha: float = 0.05) -> dict:
    """a, b: {question_id: value}. Difference = b - a over questions present (non-None) in both."""
    ids = sorted(k for k in a if k in b and a[k] is not None and b[k] is not None)
    if not ids:
        return {"diff": None, "lo": None, "hi": None, "p_b_better": None, "n": 0}
    d = np.array([b[k] - a[k] for k in ids], dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_samples, len(d)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"diff": round(float(d.mean()), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "p_b_better": round(float((means > 0).mean()), 3), "n": len(ids)}
