from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss


def _risk_ratio(y, p, threshold=0.10):
    high = p >= threshold
    low = ~high
    # Paper uses high/low risk groups; here low risk is the complement.
    high_rate = y[high].mean() if high.any() else np.nan
    low_rate = y[low].mean() if low.any() else np.nan
    return high_rate / low_rate if low_rate and np.isfinite(low_rate) else np.nan


def bootstrap_metrics(y, p, n_resamples=1000, seed=42, threshold=0.10):
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    n = len(y)

    rows = []
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        ps = p[idx]

        # Rare all-one/all-zero bootstrap samples are skipped for AUC only.
        auc = np.nan
        if len(np.unique(ys)) == 2:
            auc = roc_auc_score(ys, ps)

        rows.append({
            "iteration": i + 1,
            "auc": auc,
            "brier": brier_score_loss(ys, ps),
            "risk_ratio": _risk_ratio(ys, ps, threshold),
        })

    out = pd.DataFrame(rows)
    ci = {}
    for c in ["auc", "brier", "risk_ratio"]:
        vals = out[c].dropna().to_numpy()
        ci[c] = {
            "lower_2.5": float(np.percentile(vals, 2.5)),
            "upper_97.5": float(np.percentile(vals, 97.5)),
            "mean": float(np.mean(vals)),
        }
    return out, ci


def shap_bootstrap_indices(n, n_resamples=100, seed=42):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(n_resamples)]
