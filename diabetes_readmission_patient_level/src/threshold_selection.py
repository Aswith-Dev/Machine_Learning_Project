from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def threshold_metrics(y, p, thresholds):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    rows = []

    for threshold in thresholds:
        pred = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        precision = precision_score(y, pred, zero_division=0)
        recall = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) else np.nan

        rows.append({
            "threshold": float(threshold),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        })

    return pd.DataFrame(rows)


def select_best_threshold(y, p, min_threshold=0.01, max_threshold=0.30, step=0.01):
    """
    Select the operating threshold using TRAINING-ONLY out-of-fold predictions.

    Default objective: maximize minority-class F1. Ties are resolved by choosing
    the threshold closest to the paper's 10% operating point, then the larger
    threshold for deterministic behavior.
    """
    thresholds = np.round(
        np.arange(min_threshold, max_threshold + step / 2, step), 6
    )
    metrics = threshold_metrics(y, p, thresholds)
    best_f1 = metrics["f1"].max()
    candidates = metrics[np.isclose(metrics["f1"], best_f1)]
    candidates = candidates.assign(
        distance_to_paper_threshold=(candidates["threshold"] - 0.10).abs()
    )
    best = candidates.sort_values(
        ["distance_to_paper_threshold", "threshold"], ascending=[True, False]
    ).iloc[0]
    return float(best["threshold"]), metrics, best.to_dict()
