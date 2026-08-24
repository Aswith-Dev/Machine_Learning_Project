from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def decision_curve(y, p, thresholds=np.arange(0.01, 0.301, 0.01)):
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)

    rows = []
    prevalence = y.mean()

    for t in thresholds:
        pred = p >= t
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))

        model_nb = tp / n - (fp / n) * (t / (1 - t))
        treat_all = prevalence - (1 - prevalence) * (t / (1 - t))
        treat_none = 0.0

        rows.append({
            "threshold": t,
            "model_net_benefit": model_nb,
            "treat_all_net_benefit": treat_all,
            "treat_none_net_benefit": treat_none,
        })

    return pd.DataFrame(rows)


def clinical_impact(y, p, thresholds=np.arange(0.01, 0.301, 0.01)):
    y = np.asarray(y)
    p = np.asarray(p)
    rows = []

    for t in thresholds:
        high = p >= t
        high_per_1000 = high.mean() * 1000
        true_readmissions_per_1000 = y[high].sum() / len(y) * 1000
        rows.append({
            "threshold": t,
            "predicted_high_risk_per_1000": high_per_1000,
            "true_readmissions_per_1000": true_readmissions_per_1000,
        })

    return pd.DataFrame(rows)


def threshold_metrics(y, p, thresholds):
    rows = []
    y = np.asarray(y)
    p = np.asarray(p)

    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()

        rows.append({
            "threshold": t,
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "ppv": tp / (tp + fp) if tp + fp else np.nan,
            "npv": tn / (tn + fn) if tn + fn else np.nan,
            "high_risk_per_1000": pred.mean() * 1000,
        })
    return pd.DataFrame(rows)


def risk_stratification(y, p):
    q1, q2 = np.quantile(p, [1/3, 2/3])
    group = np.where(p <= q1, "Low", np.where(p <= q2, "Medium", "High"))
    d = pd.DataFrame({"risk_group": group, "y": y})
    return d.groupby("risk_group", sort=False).agg(
        patients=("y", "size"),
        readmissions=("y", "sum"),
        readmission_rate=("y", "mean"),
    ).reset_index()
