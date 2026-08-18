from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    f1_score, confusion_matrix, precision_score, recall_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from .models import make_xgb
from .pipeline import build_model_pipeline


def fit_final_xgb(X_raw, y, best_params, seed=42):
    spw = (y == 0).sum() / (y == 1).sum()
    model = make_xgb(
        spw,
        seed,
        n_estimators=int(best_params["model__n_estimators"]),
        max_depth=int(best_params["model__max_depth"]),
        learning_rate=float(best_params["model__learning_rate"]),
    )
    pipe = build_model_pipeline(X_raw, model)
    pipe.fit(X_raw, y)
    return pipe


def calibrate_full_model(fitted_model, X_raw, y):
    calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(fitted_model),
        method="sigmoid",
        cv=None,
    )
    calibrator.fit(X_raw, y)
    return calibrator


def evaluate_calibrated(model, X_raw, y, threshold=0.10):
    p = model.predict_proba(X_raw)[:, 1]
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "ppv": float(precision_score(y, pred, zero_division=0)),
        "npv": float(tn / (tn + fn)) if (tn + fn) else float("nan"),
        "high_risk_per_1000": float(pred.mean() * 1000),
        "true_readmissions_per_1000": float(((pred == 1) & (y.to_numpy() == 1)).sum() / len(y) * 1000),
    }
