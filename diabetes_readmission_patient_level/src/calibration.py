from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    f1_score, precision_score, recall_score, log_loss,
)

from .models import make_xgb
from .pipeline import build_model_pipeline


class SigmoidProbabilityCalibrator:
    """Platt-style sigmoid calibration."""

    def __init__(self):
        self.model = LogisticRegression(solver="lbfgs", max_iter=1000)

    def fit(self, probabilities, y):
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
        score = np.log(p / (1.0 - p)).reshape(-1, 1)
        self.model.fit(score, np.asarray(y).astype(int))
        return self

    def predict_proba(self, probabilities):
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
        score = np.log(p / (1.0 - p)).reshape(-1, 1)
        return self.model.predict_proba(score)[:, 1]


def fit_final_xgb(X_raw, y, best_params, seed=42):
    spw = (np.asarray(y) == 0).sum() / (np.asarray(y) == 1).sum()
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


def generate_cross_fitted_calibrated_oof(
    X_raw, y, groups, best_params, n_splits=5, seed=42
):
    """Generate patient-level OOF raw probabilities and calibrated probabilities.

    The base probabilities are strictly out-of-fold by patient. The sigmoid
    outcome calibrator is then fitted on the complete development OOF set.
    This is the standard training-only calibration design used for the final
    model; no final-test observation participates in either calibration stage.
    """
    y = pd.Series(y).reset_index(drop=True)
    X_raw = X_raw.reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)
    raw_oof = np.full(len(y), np.nan, dtype=float)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X_raw, y, groups=groups), start=1):
        train_groups = set(groups.iloc[train_idx])
        valid_groups = set(groups.iloc[valid_idx])
        if train_groups.intersection(valid_groups):
            raise RuntimeError(f"Calibration OOF fold {fold} has patient overlap.")

        model = fit_final_xgb(X_raw.iloc[train_idx], y.iloc[train_idx], best_params, seed)
        raw_oof[valid_idx] = model.predict_proba(X_raw.iloc[valid_idx])[:, 1]

    if np.isnan(raw_oof).any():
        raise RuntimeError("OOF prediction generation left missing predictions.")

    # Fit outcome calibration only on patient-level OOF predictions.
    outcome_calibrator = SigmoidProbabilityCalibrator().fit(raw_oof, y)
    calibrated_oof = outcome_calibrator.predict_proba(raw_oof)
    return raw_oof, calibrated_oof


def generate_oof_predictions(X_raw, y, groups, best_params, n_splits=5, seed=42):
    """Backward-compatible wrapper returning patient-level raw OOF predictions."""
    raw_oof, _ = generate_cross_fitted_calibrated_oof(
        X_raw, y, groups, best_params, n_splits=n_splits, seed=seed
    )
    return raw_oof


def fit_training_calibration(X_raw, y, groups, best_params, n_splits=5, seed=42):
    """Fit final model/calibrator and produce leakage-controlled training OOF probabilities."""
    oof_raw, oof_calibrated = generate_cross_fitted_calibrated_oof(
        X_raw, y, groups, best_params, n_splits=n_splits, seed=seed
    )

    # Final outcome calibrator is trained on all development OOF predictions;
    # it is applied to the completely unseen test set only after this stage.
    final_calibrator = SigmoidProbabilityCalibrator().fit(oof_raw, y)
    final_model = fit_final_xgb(X_raw, y, best_params, seed)
    return final_model, final_calibrator, oof_raw, oof_calibrated


def evaluate_calibrated(model, calibrator, X_raw, y, threshold=0.10):
    p_raw = model.predict_proba(X_raw)[:, 1]
    p = calibrator.predict_proba(p_raw)
    return evaluate_calibrated_from_probabilities(y, p, threshold)


def evaluate_calibrated_from_probabilities(y, p, threshold):
    y_arr = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y_arr == 1)).sum())
    tn = int(((pred == 0) & (y_arr == 0)).sum())
    fp = int(((pred == 1) & (y_arr == 0)).sum())
    fn = int(((pred == 0) & (y_arr == 1)).sum())
    return {
        "auc": float(roc_auc_score(y_arr, p)),
        "pr_auc": float(average_precision_score(y_arr, p)),
        "brier": float(brier_score_loss(y_arr, p)),
        "f1": float(f1_score(y_arr, pred, zero_division=0)),
        "sensitivity": float(recall_score(y_arr, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "ppv": float(precision_score(y_arr, pred, zero_division=0)),
        "npv": float(tn / (tn + fn)) if (tn + fn) else float("nan"),
        "threshold": float(threshold),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
