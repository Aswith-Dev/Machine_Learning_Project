from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score


class CorrectnessCalibrator:
    """Second-stage Platt-style calibration of P(prediction is correct).

    Input is the model confidence: the probability assigned to the class that
    the base model actually predicts. Target is 1 when that prediction matches
    the observed outcome and 0 otherwise.
    """

    def __init__(self):
        self.model = LogisticRegression(solver="lbfgs", max_iter=1000)

    def fit(self, confidence, correct):
        c = np.clip(np.asarray(confidence, dtype=float), 1e-7, 1 - 1e-7)
        score = np.log(c / (1.0 - c)).reshape(-1, 1)
        self.model.fit(score, np.asarray(correct).astype(int))
        return self

    def predict_proba(self, confidence):
        c = np.clip(np.asarray(confidence, dtype=float), 1e-7, 1 - 1e-7)
        score = np.log(c / (1.0 - c)).reshape(-1, 1)
        return self.model.predict_proba(score)[:, 1]


def prediction_confidence(probability, predicted_class):
    p = np.asarray(probability, dtype=float)
    pred = np.asarray(predicted_class).astype(int)
    return np.where(pred == 1, p, 1.0 - p)


def correctness_targets(y_true, predicted_class):
    return (np.asarray(y_true).astype(int) == np.asarray(predicted_class).astype(int)).astype(int)


def expected_calibration_error(confidence, correct, n_bins=10):
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct).astype(int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    n = len(confidence)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence >= lo) & ((confidence < hi) if i < n_bins - 1 else (confidence <= hi))
        if not mask.any():
            continue
        avg_conf = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        count = int(mask.sum())
        ece += (count / n) * abs(avg_conf - accuracy)
        rows.append({
            "bin_lower": lo,
            "bin_upper": hi,
            "count": count,
            "mean_confidence": avg_conf,
            "observed_accuracy": accuracy,
            "absolute_gap": abs(avg_conf - accuracy),
        })
    return float(ece), pd.DataFrame(rows)


def reliability_metrics(predicted_correctness, actual_correctness, n_bins=10):
    p = np.clip(np.asarray(predicted_correctness, dtype=float), 1e-7, 1 - 1e-7)
    y = np.asarray(actual_correctness).astype(int)
    ece, bins = expected_calibration_error(p, y, n_bins=n_bins)
    return {
        "correctness_brier": float(brier_score_loss(y, p)),
        "correctness_log_loss": float(log_loss(y, p, labels=[0, 1])),
        "correctness_ece": ece,
        "actual_accuracy": float(y.mean()),
        "mean_correctness_confidence": float(p.mean()),
    }, bins


def selective_prediction_table(correctness_confidence, actual_correctness, thresholds=None):
    confidence = np.asarray(correctness_confidence, dtype=float)
    correct = np.asarray(actual_correctness).astype(int)
    if thresholds is None:
        thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    rows = []
    for t in thresholds:
        selected = confidence >= t
        n = int(selected.sum())
        coverage = float(selected.mean())
        accuracy = float(correct[selected].mean()) if n else float("nan")
        rows.append({
            "confidence_threshold": float(t),
            "selected_predictions": n,
            "coverage": coverage,
            "accuracy": accuracy,
            "error_rate": float(1.0 - accuracy) if n else float("nan"),
        })
    return pd.DataFrame(rows)


def build_correctness_dataset(y_true, calibrated_probability, threshold):
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(calibrated_probability, dtype=float)
    pred = (p >= threshold).astype(int)
    conf = prediction_confidence(p, pred)
    correct = correctness_targets(y_true, pred)
    return pd.DataFrame({
        "y_true": y_true,
        "calibrated_probability": p,
        "predicted_class": pred,
        "base_confidence": conf,
        "prediction_correct": correct,
    })


def fit_correctness_calibrator(y_true, calibrated_probability, threshold):
    d = build_correctness_dataset(y_true, calibrated_probability, threshold)
    calibrator = CorrectnessCalibrator().fit(d["base_confidence"], d["prediction_correct"])
    d["correctness_confidence"] = calibrator.predict_proba(d["base_confidence"])
    d["uncertainty"] = 1.0 - d["correctness_confidence"]
    return calibrator, d
