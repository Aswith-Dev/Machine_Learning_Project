import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.correctness_calibration import (
    CorrectnessCalibrator,
    correctness_targets,
    prediction_confidence,
    reliability_metrics,
)
from src.data_split import create_train_test_split


def test_prediction_confidence_is_probability_of_selected_class():
    p = np.array([0.9, 0.2, 0.05, 0.7])
    pred = np.array([1, 1, 0, 0])
    conf = prediction_confidence(p, pred)
    np.testing.assert_allclose(conf, [0.9, 0.2, 0.95, 0.3])


def test_correctness_target():
    y = np.array([1, 0, 1, 0])
    pred = np.array([1, 1, 1, 0])
    np.testing.assert_array_equal(correctness_targets(y, pred), [1, 0, 1, 1])


def test_correctness_calibrator_outputs_probability():
    confidence = np.linspace(0.05, 0.95, 100)
    correct = (confidence > 0.55).astype(int)
    calibrator = CorrectnessCalibrator().fit(confidence, correct)
    out = calibrator.predict_proba(confidence)
    assert np.all((out >= 0) & (out <= 1))


def test_reliability_metrics():
    confidence = np.array([0.9, 0.8, 0.2, 0.1])
    correct = np.array([1, 1, 0, 0])
    metrics, bins = reliability_metrics(confidence, correct)
    assert metrics["correctness_brier"] >= 0
    assert 0 <= metrics["correctness_ece"] <= 1
    assert not bins.empty
