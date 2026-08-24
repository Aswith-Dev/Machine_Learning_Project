from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

from .calibration import SigmoidProbabilityCalibrator
from .correctness_calibration import (
    CorrectnessCalibrator, prediction_confidence, correctness_targets,
    reliability_metrics, selective_prediction_table
)
from .features import prepare_target
from .explainability import _get_transformed_data, _tree_shap_values

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config" / "config.json"


def load_cfg():
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def evaluate_test():
    cfg = load_cfg()
    test_path = ROOT / cfg["paths"]["test_raw"]
    model_path = ROOT / "results" / "xgb_final_model.joblib"
    calibrator_path = ROOT / "results" / "sigmoid_calibrator.joblib"
    threshold_path = ROOT / "results" / "tables" / "final_threshold.csv"
    correctness_calibrator_path = ROOT / "results" / "correctness_calibrator.joblib"

    for path in [test_path, model_path, calibrator_path, threshold_path, correctness_calibrator_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found: {path}. Run the training workflow first."
            )

    test_df = pd.read_csv(test_path)
    y = prepare_target(test_df).to_numpy()
    record_ids = test_df["original_row_id"].to_numpy()
    X = test_df.drop(columns=[c for c in ["readmitted", "original_row_id"] if c in test_df.columns])

    model = joblib.load(model_path)
    calibrator: SigmoidProbabilityCalibrator = joblib.load(calibrator_path)
    correctness_calibrator: CorrectnessCalibrator = joblib.load(correctness_calibrator_path)
    threshold = float(pd.read_csv(threshold_path).iloc[0]["selected_threshold"])

    raw_p = model.predict_proba(X)[:, 1]
    p = calibrator.predict_proba(raw_p)
    pred = (p >= threshold).astype(int)
    confidence = prediction_confidence(p, pred)
    prediction_correct = correctness_targets(y, pred)
    correctness_confidence = correctness_calibrator.predict_proba(confidence)
    uncertainty = 1.0 - correctness_confidence

    # Per-record predictions. Confidence is the probability assigned to the
    # predicted class: p(readmission) for class 1, 1-p for class 0.
    predictions = pd.DataFrame({
        "original_row_id": record_ids,
        "y_true": y,
        "actual_label": np.where(y == 1, "readmitted_within_30_days", "not_readmitted_within_30_days"),
        "predicted_probability": p,
        "predicted_class": pred,
        "predicted_label": np.where(pred == 1, "readmitted_within_30_days", "not_readmitted_within_30_days"),
        "confidence": confidence,
        "correctness_confidence": correctness_confidence,
        "uncertainty": uncertainty,
        "prediction_correct": prediction_correct,
        "threshold": threshold,
    })

    out_tables = ROOT / "results" / "tables"
    out_figures = ROOT / "results" / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(out_tables / "test_predictions.csv", index=False)

    # Evaluate the second-stage correctness calibration only on the untouched test patients.
    correctness_metrics, correctness_bins = reliability_metrics(
        correctness_confidence, prediction_correct, n_bins=10
    )
    correctness_metrics["n_test"] = int(len(y))
    correctness_bins.to_csv(out_tables / "test_correctness_reliability_bins.csv", index=False)
    pd.DataFrame([correctness_metrics]).to_csv(
        out_tables / "test_correctness_calibration_metrics.csv", index=False
    )
    selective_prediction_table(
        correctness_confidence, prediction_correct
    ).to_csv(out_tables / "test_selective_prediction.csv", index=False)

    # Reliability diagram: predicted correctness confidence vs observed accuracy.
    if not correctness_bins.empty:
        plt.figure()
        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
        plt.plot(
            correctness_bins["mean_confidence"],
            correctness_bins["observed_accuracy"],
            marker="o",
            label="Correctness calibration",
        )
        plt.xlabel("Predicted probability that prediction is correct")
        plt.ylabel("Observed prediction accuracy")
        plt.title("Individual Prediction Correctness Calibration")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_figures / "test_correctness_reliability.png", dpi=300, bbox_inches="tight")
        plt.close()

    cm = confusion_matrix(y, pred, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_csv(out_tables / "test_confusion_matrix.csv")

    report = classification_report(
        y,
        pred,
        labels=[0, 1],
        target_names=["not_readmitted_within_30_days", "readmitted_within_30_days"],
        output_dict=True,
        zero_division=0,
    )
    (out_tables / "test_classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(report).T.to_csv(out_tables / "test_classification_report.csv")

    metrics = {
        "n_test": int(len(y)),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "threshold": threshold,
        "mean_correctness_confidence": float(correctness_confidence.mean()),
        "correctness_brier": float(correctness_metrics["correctness_brier"]),
        "correctness_log_loss": float(correctness_metrics["correctness_log_loss"]),
        "correctness_ece": float(correctness_metrics["correctness_ece"]),
        "prediction_accuracy": float(prediction_correct.mean()),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "specificity": float(cm[0, 0] / (cm[0, 0] + cm[0, 1])) if cm[0].sum() else float("nan"),
        "npv": float(cm[0, 0] / (cm[0, 0] + cm[1, 0])) if (cm[0, 0] + cm[1, 0]) else float("nan"),
    }
    (out_tables / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(out_tables / "test_metrics.csv", index=False)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not readmitted", "Readmitted <30d"],
    )
    disp.plot()
    plt.title(f"Held-Out Test Confusion Matrix (threshold={threshold:.3f})")
    plt.tight_layout()
    plt.savefig(out_figures / "test_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---------- Final-test SHAP analysis ----------
    # SHAP is calculated with the same fitted XGBoost model used for prediction.
    # The threshold is used only to label the record as high/low risk; SHAP itself
    # remains a continuous contribution analysis.
    Xt, feature_names, xgb_model = _get_transformed_data(model, X)
    shap_values, explainer = _tree_shap_values(xgb_model, Xt)

    shap_matrix = pd.DataFrame(shap_values, columns=feature_names)
    shap_matrix.insert(0, "original_row_id", record_ids)
    shap_matrix.insert(1, "predicted_probability", p)
    shap_matrix.insert(2, "predicted_class", pred)
    shap_matrix.insert(3, "confidence", confidence)
    shap_matrix.insert(4, "correctness_confidence", correctness_confidence)
    shap_matrix.insert(5, "uncertainty", uncertainty)
    shap_matrix.insert(6, "threshold", threshold)
    shap_matrix.to_csv(out_tables / "test_shap_values.csv", index=False)

    global_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    global_importance.to_csv(out_tables / "test_shap_global_importance.csv", index=False)

    # Per-record top SHAP contributors, preserving the final threshold and confidence.
    top_k = int(cfg["test_evaluation"].get("shap_top_k", 10))
    local_rows = []
    for i in range(len(X)):
        row = shap_values[i]
        order = np.argsort(np.abs(row))[::-1][:top_k]
        top = [
            {
                "feature": str(feature_names[j]),
                "shap_value": float(row[j]),
                "direction": "increases_risk" if row[j] > 0 else "decreases_risk",
            }
            for j in order
        ]
        local_rows.append({
            "original_row_id": int(record_ids[i]),
            "y_true": int(y[i]),
            "predicted_probability": float(p[i]),
            "predicted_class": int(pred[i]),
            "predicted_label": "readmitted_within_30_days" if pred[i] else "not_readmitted_within_30_days",
            "confidence": float(confidence[i]),
            "correctness_confidence": float(correctness_confidence[i]),
            "uncertainty": float(uncertainty[i]),
            "prediction_correct": int(prediction_correct[i]),
            "threshold": threshold,
            "top_shap_features": json.dumps(top),
        })
    pd.DataFrame(local_rows).to_csv(out_tables / "test_record_level_shap.csv", index=False)

    plt.figure()
    shap.summary_plot(
        shap_values,
        Xt,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(out_figures / "test_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Held-out TEST evaluation complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Selected threshold: {threshold:.3f}")
    print(f"Mean correctness confidence: {correctness_confidence.mean():.3f}")
    print(f"Correctness ECE: {correctness_metrics["correctness_ece"]:.4f}")
    print("Per-record predictions: results/tables/test_predictions.csv")
    print("Per-record SHAP: results/tables/test_record_level_shap.csv")


if __name__ == "__main__":
    evaluate_test()
