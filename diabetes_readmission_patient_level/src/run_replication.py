from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

from .data_split import create_train_test_split
from .features import build_engineered_dataset, prepare_target
from .nested_cv import run_nested_xgb, run_baseline_comparison
from .calibration import fit_training_calibration, evaluate_calibrated
from .correctness_calibration import (
    fit_correctness_calibrator, reliability_metrics, selective_prediction_table,
)
from .bootstrap_analysis import bootstrap_metrics
from .clinical_utility import decision_curve, clinical_impact, risk_stratification
from .threshold_selection import threshold_metrics, select_best_threshold
from .explainability import shap_summary, shap_stability


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config" / "config.json"


def load_cfg():
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def stage_split():
    cfg = load_cfg()
    raw = ROOT / cfg["paths"]["raw_data"]
    train = ROOT / cfg["paths"]["train_raw"]
    test = ROOT / cfg["paths"]["test_raw"]
    manifest = ROOT / cfg["paths"]["split_manifest"]
    train_df, test_df = create_train_test_split(
        raw,
        train,
        test,
        manifest,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
    )
    print(f"Train split: {train_df.shape}")
    print(f"Test split:  {test_df.shape}")
    print(f"Saved: {train}")
    print(f"Saved: {test}")


def load_train_raw():
    cfg = load_cfg()
    path = ROOT / cfg["paths"]["train_raw"]
    if not path.exists():
        raise FileNotFoundError("Run --stage split first.")
    df = pd.read_csv(path)
    y = prepare_target(df)
    drop_cols = [c for c in ["readmitted", "original_row_id"] if c in df.columns]
    groups = df["patient_nbr"].copy()
    X = df.drop(columns=drop_cols).copy()
    return X, y, groups, cfg


def stage_preprocess():
    cfg = load_cfg()
    train_path = ROOT / cfg["paths"]["train_raw"]
    if not train_path.exists():
        raise FileNotFoundError("Run --stage split first.")
    df = pd.read_csv(train_path)
    # This saved dataset is TRAIN-ONLY. The held-out test set is never used here.
    df = df.drop(columns=["original_row_id"], errors="ignore")
    X, y2 = build_engineered_dataset(df)
    out = X.copy()
    out["target"] = y2
    path = ROOT / cfg["paths"]["processed_train_data"]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Saved train-only engineered data: {path}")
    print(f"Engineered shape: {out.shape}")
    print(f"Predictor count before one-hot: {X.shape[1]}")
    print(f"Positive class: {int(y2.sum())}; negative class: {int((y2==0).sum())}")


def stage_nested():
    X, y, groups, cfg = load_train_raw()
    out = ROOT / "results" / "tables"
    run_nested_xgb(X, y, groups, cfg, out)
    run_baseline_comparison(X, y, groups, cfg, out)
    print("Nested CV and baseline comparison complete on TRAIN ONLY.")


def get_final_best_params():
    p = ROOT / "results" / "tables" / "xgb_best_params_by_fold.csv"
    if not p.exists():
        raise FileNotFoundError("Run --stage nested_cv first.")
    df = pd.read_csv(p)
    best = {}
    for c in ["model__n_estimators", "model__max_depth", "model__learning_rate"]:
        best[c] = df[c].mode().iloc[0]
    return best


def stage_final():
    X, y, groups, cfg = load_train_raw()
    best = get_final_best_params()

    # OOF predictions are generated only inside TRAIN for calibration and
    # threshold selection. The held-out TEST set is untouched.
    model, calibrator, oof_raw, oof_calibrated = fit_training_calibration(
        X,
        y,
        groups,
        best,
        n_splits=cfg["calibration"]["oof_folds"],
        seed=cfg["seed"],
    )

    threshold, tm, best_row = select_best_threshold(
        y,
        oof_calibrated,
        min_threshold=cfg["threshold_selection"]["min"],
        max_threshold=cfg["threshold_selection"]["max"],
        step=cfg["threshold_selection"]["step"],
    )

    out = ROOT / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)

    # Second-stage calibration: estimate P(the selected prediction is correct).
    # This is fitted only from training OOF predictions; TEST is untouched.
    correctness_calibrator, correctness_oof = fit_correctness_calibrator(
        y.to_numpy(), oof_calibrated, threshold
    )
    correctness_oof["record_index"] = np.arange(len(y))
    correctness_oof["threshold"] = threshold
    correctness_metrics, correctness_bins = reliability_metrics(
        correctness_oof["correctness_confidence"],
        correctness_oof["prediction_correct"],
        n_bins=10,
    )
    correctness_oof.to_csv(ROOT / "results" / "tables" / "train_oof_correctness_predictions.csv", index=False)
    correctness_bins.to_csv(ROOT / "results" / "tables" / "train_oof_correctness_reliability_bins.csv", index=False)
    pd.DataFrame([correctness_metrics]).to_csv(
        ROOT / "results" / "tables" / "train_oof_correctness_metrics.csv", index=False
    )
    selective_prediction_table(
        correctness_oof["correctness_confidence"],
        correctness_oof["prediction_correct"],
    ).to_csv(ROOT / "results" / "tables" / "train_oof_selective_prediction.csv", index=False)

    pd.DataFrame([best]).to_csv(out / "final_xgb_parameters.csv", index=False)
    tm.to_csv(out / "train_oof_threshold_metrics.csv", index=False)
    pd.DataFrame([{
        "selected_threshold": threshold,
        "selection_metric": "f1",
        "selection_source": "training-only out-of-fold calibrated probabilities",
        "best_f1": float(best_row["f1"]),
    }]).to_csv(out / "final_threshold.csv", index=False)

    oof_pred = (oof_calibrated >= threshold).astype(int)
    oof_metrics = evaluate_calibrated_from_probabilities(
        y.to_numpy(), oof_calibrated, threshold
    )
    pd.DataFrame([oof_metrics]).to_csv(out / "train_oof_calibrated_metrics.csv", index=False)
    pd.DataFrame({
        "record_index": np.arange(len(y)),
        "y_true": y.to_numpy(),
        "raw_probability": oof_raw,
        "calibrated_probability": oof_calibrated,
        "predicted_class": oof_pred,
        "confidence": correctness_oof["base_confidence"].to_numpy(),
        "correctness_confidence": correctness_oof["correctness_confidence"].to_numpy(),
        "uncertainty": correctness_oof["uncertainty"].to_numpy(),
        "prediction_correct": correctness_oof["prediction_correct"].to_numpy(),
        "threshold": threshold,
    }).to_csv(out / "train_oof_predictions.csv", index=False)

    joblib.dump(model, ROOT / "results" / "xgb_final_model.joblib")
    joblib.dump(calibrator, ROOT / "results" / "sigmoid_calibrator.joblib")
    joblib.dump(correctness_calibrator, ROOT / "results" / "correctness_calibrator.joblib")

    print(json.dumps({"selected_threshold": threshold, **oof_metrics, **{f"correctness_{k}": v for k, v in correctness_metrics.items()}}, indent=2))


def evaluate_calibrated_from_probabilities(y, p, threshold):
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, brier_score_loss,
        f1_score, precision_score, recall_score,
    )

    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "ppv": float(precision_score(y, pred, zero_division=0)),
        "npv": float(tn / (tn + fn)) if tn + fn else float("nan"),
        "threshold": float(threshold),
    }


def load_train_oof_predictions():
    p = ROOT / "results" / "tables" / "train_oof_predictions.csv"
    if not p.exists():
        raise FileNotFoundError("Run --stage final_model first.")
    d = pd.read_csv(p)
    return d["y_true"].to_numpy(), d["calibrated_probability"].to_numpy()


def stage_bootstrap():
    y, p = load_train_oof_predictions()
    cfg = load_cfg()
    threshold = float(pd.read_csv(ROOT / "results" / "tables" / "final_threshold.csv").iloc[0]["selected_threshold"])
    out, ci = bootstrap_metrics(
        y,
        p,
        n_resamples=cfg["bootstrap"]["n_resamples"],
        seed=cfg["seed"],
        threshold=threshold,
    )
    out.to_csv(ROOT / "results" / "tables" / "bootstrap_train_oof_distributions.csv", index=False)
    (ROOT / "results" / "tables" / "bootstrap_train_oof_confidence_intervals.json").write_text(
        json.dumps(ci, indent=2), encoding="utf-8"
    )
    print(json.dumps(ci, indent=2))


def stage_clinical():
    y, p = load_train_oof_predictions()
    cfg = load_cfg()
    thresholds = np.arange(0.01, 0.301, 0.01)
    threshold = float(pd.read_csv(ROOT / "results" / "tables" / "final_threshold.csv").iloc[0]["selected_threshold"])

    dca = decision_curve(y, p, thresholds)
    cic = clinical_impact(y, p, thresholds)
    tm = threshold_metrics(y, p, thresholds)
    rs = risk_stratification(y, p)

    dca.to_csv(ROOT / "results" / "tables" / "decision_curve_train_oof.csv", index=False)
    cic.to_csv(ROOT / "results" / "tables" / "clinical_impact_train_oof.csv", index=False)
    tm.to_csv(ROOT / "results" / "tables" / "threshold_metrics_train_oof.csv", index=False)
    rs.to_csv(ROOT / "results" / "tables" / "risk_stratification_train_oof.csv", index=False)

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(dca["threshold"], dca["model_net_benefit"], label="Model")
    plt.plot(dca["threshold"], dca["treat_all_net_benefit"], label="Treat-All")
    plt.plot(dca["threshold"], dca["treat_none_net_benefit"], label="Treat-None")
    plt.axvline(threshold, linestyle="--", label=f"Selected threshold = {threshold:.2f}")
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "decision_curve_train_oof.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(cic["threshold"], cic["predicted_high_risk_per_1000"], label="Predicted High Risk (per 1000)")
    plt.plot(cic["threshold"], cic["true_readmissions_per_1000"], label="True Readmissions (per 1000)")
    plt.axvline(threshold, linestyle="--", label=f"Selected threshold = {threshold:.2f}")
    plt.xlabel("Threshold Probability")
    plt.ylabel("Patients per 1000")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "clinical_impact_curve_train_oof.png", dpi=300)
    plt.close()

    precision, recall, _ = precision_recall_curve(y, p)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "precision_recall_curve_train_oof.png", dpi=300)
    plt.close()

    print(f"Clinical utility outputs generated from TRAIN OOF predictions; threshold={threshold:.3f}.")


def stage_shap():
    cfg = load_cfg()
    X, y, _, _ = load_train_raw()
    model_path = ROOT / "results" / "xgb_final_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Run --stage final_model first.")

    model = joblib.load(model_path)
    importance = shap_summary(model, X, ROOT / "results" / "figures")
    stability = shap_stability(
        X,
        y,
        model,
        n_resamples=cfg["shap_stability"]["n_resamples"],
        top_k=cfg["shap_stability"]["top_k"],
        seed=cfg["seed"],
    )
    stability.to_csv(ROOT / "results" / "tables" / "shap_stability_train.csv", index=False)
    print(importance.head(20).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=[
        "split", "preprocess", "nested_cv", "final_model", "bootstrap", "clinical", "shap"
    ])
    args = parser.parse_args()

    stages = {
        "split": stage_split,
        "preprocess": stage_preprocess,
        "nested_cv": stage_nested,
        "final_model": stage_final,
        "bootstrap": stage_bootstrap,
        "clinical": stage_clinical,
        "shap": stage_shap,
    }
    stages[args.stage]()


if __name__ == "__main__":
    main()
