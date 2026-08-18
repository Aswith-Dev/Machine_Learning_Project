from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

from .features import build_engineered_dataset, prepare_target
from .nested_cv import run_nested_xgb, run_baseline_comparison
from .calibration import fit_final_xgb, calibrate_full_model, evaluate_calibrated
from .bootstrap_analysis import bootstrap_metrics
from .clinical_utility import decision_curve, clinical_impact, threshold_metrics, risk_stratification
from .explainability import shap_summary, shap_stability


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config" / "config.json"


def load_cfg():
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def load_raw():
    cfg = load_cfg()
    df = pd.read_csv(ROOT / cfg["paths"]["raw_data"])
    return df.drop(columns=["readmitted"]).copy(), prepare_target(df), cfg


def stage_preprocess():
    cfg = load_cfg()
    df = pd.read_csv(ROOT / cfg["paths"]["raw_data"])
    X, y = build_engineered_dataset(df)
    out = X.copy()
    out["target"] = y
    path = ROOT / cfg["paths"]["processed_data"]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Saved {path}")
    print(f"Engineered shape: {out.shape}")
    print(f"Predictor count before one-hot: {X.shape[1]}")
    print(f"Positive class: {int(y.sum())}; negative class: {int((y==0).sum())}")


def stage_nested():
    X, y, cfg = load_raw()
    out = ROOT / "results" / "tables"
    run_nested_xgb(X, y, cfg, out)
    run_baseline_comparison(X, y, cfg, out)
    print("Nested CV and baseline comparison complete.")


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
    X, y, cfg = load_raw()
    best = get_final_best_params()
    model = fit_final_xgb(X, y, best, cfg["seed"])
    calibrated = calibrate_full_model(model, X, y)
    metrics = evaluate_calibrated(calibrated, X, y, cfg["clinical"]["operating_threshold"])

    out = ROOT / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out/"final_calibrated_metrics.csv", index=False)
    pd.DataFrame([best]).to_csv(out/"final_xgb_parameters.csv", index=False)

    import joblib
    joblib.dump(model, ROOT/"results"/"xgb_final_model.joblib")
    joblib.dump(calibrated, ROOT/"results"/"xgb_calibrated_model.joblib")

    p = calibrated.predict_proba(X)[:,1]
    pd.DataFrame({"y_true": y, "predicted_probability": p}).to_csv(
        out/"full_dataset_calibrated_predictions.csv", index=False
    )
    print(json.dumps(metrics, indent=2))


def load_final_predictions():
    p = ROOT / "results" / "tables" / "full_dataset_calibrated_predictions.csv"
    if not p.exists():
        raise FileNotFoundError("Run --stage final_model first.")
    d = pd.read_csv(p)
    return d["y_true"].to_numpy(), d["predicted_probability"].to_numpy()


def stage_bootstrap():
    y, p = load_final_predictions()
    cfg = load_cfg()
    out, ci = bootstrap_metrics(
        y, p,
        n_resamples=cfg["bootstrap"]["n_resamples"],
        seed=cfg["seed"],
        threshold=cfg["clinical"]["operating_threshold"],
    )
    out.to_csv(ROOT/"results"/"tables"/"bootstrap_distributions.csv", index=False)
    (ROOT/"results"/"tables"/"bootstrap_confidence_intervals.json").write_text(
        json.dumps(ci, indent=2)
    )
    print(json.dumps(ci, indent=2))


def stage_clinical():
    y, p = load_final_predictions()
    thresholds = np.arange(0.01, 0.301, 0.01)

    dca = decision_curve(y, p, thresholds)
    cic = clinical_impact(y, p, thresholds)
    tm = threshold_metrics(y, p, [0.05,0.10,0.15,0.20,0.25,0.30])
    rs = risk_stratification(y, p)

    dca.to_csv(ROOT/"results"/"tables"/"decision_curve.csv", index=False)
    cic.to_csv(ROOT/"results"/"tables"/"clinical_impact.csv", index=False)
    tm.to_csv(ROOT/"results"/"tables"/"threshold_metrics.csv", index=False)
    rs.to_csv(ROOT/"results"/"tables"/"risk_stratification.csv", index=False)

    plt.figure()
    plt.plot(dca["threshold"], dca["model_net_benefit"], label="Model")
    plt.plot(dca["threshold"], dca["treat_all_net_benefit"], label="Treat-All")
    plt.plot(dca["threshold"], dca["treat_none_net_benefit"], label="Treat-None")
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT/"results"/"figures"/"decision_curve.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(cic["threshold"], cic["predicted_high_risk_per_1000"], label="Predicted High Risk (per 1000)")
    plt.plot(cic["threshold"], cic["true_readmissions_per_1000"], label="True Readmissions (per 1000)")
    plt.xlabel("Threshold Probability")
    plt.ylabel("Patients per 1000")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT/"results"/"figures"/"clinical_impact_curve.png", dpi=300)
    plt.close()

    precision, recall, _ = precision_recall_curve(y, p)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT/"results"/"figures"/"precision_recall_curve.png", dpi=300)
    plt.close()

    print("Clinical utility outputs generated.")


def stage_shap():
    import joblib
    import pandas as pd

    cfg = load_cfg()

    # Load the raw dataset
    df = pd.read_csv(ROOT/"data"/"raw"/"diabetic_data.csv")

    # Build the same engineered feature matrix used by the final model
    X, y = build_engineered_dataset(df)

    model_path = ROOT/"results"/"xgb_final_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError("Run --stage final_model first.")

    model = joblib.load(model_path)

    # Ensure SHAP receives exactly the same feature columns
    # and ordering used during model training.
    expected_columns = list(
        model.named_steps["preprocess"].feature_names_in_
    )

    X = X.loc[:, expected_columns].copy()

    importance = shap_summary(
        model,
        X,
        ROOT/"results"/"figures"
    )

    stability = shap_stability(
        X,
        y,
        model,
        n_resamples=cfg["shap_stability"]["n_resamples"],
        top_k=cfg["shap_stability"]["top_k"],
        seed=cfg["seed"],
    )

    stability.to_csv(
        ROOT/"results"/"tables"/"shap_stability.csv",
        index=False
    )

    print(importance.head(20).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=[
        "preprocess", "nested_cv", "final_model", "bootstrap", "clinical", "shap"
    ])
    args = parser.parse_args()

    if args.stage == "preprocess":
        stage_preprocess()
    elif args.stage == "nested_cv":
        stage_nested()
    elif args.stage == "final_model":
        stage_final()
    elif args.stage == "bootstrap":
        stage_bootstrap()
    elif args.stage == "clinical":
        stage_clinical()
    elif args.stage == "shap":
        stage_shap()


if __name__ == "__main__":
    main()
