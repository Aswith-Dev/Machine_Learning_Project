"""Fast integrity checks for the patient-level reliability extension.

This script does not rerun the computationally expensive nested XGBoost search.
It validates the split and the new correctness-calibration utilities.
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.correctness_calibration import (
    fit_correctness_calibrator,
    reliability_metrics,
    selective_prediction_table,
)

ROOT = Path(__file__).resolve().parent


def main():
    train_path = ROOT / "data/splits/train.csv"
    test_path = ROOT / "data/splits/test.csv"
    manifest_path = ROOT / "data/splits/split_manifest.json"

    train = pd.read_csv(train_path, usecols=["patient_nbr", "readmitted"])
    test = pd.read_csv(test_path, usecols=["patient_nbr", "readmitted"])

    overlap = set(train.patient_nbr).intersection(set(test.patient_nbr))
    assert not overlap, f"Train/test patient leakage: {len(overlap)} patients"

    y = train.readmitted.map({"<30": 1, ">30": 0, "NO": 0}).astype(int)
    groups = train.patient_nbr
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_rows = []
    for fold, (tr, va) in enumerate(cv.split(train, y, groups), start=1):
        ov = set(groups.iloc[tr]).intersection(set(groups.iloc[va]))
        assert not ov, f"Outer fold {fold} patient leakage: {len(ov)}"
        fold_rows.append({
            "fold": fold,
            "train_patients": int(groups.iloc[tr].nunique()),
            "validation_patients": int(groups.iloc[va].nunique()),
            "patient_overlap": len(ov),
        })

    # Small deterministic smoke test for correctness calibration.
    p = pd.Series([0.95, 0.85, 0.70, 0.30, 0.10, 0.05])
    y_demo = pd.Series([1, 1, 0, 1, 0, 1])
    cal, d = fit_correctness_calibrator(y_demo, p, threshold=0.5)
    metrics, bins = reliability_metrics(d["correctness_confidence"], d["prediction_correct"])
    selective = selective_prediction_table(d["correctness_confidence"], d["prediction_correct"])
    assert ((d["correctness_confidence"] >= 0) & (d["correctness_confidence"] <= 1)).all()
    assert 0 <= metrics["correctness_ece"] <= 1

    manifest = json.loads(manifest_path.read_text())
    print("PASS: train/test patient overlap = 0")
    print("PASS: all 5 outer folds have patient overlap = 0")
    print(json.dumps({
        "train_rows": len(train),
        "test_rows": len(test),
        "train_patients": train.patient_nbr.nunique(),
        "test_patients": test.patient_nbr.nunique(),
        "manifest_patient_overlap": manifest["patient_overlap"],
        "correctness_smoke_ece": metrics["correctness_ece"],
    }, indent=2))
    pd.DataFrame(fold_rows).to_csv(ROOT / "results" / "tables" / "patient_group_validation.csv", index=False)
    print("Wrote results/tables/patient_group_validation.csv")


if __name__ == "__main__":
    main()
