from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split


def create_train_test_split(
    raw_path: str | Path,
    train_path: str | Path,
    test_path: str | Path,
    manifest_path: str | Path,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic patient-level stratified holdout.

    The split unit is patient_nbr, not encounter rows. A patient and all of
    that patient's encounters are assigned to exactly one side of the split.
    Patient-level stratification uses whether a patient has at least one
    <30-day readmission encounter as the patient-level stratum.
    """
    raw_path = Path(raw_path)
    train_path = Path(train_path)
    test_path = Path(test_path)
    manifest_path = Path(manifest_path)

    df = pd.read_csv(raw_path)
    required = {"readmitted", "patient_nbr"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {sorted(missing)}")

    patient_strata = (
        df.groupby("patient_nbr")["readmitted"]
        .apply(lambda s: int((s.astype(str) == "<30").any()))
        .rename("patient_has_readmission_lt30")
        .reset_index()
    )

    train_patients, test_patients = train_test_split(
        patient_strata["patient_nbr"],
        test_size=test_size,
        stratify=patient_strata["patient_has_readmission_lt30"],
        random_state=random_state,
    )

    train_patient_set = set(train_patients.tolist())
    test_patient_set = set(test_patients.tolist())
    overlap = train_patient_set.intersection(test_patient_set)
    if overlap:
        raise RuntimeError(f"Patient leakage detected during holdout split: {len(overlap)} patients overlap.")

    train_df = df[df["patient_nbr"].isin(train_patient_set)].copy()
    test_df = df[df["patient_nbr"].isin(test_patient_set)].copy()

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve original row identity so every held-out record can be traced back
    # without using patient/encounter identifiers as model features.
    train_df.insert(0, "original_row_id", train_df.index.astype(int))
    test_df.insert(0, "original_row_id", test_df.index.astype(int))
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    manifest = {
        "raw_rows": int(len(df)),
        "raw_unique_patients": int(df["patient_nbr"].nunique()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_unique_patients": int(train_df["patient_nbr"].nunique()),
        "test_unique_patients": int(test_df["patient_nbr"].nunique()),
        "patient_overlap": int(train_df["patient_nbr"].isin(test_df["patient_nbr"]).sum()),
        "test_size_requested": float(test_size),
        "random_state": int(random_state),
        "split_unit": "patient_nbr",
        "stratified_by": "patient_has_readmission_lt30",
        "train_positive_rate": float((train_df["readmitted"] == "<30").mean()),
        "test_positive_rate": float((test_df["readmitted"] == "<30").mean()),
        "note": "Patient-level holdout: all encounters from each patient are assigned to exactly one side. Patient-level grouping is also enforced in nested CV and calibration OOF generation.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return train_df, test_df
