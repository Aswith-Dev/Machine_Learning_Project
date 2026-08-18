from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


AGE_MIDPOINTS = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}

MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

CATEGORICAL_BASE = [
    "race", "gender", "age", "max_glu_serum", "A1Cresult",
    "change", "diabetesMed"
] + MEDICATION_COLUMNS

NUMERIC_BASE = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]

DIAG_FLAG_NAMES = [
    "circulatory", "respiratory", "digestive", "diabetes",
    "injury", "cancer", "other"
]


def _clean_code(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _numeric_code(x):
    s = _clean_code(x)
    if not s:
        return None
    m = re.match(r"^(\d{3})(?:\.\d+)?$", s)
    if m:
        return int(m.group(1))
    m = re.match(r"^(\d{3})$", s)
    return int(m.group(1)) if m else None


def diagnosis_group(code) -> str:
    """ICD-9 grouping used to implement the categories explicitly listed in the paper."""
    s = _clean_code(code).upper()
    n = _numeric_code(s)

    if n is None:
        return "other"

    if 390 <= n <= 459 or n == 785:
        return "circulatory"
    if 460 <= n <= 519 or n == 786:
        return "respiratory"
    if 520 <= n <= 579 or n == 787:
        return "digestive"
    if 250 <= n < 251:
        return "diabetes"
    if 800 <= n <= 999:
        return "injury"
    if 140 <= n <= 239:
        return "cancer"
    return "other"


def age_to_midpoint(x):
    if pd.isna(x):
        return np.nan
    return AGE_MIDPOINTS.get(str(x).strip(), np.nan)


def map_admission_type(x) -> str:
    try:
        v = int(x)
    except Exception:
        return "Other"
    return {1: "Emergency", 2: "Urgent", 3: "Elective"}.get(v, "Other")


def map_discharge_group(x) -> str:
    try:
        v = int(x)
    except Exception:
        return "Other"

    if v in {1, 6, 8, 13}:
        return "Home"
    if v in {2, 3, 4, 5, 10, 12, 15, 16, 17, 22, 23, 24, 27, 28, 29, 30}:
        return "Transfer"
    if v in {11, 19, 20, 21}:
        return "Expired"
    return "Other"


def map_source_group(x) -> str:
    try:
        v = int(x)
    except Exception:
        return "Other"

    if v == 7:
        return "Emergency Room"
    if v in {1, 2, 3}:
        return "Physician Referral"
    if v in {4, 5, 6, 10, 18, 19, 22}:
        return "Transfer"
    return "Other"


class TopKMedicalSpecialty(BaseEstimator, TransformerMixin):
    """Fit top-k specialties on the training fold; remaining values become Other."""

    def __init__(self, column="medical_specialty", k=10):
        self.column = column
        self.k = k

    def fit(self, X, y=None):
        s = X[self.column].fillna("Unknown").replace("?", "Unknown")
        self.top_categories_ = list(s.value_counts().head(self.k).index)
        return self

    def transform(self, X):
        X = X.copy()
        s = X[self.column].fillna("Unknown").replace("?", "Unknown")
        X[self.column] = np.where(s.isin(self.top_categories_), s, "Other")
        return X


class PaperFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Deterministic feature engineering described by the paper.

    The paper explicitly names:
      - age midpoint encoding
      - diagnosis grouping/flags
      - administrative grouping
      - top-10 specialty grouping

    It also exposes aggregate features in the paper's SHAP figure. Their exact
    formulas are not printed in the supplied main article; the formulas below are
    therefore explicit replication assumptions rather than claimed paper text.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Explicit categorical missing-value handling.
        for c in X.columns:
            if X[c].dtype == "object":
                X[c] = X[c].replace("?", "Unknown")
                X[c] = X[c].fillna("Unknown")

        # Age intervals -> numerical midpoints.
        if "age" in X:
            X["age"] = X["age"].map(age_to_midpoint)

        # Diagnosis presence flags across diag_1/2/3.
        diag_cols = ["diag_1", "diag_2", "diag_3"]
        for group in DIAG_FLAG_NAMES:
            X[f"has_{group}"] = 0

        grouped = {}
        for c in diag_cols:
            if c in X:
                grouped[c] = X[c].map(diagnosis_group)
        if grouped:
            dg = pd.DataFrame(grouped, index=X.index)
            for group in DIAG_FLAG_NAMES:
                X[f"has_{group}"] = dg.eq(group).any(axis=1).astype(int)

        # Remove original high-cardinality diagnosis columns.
        X = X.drop(columns=[c for c in diag_cols if c in X])

        # Administrative consolidation.
        if "admission_type_id" in X:
            X["admission_type_group"] = X["admission_type_id"].map(map_admission_type)
            X = X.drop(columns=["admission_type_id"])

        if "discharge_disposition_id" in X:
            X["discharge_group"] = X["discharge_disposition_id"].map(map_discharge_group)
            X = X.drop(columns=["discharge_disposition_id"])

        if "admission_source_id" in X:
            X["admission_source_group"] = X["admission_source_id"].map(map_source_group)
            X = X.drop(columns=["admission_source_id"])

        # Explicitly named aggregate concepts visible in the paper's SHAP figure.
        # These formulas are marked as assumptions because the main paper does not publish them.
        nums = lambda c: pd.to_numeric(X[c], errors="coerce").fillna(0) if c in X else 0

        if all(c in X for c in ["number_diagnoses", "time_in_hospital"]):
            X["disease_burden_score"] = nums("number_diagnoses")
            X["hospital_pressure_index"] = nums("time_in_hospital") * nums("number_diagnoses")

        if all(c in X for c in ["number_emergency", "number_inpatient"]):
            X["acute_care_utilization"] = nums("number_emergency") + nums("number_inpatient")

        if all(c in X for c in ["number_outpatient", "number_emergency", "number_inpatient"]):
            X["total_healthcare_visits"] = (
                nums("number_outpatient") + nums("number_emergency") + nums("number_inpatient")
            )

        # Drop non-model identifiers and variables explicitly excluded by the paper.
        X = X.drop(columns=[
            c for c in ["encounter_id", "patient_nbr", "weight", "payer_code"]
            if c in X
        ])

        return X


def prepare_target(df: pd.DataFrame) -> pd.Series:
    y = df["readmitted"].astype(str).map({"<30": 1, ">30": 0, "NO": 0})
    if y.isna().any():
        raise ValueError("Unexpected values in readmitted column.")
    return y.astype(int)


def build_engineered_dataset(df: pd.DataFrame):
    y = prepare_target(df)
    X = df.drop(columns=["readmitted"]).copy()

    engineer = PaperFeatureEngineer()
    X = engineer.fit_transform(X)

    # Fit top-10 specialty categories on the complete dataset for the saved
    # final engineered dataset. During model CV, the same transformer is fitted
    # only on each training fold inside the pipeline.
    if "medical_specialty" in X:
        s = X["medical_specialty"].replace("?", "Unknown").fillna("Unknown")
        top10 = list(s.value_counts().head(10).index)
        X["medical_specialty"] = np.where(s.isin(top10), s, "Other")

    return X, y
