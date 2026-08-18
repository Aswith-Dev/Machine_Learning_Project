from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import PaperFeatureEngineer, TopKMedicalSpecialty


def load_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def engineered_schema(X_raw: pd.DataFrame) -> pd.DataFrame:
    """Create only the deterministic schema needed to configure ColumnTransformer."""
    return PaperFeatureEngineer().fit_transform(X_raw)


def build_preprocessor(X_engineered: pd.DataFrame) -> ColumnTransformer:
    numeric = X_engineered.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = X_engineered.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(X_raw: pd.DataFrame, model):
    schema = engineered_schema(X_raw)
    pre = build_preprocessor(schema)

    return Pipeline([
        ("feature_engineering", PaperFeatureEngineer()),
        ("top10_specialty", TopKMedicalSpecialty()),
        ("preprocess", pre),
        ("model", model),
    ])
