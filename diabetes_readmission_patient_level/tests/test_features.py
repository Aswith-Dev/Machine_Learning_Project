import pandas as pd

from src.features import build_engineered_dataset, prepare_target
from src.pipeline import build_model_pipeline
from src.models import make_xgb


def test_target_counts():
    df = pd.read_csv("data/raw/diabetic_data.csv")
    y = prepare_target(df)
    assert len(y) == 101766
    assert int(y.sum()) == 11357
    assert int((y == 0).sum()) == 90409


def test_feature_engineering_runs():
    df = pd.read_csv("data/raw/diabetic_data.csv")
    X, y = build_engineered_dataset(df)
    assert len(X) == 101766
    assert len(y) == 101766
    assert "age" in X.columns
    assert "has_circulatory" in X.columns
    assert "admission_type_group" in X.columns
    assert "discharge_group" in X.columns
    assert "admission_source_group" in X.columns
    assert "payer_code" not in X.columns
    assert "weight" not in X.columns
    assert "encounter_id" not in X.columns
    assert "patient_nbr" not in X.columns


def test_full_pipeline_smoke():
    df = pd.read_csv("data/raw/diabetic_data.csv").iloc[:500].copy()
    y = prepare_target(df)
    X = df.drop(columns=["readmitted"])
    model = make_xgb((y == 0).sum() / (y == 1).sum(), 42, n_estimators=5, max_depth=2, learning_rate=0.1)
    pipe = build_model_pipeline(X, model)
    pipe.fit(X, y)
    p = pipe.predict_proba(X)[:, 1]
    assert len(p) == 500
    assert ((p >= 0) & (p <= 1)).all()
