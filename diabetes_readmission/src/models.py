from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def make_xgb(scale_pos_weight: float, random_state: int = 42, **kwargs):
    params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    params.update(kwargs)
    return XGBClassifier(**params)


def make_logistic(random_state: int = 42):
    return LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=random_state,
    )


def make_random_forest(random_state: int = 42):
    return RandomForestClassifier(
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def make_lightgbm(random_state: int = 42):
    return LGBMClassifier(
        class_weight="balanced",
        objective="binary",
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def make_stacking(scale_pos_weight: float, random_state: int = 42):
    base = [
        ("lr", make_logistic(random_state)),
        ("xgb", make_xgb(scale_pos_weight, random_state)),
        ("lgbm", make_lightgbm(random_state)),
    ]
    return StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(
            max_iter=2000,
            random_state=random_state,
        ),
        stack_method="predict_proba",
        n_jobs=-1,
    )
