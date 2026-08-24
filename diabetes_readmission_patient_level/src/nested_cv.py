from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score
)

from .models import (
    make_xgb, make_logistic, make_random_forest, make_lightgbm,
    make_stacking
)
from .pipeline import build_model_pipeline


def _validate_groups(groups, n_rows):
    groups = pd.Series(groups).reset_index(drop=True)
    if len(groups) != n_rows:
        raise ValueError("groups must have the same number of rows as X and y.")
    if groups.isna().any():
        raise ValueError("groups contains missing patient identifiers.")
    return groups


def run_nested_xgb(X_raw, y, groups, cfg, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y = pd.Series(y).reset_index(drop=True)
    X_raw = X_raw.reset_index(drop=True)
    groups = _validate_groups(groups, len(y))

    outer = StratifiedGroupKFold(
        n_splits=cfg["cv"]["outer_folds"],
        shuffle=cfg["cv"]["shuffle"],
        random_state=cfg["cv"]["random_state"],
    )

    rows = []
    fold_params = []
    fold_group_checks = []

    for fold, (train_idx, test_idx) in enumerate(
        outer.split(X_raw, y, groups=groups), start=1
    ):
        Xtr, Xte = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        groups_tr = groups.iloc[train_idx]
        groups_te = groups.iloc[test_idx]

        overlap = set(groups_tr).intersection(set(groups_te))
        if overlap:
            raise RuntimeError(f"Outer fold {fold} has {len(overlap)} patient overlaps.")
        fold_group_checks.append({
            "fold": fold,
            "train_patients": int(groups_tr.nunique()),
            "validation_patients": int(groups_te.nunique()),
            "patient_overlap": int(len(overlap)),
        })

        neg = int((ytr == 0).sum())
        pos = int((ytr == 1).sum())
        spw = neg / pos

        xgb = make_xgb(spw, cfg["seed"])
        pipe = build_model_pipeline(Xtr, xgb)

        inner = StratifiedGroupKFold(
            n_splits=cfg["cv"]["inner_folds"],
            shuffle=True,
            random_state=cfg["seed"],
        )

        grid = {
            "model__n_estimators": cfg["xgboost"]["n_estimators"],
            "model__max_depth": cfg["xgboost"]["max_depth"],
            "model__learning_rate": cfg["xgboost"]["learning_rate"],
        }

        search = GridSearchCV(
            pipe,
            grid,
            scoring=cfg["cv"]["scoring"],
            cv=inner,
            n_jobs=-1,
            refit=True,
        )
        search.fit(Xtr, ytr, groups=groups_tr)

        pred = search.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, pred)
        pred_class = (pred >= 0.5).astype(int)

        rows.append({
            "fold": fold,
            "roc_auc": auc,
            "average_precision": average_precision_score(yte, pred),
            "precision": precision_score(yte, pred_class, zero_division=0),
            "recall": recall_score(yte, pred_class, zero_division=0),
            "f1": f1_score(yte, pred_class, zero_division=0),
        })
        fold_params.append({"fold": fold, **search.best_params_})

    metrics = pd.DataFrame(rows)
    params = pd.DataFrame(fold_params)
    checks = pd.DataFrame(fold_group_checks)
    metrics.to_csv(output_dir / "xgb_nested_fold_metrics.csv", index=False)
    params.to_csv(output_dir / "xgb_best_params_by_fold.csv", index=False)
    checks.to_csv(output_dir / "xgb_nested_patient_group_checks.csv", index=False)

    summary = {
        "roc_auc_mean": float(metrics["roc_auc"].mean()),
        "roc_auc_std": float(metrics["roc_auc"].std(ddof=1)),
        "f1_mean": float(metrics["f1"].mean()),
        "pr_auc_mean": float(metrics["average_precision"].mean()),
        "cv_type": "StratifiedGroupKFold",
        "group_variable": "patient_nbr",
        "patient_overlap_across_outer_folds": 0,
    }
    (output_dir / "xgb_nested_summary.json").write_text(json.dumps(summary, indent=2))
    return metrics, params


def run_baseline_comparison(X_raw, y, groups, cfg, output_dir):
    """Baseline comparison using the same patient-level outer grouping."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y = pd.Series(y).reset_index(drop=True)
    X_raw = X_raw.reset_index(drop=True)
    groups = _validate_groups(groups, len(y))

    outer = StratifiedGroupKFold(
        n_splits=cfg["cv"]["outer_folds"],
        shuffle=cfg["cv"]["shuffle"],
        random_state=cfg["cv"]["random_state"],
    )

    rows = []
    for fold, (train_idx, test_idx) in enumerate(
        outer.split(X_raw, y, groups=groups), start=1
    ):
        Xtr, Xte = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        groups_tr, groups_te = groups.iloc[train_idx], groups.iloc[test_idx]
        if set(groups_tr).intersection(set(groups_te)):
            raise RuntimeError(f"Baseline outer fold {fold} has patient leakage.")
        spw = (ytr == 0).sum() / (ytr == 1).sum()

        model_defs = {
            "LogisticRegression": make_logistic(cfg["seed"]),
            "RandomForest": make_random_forest(cfg["seed"]),
            "XGBoost": make_xgb(spw, cfg["seed"]),
            "LightGBM": make_lightgbm(cfg["seed"]),
            "Stacking": make_stacking(spw, cfg["seed"]),
        }

        for name, estimator in model_defs.items():
            pipe = build_model_pipeline(Xtr, estimator)
            pipe.fit(Xtr, ytr)
            pred = pipe.predict_proba(Xte)[:, 1]
            rows.append({
                "fold": fold,
                "model": name,
                "roc_auc": roc_auc_score(yte, pred),
                "pr_auc": average_precision_score(yte, pred),
            })

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "baseline_comparison_fold_metrics.csv", index=False)
    summary = out.groupby("model")["roc_auc"].agg(["mean", "std"]).reset_index()
    summary.to_csv(output_dir / "baseline_comparison_summary.csv", index=False)
    return out, summary
