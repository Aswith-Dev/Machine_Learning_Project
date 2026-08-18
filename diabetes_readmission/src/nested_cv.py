from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score
)

from .models import (
    make_xgb, make_logistic, make_random_forest, make_lightgbm,
    make_stacking
)
from .pipeline import build_model_pipeline


def run_nested_xgb(X_raw, y, cfg, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outer = StratifiedKFold(
        n_splits=cfg["cv"]["outer_folds"],
        shuffle=cfg["cv"]["shuffle"],
        random_state=cfg["cv"]["random_state"],
    )

    rows = []
    fold_params = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X_raw, y), start=1):
        Xtr, Xte = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

        neg = int((ytr == 0).sum())
        pos = int((ytr == 1).sum())
        spw = neg / pos

        xgb = make_xgb(spw, cfg["seed"])
        pipe = build_model_pipeline(Xtr, xgb)

        inner = StratifiedKFold(
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
        search.fit(Xtr, ytr)

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
    metrics.to_csv(output_dir/"xgb_nested_fold_metrics.csv", index=False)
    params.to_csv(output_dir/"xgb_best_params_by_fold.csv", index=False)

    summary = {
        "roc_auc_mean": float(metrics["roc_auc"].mean()),
        "roc_auc_std": float(metrics["roc_auc"].std(ddof=1)),
        "f1_mean": float(metrics["f1"].mean()),
        "pr_auc_mean": float(metrics["average_precision"].mean()),
    }
    (output_dir/"xgb_nested_summary.json").write_text(json.dumps(summary, indent=2))
    return metrics, params


def run_baseline_comparison(X_raw, y, cfg, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outer = StratifiedKFold(
        n_splits=cfg["cv"]["outer_folds"],
        shuffle=cfg["cv"]["shuffle"],
        random_state=cfg["cv"]["random_state"],
    )

    rows = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(X_raw, y), start=1):
        Xtr, Xte = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
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
    out.to_csv(output_dir/"baseline_comparison_fold_metrics.csv", index=False)
    summary = out.groupby("model")["roc_auc"].agg(["mean", "std"]).reset_index()
    summary.to_csv(output_dir/"baseline_comparison_summary.csv", index=False)
    return out, summary
