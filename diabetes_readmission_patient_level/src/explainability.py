from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.base import clone

from .bootstrap_analysis import shap_bootstrap_indices


def _get_transformed_data(fitted_pipeline, X_raw):
    """Apply the fitted feature-engineering and preprocessing steps to raw X."""
    feature_engineering = fitted_pipeline.named_steps["feature_engineering"]
    top10 = fitted_pipeline.named_steps["top10_specialty"]
    pre = fitted_pipeline.named_steps["preprocess"]
    model = fitted_pipeline.named_steps["model"]

    X_engineered = feature_engineering.transform(X_raw)
    X_engineered = top10.transform(X_engineered)

    expected_columns = list(pre.feature_names_in_)
    X_engineered = X_engineered.loc[:, expected_columns].copy()

    Xt = pre.transform(X_engineered)
    names = pre.get_feature_names_out()

    return Xt, names, model


def _tree_shap_values(model, Xt, max_background=500):
    """Compute SHAP values for the fitted XGBoost classifier."""
    if hasattr(Xt, "toarray"):
        Xt_for_explainer = Xt
    else:
        Xt_for_explainer = np.asarray(Xt)

    background = Xt_for_explainer

    if hasattr(background, "shape") and background.shape[0] > max_background:
        rng = np.random.default_rng(42)

        idx = rng.choice(
            background.shape[0],
            max_background,
            replace=False,
        )

        background = background[idx]

    try:
        explainer = shap.TreeExplainer(
            model,
            data=background,
            feature_perturbation="interventional",
            model_output="probability",
        )

        sv = explainer.shap_values(Xt_for_explainer)

    except Exception:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xt_for_explainer)

    if isinstance(sv, list):
        sv = sv[1]

    return np.asarray(sv), explainer


def shap_summary(
    fitted_pipeline,
    X_raw,
    output_dir,
    max_display=20,
    eval_size=5000,
    seed=42,
):
    """Generate the main SHAP summary on a fixed evaluation subset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(X_raw)
    eval_size = min(eval_size, n)

    rng = np.random.default_rng(seed)

    if eval_size < n:
        eval_idx = rng.choice(
            n,
            size=eval_size,
            replace=False,
        )
        X_eval = X_raw.iloc[eval_idx].copy()
    else:
        X_eval = X_raw

    Xt, names, model = _get_transformed_data(
        fitted_pipeline,
        X_eval,
    )

    sv, _ = _tree_shap_values(model, Xt)

    plt.figure()

    shap.summary_plot(
        sv,
        Xt,
        feature_names=names,
        max_display=max_display,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "shap_summary_train.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    importance = pd.DataFrame(
        {
            "feature": names,
            "mean_abs_shap": np.abs(sv).mean(axis=0),
        }
    ).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    importance.to_csv(
        output_dir / "shap_global_importance_train.csv",
        index=False,
    )

    return importance


def shap_stability(
    X,
    y,
    fitted_pipeline,
    n_resamples=100,
    top_k=10,
    seed=42,
    eval_size=1000,
):
    """
    Retrain on bootstrap samples and measure top-k SHAP feature stability.

    The complete training dataset is used for every bootstrap resample.
    SHAP values are calculated on a fixed evaluation subset to reduce
    computation while keeping the evaluation population consistent
    across all resampled models.
    """
    rng = np.random.default_rng(seed)

    counts = {}
    n = len(X)

    eval_size = min(eval_size, n)

    eval_idx = rng.choice(
        n,
        size=eval_size,
        replace=False,
    )

    X_eval = X.iloc[eval_idx].copy()

    for _ in range(n_resamples):
        idx = rng.integers(
            0,
            n,
            n,
        )

        Xb = X.iloc[idx]
        yb = y.iloc[idx]

        pipe = clone(fitted_pipeline)

        pipe.fit(
            Xb,
            yb,
        )

        Xt, names, model = _get_transformed_data(
            pipe,
            X_eval,
        )

        sv, _ = _tree_shap_values(
            model,
            Xt,
        )

        imp = pd.DataFrame(
            {
                "feature": names,
                "mean_abs_shap": np.abs(sv).mean(axis=0),
            }
        ).sort_values(
            "mean_abs_shap",
            ascending=False,
        )

        for feature in imp.head(top_k)["feature"]:
            counts[feature] = counts.get(feature, 0) + 1

    out = pd.DataFrame(
        [
            {
                "feature": feature,
                "top10_frequency": count,
                "stability_percent": 100 * count / n_resamples,
            }
            for feature, count in counts.items()
        ]
    )

    if not out.empty:
        out = out.sort_values(
            ["stability_percent", "feature"],
            ascending=[False, True],
        )

    return out
