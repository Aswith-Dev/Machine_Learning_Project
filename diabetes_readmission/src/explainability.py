from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.base import clone

from .bootstrap_analysis import shap_bootstrap_indices


def _get_transformed_data(fitted_pipeline, X):
    pre = fitted_pipeline.named_steps["preprocess"]
    model = fitted_pipeline.named_steps["model"]

    # Ensure the dataframe has exactly the same feature order
    # as the data used when the preprocessing transformer was fitted.
    expected_columns = list(pre.feature_names_in_)
    X = X.loc[:, expected_columns].copy()

    Xt = pre.transform(X)
    names = pre.get_feature_names_out()

    return Xt, names, model


def shap_summary(fitted_pipeline, X, output_dir, max_display=20):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Xt, names, model = _get_transformed_data(fitted_pipeline, X)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xt)

    if isinstance(sv, list):
        sv = sv[1]

    plt.figure()
    shap.summary_plot(
        sv,
        Xt,
        feature_names=names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir/"shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    importance = pd.DataFrame({
        "feature": names,
        "mean_abs_shap": np.abs(sv).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    importance.to_csv(output_dir/"shap_global_importance.csv", index=False)
    return importance


def shap_stability(X, y, fitted_pipeline, n_resamples=100, top_k=10, seed=42):
    """
    Retrain the complete fitted pipeline on each bootstrap sample and record
    the top-k mean-absolute-SHAP features, following the paper's stability design.
    """
    rng = np.random.default_rng(seed)
    counts = {}
    n = len(X)

    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        Xb = X.iloc[idx]
        yb = y.iloc[idx]

        pipe = clone(fitted_pipeline)
        pipe.fit(Xb, yb)

        Xt, names, model = _get_transformed_data(pipe, X)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[1]

        imp = pd.DataFrame({
            "feature": names,
            "mean_abs_shap": np.abs(sv).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

        for f in imp.head(top_k)["feature"]:
            counts[f] = counts.get(f, 0) + 1

    out = pd.DataFrame([
        {"feature": f, "top10_frequency": c, "stability_percent": 100*c/n_resamples}
        for f, c in counts.items()
    ]).sort_values(["stability_percent", "feature"], ascending=[False, True])

    return out
