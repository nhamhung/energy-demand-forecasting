"""SHAP-based model interpretability.

`model.MODEL_FACTORIES` mixes tree-based estimators (`RandomForestRegressor`,
`LGBMRegressor`) with `LinearRegression`, so `compute_shap_values` picks
the explainer explicitly: `shap.TreeExplainer` for tree-based models
(exact, near-instant), `shap.Explainer`'s generic auto-dispatch
otherwise (correctly resolves to `LinearExplainer` for a linear model).
"""

import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def compute_shap_values(
    pipeline: Pipeline, X: pd.DataFrame, max_samples: int = 500, random_state: int = 42
) -> tuple[shap.Explanation, pd.DataFrame]:
    """Compute SHAP values for a fitted pipeline ending in a regressor."""
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=random_state)

    preprocessing = pipeline[:-1]
    feature_names = X.columns.tolist()
    transformed = preprocessing.transform(X)
    X_transformed = pd.DataFrame(transformed, columns=feature_names, index=X.index)

    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "feature_importances_") or type(estimator).__name__ == "LGBMRegressor":
        explainer = shap.TreeExplainer(estimator)
    else:
        explainer = shap.Explainer(estimator, X_transformed)
    explanation = explainer(X_transformed)
    return explanation, X_transformed


def top_shap_features(explanation: shap.Explanation, top_n: int = 15) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value. A regressor has a
    single output, so unlike this portfolio's classification projects,
    there's no `class_index` to choose — every SHAP explanation here is
    already about the one thing being predicted (demand in MW).
    """
    importance = abs(explanation.values).mean(axis=0)
    return (
        pd.DataFrame({"feature": explanation.feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
