"""Model Insights page: naive-baseline comparison, model sweep via
TimeSeriesSplit (including a lightweight neural network baseline), and
live SHAP.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from . import shared
from energy_demand_forecasting import model
from energy_demand_forecasting.interpretability import top_shap_features

PALETTE = ["#2c5cc5", "#5b8def", "#8fb4f2", "#e07a5f", "#81b29a"]


@st.cache_data(show_spinner="Sweeping model families (first load only)...")
def _model_sweep():
    X, y = shared.get_features()
    rows = []
    for name, factory in model.MODEL_FACTORIES.items():
        scores = model.cross_validate_pipeline(X, y, estimator=factory(), cv=5)
        rows.append({"model": name, **scores})
    return pd.DataFrame(rows).sort_values("mape_mean")


@st.cache_data(show_spinner="Evaluating the naive seasonal baseline...")
def _naive_baseline():
    X, y = shared.get_features()
    _, X_test, _, y_test = model.chronological_train_test_split(X, y)
    return model.naive_seasonal_baseline(X_test, y_test)


def render():
    st.title("🧠 Model Insights")
    st.caption(
        "Why a naive seasonal baseline matters here, whether a neural "
        "network beats gradient boosting, and how the models compare "
        "using time-aware cross-validation (never a random shuffle)."
    )

    try:
        shared.get_pipeline()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("The bar every model has to clear")
    baseline = _naive_baseline()
    st.metric("Naive baseline MAPE (\"same hour last week\")", f"{baseline['mape']:.1%}")
    st.caption("No model at all — just `lag_168h` used directly as the prediction.")

    st.subheader("Model comparison (5-fold TimeSeriesSplit, MAPE)")
    sweep = _model_sweep()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(sweep["model"], sweep["mape_mean"], color=PALETTE[0])
    ax.axvline(baseline["mape"], linestyle="--", color="gray", label="naive baseline")
    ax.set_xlabel("MAPE")
    for bar, value in zip(bars, sweep["mape_mean"]):
        ax.text(value + 0.001, bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center", fontsize=9)
    ax.legend()
    st.pyplot(fig)
    st.caption(
        "Every model beats the naive baseline by a wide margin. LightGBM "
        "wins clearly — including against `MLP (neural network)`, a "
        "lightweight feedforward net included specifically to test "
        "whether a neural approach earns its extra complexity here. It "
        "doesn't. LightGBM is what's saved to `models/model.joblib`."
    )
    st.dataframe(sweep[["model", "mae_mean", "rmse_mean", "mape_mean", "mape_std"]], hide_index=True)

    st.subheader("What does the saved model actually rely on? (SHAP)")
    st.caption("Computed live from the saved model — first load takes a few seconds.")
    explanation, _ = shared.get_shap_explanation(sample_size=500)
    top = top_shap_features(explanation, top_n=15).sort_values("mean_abs_shap")

    fig3, ax3 = plt.subplots(figsize=(7, 6))
    ax3.barh(top["feature"], top["mean_abs_shap"], color=PALETTE[0])
    ax3.set_xlabel("Mean |SHAP value| (MW)")
    ax3.set_title("Top 15 features")
    st.pyplot(fig3)
    st.caption(
        "`lag_168h` (last week, same hour) and `lag_24h` (yesterday, "
        "same hour) dominate. `Solar_lag_24h` showing up this high is a "
        "genuine surprise — plausibly a proxy for season/weather rather "
        "than a direct effect of solar generation on demand."
    )
