"""Model Insights page: both no-model baselines, a model sweep via
TimeSeriesSplit cross-validation on the training period only (including
a lightweight neural network baseline), and live SHAP.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from . import shared
from energy_demand_forecasting import model
from energy_demand_forecasting.interpretability import top_shap_features

PALETTE = ["#2c5cc5", "#5b8def", "#8fb4f2", "#e07a5f", "#81b29a"]
PERSISTENCE = "Persistence (previous hour)"
SEASONAL = "Seasonal naive (same hour last week)"


def _train_test():
    X, y = shared.get_features()
    return model.chronological_train_test_split(X, y)


@st.cache_data(show_spinner="Sweeping model families on the training period (first load only)...")
def _model_sweep():
    X_train, _, y_train, _ = _train_test()
    rows = []
    for name, factory in model.MODEL_FACTORIES.items():
        scores = model.cross_validate_pipeline(X_train, y_train, estimator=factory(), cv=5)
        rows.append({"model": name, **scores})
    for name, scores in model.cross_validate_baselines(shared.get_clean_frame(), X_train, y_train, cv=5).items():
        rows.append({"model": name, **scores})
    return pd.DataFrame(rows).sort_values("mape_mean")


@st.cache_data(show_spinner="Evaluating both baselines on the test period...")
def _baselines():
    _, X_test, _, y_test = _train_test()
    return {
        PERSISTENCE: model.persistence_baseline(shared.get_clean_frame(), y_test),
        SEASONAL: model.naive_seasonal_baseline(X_test, y_test),
    }


def render():
    st.title("🧠 Model Insights")
    st.caption(
        "Two no-model baselines, whether a neural network beats gradient "
        "boosting, and how the models compare using time-aware "
        "cross-validation on the training period (never a random shuffle)."
    )

    try:
        shared.get_pipeline()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Two bars to clear, for two different jobs")
    baselines = _baselines()
    c1, c2 = st.columns(2)
    c1.metric("Persistence MAPE (\"same as the previous hour\")", f"{baselines[PERSISTENCE]['mape']:.1%}")
    c2.metric("Seasonal naive MAPE (\"same hour last week\")", f"{baselines[SEASONAL]['mape']:.1%}")
    st.caption(
        "Both measured on the chronological test period (the last 15% of hours). "
        "Persistence is the honest bar for a **1-hour-ahead** forecast, which is what "
        "these models make; it is not available further ahead, where the seasonal "
        "naive baseline is the fair bar (see Predict → Forecast forward)."
    )

    st.subheader("Model comparison (5-fold TimeSeriesSplit on the training period, MAPE)")
    sweep = _model_sweep()
    is_baseline = sweep["model"].isin([PERSISTENCE, SEASONAL])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    colors = [PALETTE[3] if b else PALETTE[0] for b in is_baseline]
    bars = ax.barh(sweep["model"], sweep["mape_mean"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("MAPE (mean over 5 folds; orange = no-model baseline)")
    for bar, value in zip(bars, sweep["mape_mean"]):
        ax.text(value + 0.001, bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center", fontsize=9)
    st.pyplot(fig)

    scores = sweep.set_index("model")["mape_mean"]
    learned = scores.drop([PERSISTENCE, SEASONAL])
    best_name, best = learned.idxmin(), learned.min()
    beat_seasonal = learned[learned < scores[SEASONAL]].index.tolist()
    if best <= scores[PERSISTENCE]:
        persistence_note = (
            f"One hour ahead, persistence ({scores[PERSISTENCE]:.1%}) is roughly as good as "
            f"the best model ({best_name}, {best:.1%})"
        )
    else:
        persistence_note = (
            f"One hour ahead, persistence ({scores[PERSISTENCE]:.1%}) beats every model here "
            f"(best: {best_name}, {best:.1%})"
        )
    st.caption(
        "Cross-validation uses the training period only, so its folds never touch the "
        "test period above; the baselines are scored on the same validation blocks. "
        f"{len(beat_seasonal)} of {len(learned)} models beat the seasonal naive baseline "
        f"({scores[SEASONAL]:.1%}). {persistence_note}: the models' value is in "
        "forecasting several hours ahead, where the previous hour isn't known yet (see "
        "Predict → Forecast forward). LightGBM is what's saved to `models/model.joblib`; "
        "`MLP (neural network)`, included to test whether a neural approach earns its extra "
        "complexity, doesn't. On the bundled one-year sample the training period is only "
        "about ten months, so the models are weaker than on the full dataset (see the report "
        "for full-data numbers)."
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
