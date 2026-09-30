"""Feature Engineering page: calendar cycles, lag/rolling windows,
lagged generation-mix features, and the stationarity/autocorrelation
checks that justify them.
"""

import pandas as pd
import streamlit as st

from . import shared
from energy_demand_forecasting import config, model


def render():
    st.title("🔧 Feature Engineering")
    st.caption("The real engineering decisions behind this project's feature set.")

    frame = shared.get_clean_frame()
    X, y = shared.get_features()

    st.subheader("1. A live dataset — what that means for reproducibility")
    st.markdown(
        "Unlike this portfolio's other projects, this data source is "
        "**continuously updated** — checked directly, an earlier "
        "download had 46,002 rows through April 2024; a later one had "
        "62,810 rows through March 2026, with the older rows unchanged. "
        "Every number in this app reflects one snapshot, not a fixed file."
    )

    st.subheader("2. The leakage trap in the generation-mix columns")
    st.markdown(
        f"`Production` and its {len(config.EXOGENOUS_COLS) - 1} source "
        "breakdowns are recorded at the **same hour** as `Consumption`, "
        "not before it. Using them directly would be circular for a real "
        "forecasting task — a system predicting tomorrow's demand doesn't "
        "know tomorrow's actual generation mix either. Every generation-mix "
        "feature below is lagged by 24 hours."
    )
    lag_cols = [c for c in X.columns if c.endswith("_lag_24h") and c not in ("lag_24h",)]
    st.dataframe(X[lag_cols].tail(5))

    st.subheader("3. Stationarity and autocorrelation, checked directly")
    adf_result = model.stationarity_test(frame[config.TARGET_COL])
    st.markdown(
        f"ADF statistic **{adf_result['statistic']:.2f}**, p-value "
        f"**{adf_result['p_value']:.4g}** — formally stationary (no unit "
        "root), even though the daily/monthly plots on the Overview page "
        "show an obvious repeating pattern. Those aren't contradictory: "
        "ADF tests for a *stochastic trend*, not for the presence of "
        "seasonality — a bounded, repeating pattern doesn't have one."
    )
    acf_values = model.autocorrelation(frame[config.TARGET_COL], n_lags=200)
    st.line_chart(pd.Series(acf_values, name="autocorrelation"))
    st.caption(
        f"ACF at 24h: {acf_values[24]:.3f}, at 168h (1 week): {acf_values[168]:.3f} — "
        "both genuinely high, which is what justifies using them as lag features "
        "rather than picking them by convention."
    )

    st.subheader("4. Cyclical calendar encoding, and a real Romanian holiday calendar")
    st.dataframe(X[["hour", "hour_sin", "hour_cos", "dayofweek", "is_holiday"]].tail(5))
