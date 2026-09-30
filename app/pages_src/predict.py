"""Predict page: two modes.

- **Historical**: pick any hour in the dataset and compare the model's
  single-hour prediction against what demand actually was.
- **Forecast forward**: pick a starting hour and a horizon (up to 24h
  ahead — see `config.MAX_FORECAST_HORIZON_HOURS` for why the cap),
  and watch the model forecast recursively, feeding its own predictions
  back in as inputs for later hours.
"""

import pandas as pd
import streamlit as st

from . import shared
from energy_demand_forecasting import config, model


def _load_random_hour():
    X, y = shared.get_features()
    idx = X.sample(1).index[0]
    st.session_state["selected_timestamp"] = idx


def _render_historical(X: pd.DataFrame, y: pd.Series, pipeline) -> None:
    if "selected_timestamp" not in st.session_state:
        st.session_state["selected_timestamp"] = X.index[-24]

    st.button("🎲 Load a random hour", on_click=_load_random_hour)

    min_ts, max_ts = X.index.min(), X.index.max()
    col1, col2 = st.columns(2)
    with col1:
        picked_date = st.date_input(
            "Date", value=st.session_state["selected_timestamp"].date(), min_value=min_ts.date(), max_value=max_ts.date()
        )
    with col2:
        picked_hour = st.slider("Hour of day", 0, 23, value=st.session_state["selected_timestamp"].hour)

    candidate = pd.Timestamp(picked_date) + pd.Timedelta(hours=picked_hour)
    if candidate not in X.index:
        st.warning("No data for that exact hour (outside the range with enough history) — showing the nearest available hour instead.")
        candidate = X.index[X.index.get_indexer([candidate], method="nearest")[0]]
    st.session_state["selected_timestamp"] = candidate

    row = X.loc[[candidate]]
    actual = y.loc[candidate]
    predicted = pipeline.predict(row)[0]
    error_pct = abs(predicted - actual) / actual

    st.subheader(f"{candidate:%A, %B %d, %Y — %H:%M}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Actual demand", f"{actual:,.0f} MW")
    c2.metric("Predicted demand", f"{predicted:,.0f} MW")
    c3.metric("Error", f"{error_pct:.1%}")

    st.subheader("In context: the surrounding week")
    window = y.loc[candidate - pd.Timedelta(days=3) : candidate + pd.Timedelta(days=3)]
    chart_df = pd.DataFrame({"Actual": window})
    chart_df.loc[candidate, "Selected hour (predicted)"] = predicted
    st.line_chart(chart_df)

    with st.expander("What features fed this prediction?"):
        st.dataframe(row.T.rename(columns={candidate: "value"}))


@st.cache_data(show_spinner="Measuring how error changes with forecast horizon (first load only)...")
def _error_by_horizon_sample():
    frame = shared.get_clean_frame()
    pipeline = shared.get_pipeline()
    n = len(frame)
    # Anchors spread across the most recent 20% of the series, spaced apart,
    # each with a full 24h of real future data to check the forecast against.
    candidates = frame.index[int(n * 0.8) : n - config.MAX_FORECAST_HORIZON_HOURS - 1]
    step = max(len(candidates) // 30, 1)
    anchors = list(candidates[::step])[:30]
    return model.forecast_error_by_horizon(pipeline, frame, anchors, max_horizon=config.MAX_FORECAST_HORIZON_HOURS)


def _render_forecast(frame: pd.DataFrame, pipeline) -> None:
    st.caption(
        f"Capped at {config.MAX_FORECAST_HORIZON_HOURS}h ahead, not days — matching a related "
        "piece of prior work on a different electricity dataset, which found multi-step "
        "forecasts hold up much better over a short horizon than a genuinely long one."
    )

    max_anchor = frame.index[-config.MAX_FORECAST_HORIZON_HOURS - 1]
    if "forecast_anchor" not in st.session_state:
        st.session_state["forecast_anchor"] = max_anchor

    col1, col2 = st.columns(2)
    with col1:
        picked_date = st.date_input(
            "Forecast starting from this date",
            value=st.session_state["forecast_anchor"].date(),
            min_value=frame.index.min().date(),
            max_value=max_anchor.date(),
        )
    with col2:
        picked_hour = st.slider("...and this hour", 0, 23, value=st.session_state["forecast_anchor"].hour)

    anchor = pd.Timestamp(picked_date) + pd.Timedelta(hours=picked_hour)
    if anchor not in frame.index or anchor > max_anchor:
        anchor = frame.index[frame.index.get_indexer([min(anchor, max_anchor)], method="nearest")[0]]
    st.session_state["forecast_anchor"] = anchor

    horizon = st.slider("Forecast horizon (hours ahead)", 1, config.MAX_FORECAST_HORIZON_HOURS, value=config.MAX_FORECAST_HORIZON_HOURS)

    forecast = model.recursive_forecast(pipeline, frame, anchor, horizon_hours=horizon)
    has_actuals = forecast.index[-1] in frame.index
    actual_future = frame.loc[forecast.index, config.TARGET_COL] if has_actuals else None

    st.subheader(f"Forecasting {horizon}h ahead from {anchor:%A, %B %d, %Y — %H:%M}")
    if has_actuals:
        final_error = abs(forecast.iloc[-1] - actual_future.iloc[-1]) / actual_future.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Predicted demand at +{horizon}h", f"{forecast.iloc[-1]:,.0f} MW")
        c2.metric("Actual demand (real data exists for this anchor)", f"{actual_future.iloc[-1]:,.0f} MW")
        c3.metric("Error at this horizon", f"{final_error:.1%}")
    else:
        st.metric(f"Predicted demand at +{horizon}h", f"{forecast.iloc[-1]:,.0f} MW")
        st.caption("This anchor is close to the end of the data — no real future value to compare against yet.")

    recent_actual = frame[config.TARGET_COL].loc[anchor - pd.Timedelta(days=2) : anchor]
    chart_df = pd.DataFrame({"Actual (history)": recent_actual})
    chart_df = pd.concat([chart_df, forecast.rename("Forecast").to_frame()], axis=1)
    if has_actuals:
        chart_df["Actual (what really happened)"] = actual_future
    st.line_chart(chart_df)

    st.subheader("Does error compound with a longer horizon?")
    st.caption(
        "Averaged across 30 sampled forecast-start points, each checked against real "
        "future data. Recursive forecasting can compound error the further out it goes — "
        "here, capping the horizon at 24h keeps the lag and generation-mix features "
        "grounded in real data throughout, so only the rolling-window features ever read "
        "back a prediction instead of a real value."
    )
    error_by_horizon = _error_by_horizon_sample()
    st.line_chart(error_by_horizon.set_index("steps_ahead")["mape"])


def render():
    st.title("🎯 Predict: Hourly Electricity Demand")

    try:
        pipeline = shared.get_pipeline()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    mode = st.radio(
        "Mode",
        ["Historical (actual vs. predicted)", f"Forecast forward (up to {config.MAX_FORECAST_HORIZON_HOURS}h ahead)"],
        horizontal=True,
    )

    if mode.startswith("Historical"):
        st.caption(
            "Pick any hour in the dataset's history and compare the model's single-hour "
            "prediction (made only from that hour's calendar position, its own recent "
            "past, and yesterday's generation mix) against what demand actually was."
        )
        X, y = shared.get_features()
        _render_historical(X, y, pipeline)
    else:
        frame = shared.get_clean_frame()
        _render_forecast(frame, pipeline)
