"""Dataset Overview page: what's in the Romania hourly demand/production
series before any modeling — the cleaned time series, the three real
data-quality issues, the generation mix, and the daily/weekly/seasonal
patterns.
"""

import plotly.express as px
import streamlit as st

from . import shared
from energy_demand_forecasting import config, data


def render():
    st.title("📊 Dataset Overview")
    st.caption(
        "Real hourly electricity Consumption and Production (by "
        "generation source) for Romania's national grid, 2019-onward — "
        "a live, continuously-updated dataset, checked directly (see "
        "the Feature Engineering page for what that means for "
        "reproducibility)."
    )

    frame = shared.get_clean_frame()

    c1, c2, c3 = st.columns(3)
    c1.metric("Hourly readings", f"{len(frame):,}")
    c2.metric("Years covered", f"{frame.index.year.nunique()}")
    c3.metric("Peak demand", f"{frame[config.TARGET_COL].max():,.0f} MW")
    st.caption(
        f"Peak demand on {frame[config.TARGET_COL].idxmax():%d %B %Y, %H:%M}, after cleaning "
        "(implausible readings removed — see below)."
    )

    st.subheader("The full series")
    fig = px.line(frame[config.TARGET_COL].reset_index(), x=config.DATETIME_COL, y=config.TARGET_COL, title="Hourly demand")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Three real data-quality issues — and they're not the same issue")
    raw = data.load_raw()
    dupes = raw[raw[config.DATETIME_COL].duplicated(keep=False)]
    gaps = data.gap_lengths(raw)
    implausible = data.implausible_readings(raw)
    col1, col2, col3 = st.columns(3)
    col1.metric("Duplicate timestamps (DST fall-back)", len(dupes))
    col2.metric("Longest single missing-data gap", f"{gaps.max()} hours" if len(gaps) else "0")
    col3.metric("Implausible readings removed", f"{len(implausible)} hours")
    st.caption(
        "The duplicates are a one-hour-per-year Daylight Saving Time "
        "artifact. The long gaps are a real, unrelated reporting/scraping "
        "outage concentrated in 2024-2025 — up to 83 consecutive hours, "
        "nothing like the isolated DST gaps. The implausible readings are a "
        "third fault in the 2025-2026 feed: stretches where every column is a "
        "scaled-up copy of the hour before, so night-time demand climbs to "
        "10,000-12,000 MW (and nuclear output past the plant's capacity) before "
        "dropping back within an hour. They are treated as missing and "
        "interpolated, like the gaps."
    )
    if len(implausible):
        with st.expander("Which readings were removed?"):
            st.dataframe(implausible.round(2))

    st.subheader("Generation mix and renewable share")
    renewable_share = frame[config.RENEWABLE_COLS].sum(axis=1) / frame["Production"]
    by_year = renewable_share.groupby(frame.index.year).mean().reset_index()
    by_year.columns = ["year", "renewable_share"]
    st.plotly_chart(px.bar(by_year, x="year", y="renewable_share", title="Renewable share of production, by year"), width="stretch")
    solar_by_year = frame.groupby(frame.index.year)["Solar"].mean().reset_index()
    st.plotly_chart(px.bar(solar_by_year, x=config.DATETIME_COL, y="Solar", title="Mean solar output (MW), by year"), width="stretch")
    st.caption(
        "Mean solar output roughly doubled between 2023 and 2025 — a "
        "real, checkable signature of Romania's recent solar buildout."
    )

    st.subheader("Daily, weekly, and seasonal patterns")
    t1, t2 = st.tabs(["By hour of day", "By month"])
    with t1:
        by_hour = frame.groupby(frame.index.hour)[config.TARGET_COL].mean().reset_index()
        by_hour.columns = ["hour", config.TARGET_COL]
        st.plotly_chart(px.line(by_hour, x="hour", y=config.TARGET_COL, markers=True, title="Mean demand by hour of day"), width="stretch")
    with t2:
        by_month = frame.groupby(frame.index.month)[config.TARGET_COL].mean().reset_index()
        by_month.columns = ["month", config.TARGET_COL]
        st.plotly_chart(px.line(by_month, x="month", y=config.TARGET_COL, markers=True, title="Mean demand by month (winter-dominant, not bimodal)"), width="stretch")
