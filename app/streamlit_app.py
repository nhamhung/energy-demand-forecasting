"""Multi-page Streamlit app entry point.

Run locally:
    streamlit run app/streamlit_app.py

Or via Docker (from the project root):
    docker build -t energy-demand-forecasting-app -f app/Dockerfile .
    docker run -p 8501:8501 energy-demand-forecasting-app
"""

import streamlit as st

from pages_src import feature_engineering, model_insights, overview, predict

st.set_page_config(page_title="Energy Demand Forecasting", page_icon="⚡", layout="wide")

pages = [
    st.Page(predict.render, title="Predict", icon="🎯", url_path="predict", default=True),
    st.Page(overview.render, title="Dataset Overview", icon="📊", url_path="overview"),
    st.Page(feature_engineering.render, title="Feature Engineering", icon="🔧", url_path="feature-engineering"),
    st.Page(model_insights.render, title="Model Insights", icon="🧠", url_path="model-insights"),
]

st.navigation(pages).run()
