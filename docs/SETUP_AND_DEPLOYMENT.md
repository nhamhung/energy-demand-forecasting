# Setup and Deployment

## Local setup

```bash
git clone https://github.com/nhamhhung/energy-demand-forecasting.git
cd energy-demand-forecasting
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r app/requirements.txt
pytest tests/
python scripts/smoke_app.py
streamlit run app/streamlit_app.py
```

## Deploy your fork

1. Replace the owner and URLs in `docs/deployment-config.yml`.
2. Push to a public GitHub repository whose default branch is `main`.
3. In Settings → Pages, select **GitHub Actions**.
4. In Streamlit Community Cloud, deploy branch `main` with entrypoint `app/streamlit_app.py`.
5. Configure Kaggle using either KAGGLE_API_TOKEN or a [kaggle] secrets section containing username and key.
6. Wait for CI and Pages, then record acceptance in `docs/DEPLOYMENT_ACCEPTANCE.md`.

## Required checks

Require `ruff`, both pytest jobs, both smoke jobs, and `pages-build`. Require pull requests and linear history; disable force pushes and deletion.

## Rollback

Revert through a protected pull request, rerun every required check, and verify Pages and Streamlit.
