# Energy Demand Forecasting

A worked, end-to-end data science project forecasting hourly
electricity demand (`Consumption`, megawatts) for Romania's national
grid, using real, published data from 2019 onward together with its
generation mix (nuclear, wind, hydro, oil & gas, coal, solar, biomass).

## Why this dataset

I originally built this project on US grid data (PJM Interconnection),
then switched to this Romanian dataset
(`stefancomanita/hourly-electricity-consumption-and-production` on
Kaggle) specifically because the generation-mix breakdown is genuinely
useful, interesting data — it lets me ask whether recent generation mix
carries forecasting signal, and it comes with its own real
energy-transition story (mean solar output roughly doubled between 2023
and 2025).

**This dataset is live and continuously updated** — checked directly,
an earlier download had 46,002 rows through April 2024; the version
used here has 62,810 rows through March 2026, with older rows
unchanged. Every number quoted in this project reflects one snapshot,
not a fixed, versioned file — see `config.py`'s module docstring.

I still couldn't find a real, public, hourly demand dataset for
Vietnam's own grid — checked directly, EVN's public data releases are
highly aggregated. See `report/report.qmd`'s Discussion section for
what a genuine Vietnam version would actually need.

## What's here

| Deliverable | Where |
|---|---|
| A well-documented notebook building the model, applying good practices | `notebooks/01_eda_and_modeling.ipynb` |
| A multi-page Streamlit app to explore the data and try the model | `app/streamlit_app.py` + `app/pages_src/` (+ `app/Dockerfile`) |
| A research-style writeup | `report/report.qmd` |
| A script that trains and saves the production model | `scripts/train.py` |

All four share one feature-engineering/model source of truth in
`src/energy_demand_forecasting/`, so the notebook, the app, and the
script can never quietly drift apart.

## A real leakage trap in the generation-mix columns

`Production` and its 7 source breakdowns are recorded at the **same
hour** as `Consumption`, not before it. Using them directly to predict
that same hour's demand would be circular for a genuine forecasting
task — generation is dispatched *in response to* demand, not fixed in
advance, so a real system forecasting tomorrow's demand doesn't know
tomorrow's actual generation mix either. Every generation-mix feature
in `features.py` is lagged by 24 hours specifically to avoid this.

## Two real data-quality issues, and they are not the same issue

- A **Daylight Saving Time** artifact (1 missing hour every spring, 1
  duplicated hour every autumn) — the same underlying cause as any
  DST-observing country, verified directly.
- A **separate, unrelated real outage problem** concentrated in
  2024-2025: multi-hour gaps up to 83 consecutive hours, consistent
  with a genuine reporting/scraping gap in the live feed this dataset's
  maintainer pulls from. Not present at all in 2019-2023.

See `config.py`'s and `data.py`'s docstrings for the full detail and
how each is fixed.

## Statistical rigor before modeling

- An **Augmented Dickey-Fuller stationarity test** — and why the
  series being formally "stationary" doesn't contradict its obvious
  daily/monthly seasonality (ADF tests for a stochastic trend, not for
  the presence of a repeating pattern).
- An **autocorrelation function**, used to empirically justify the
  `lag_24h`/`lag_168h` feature choices rather than picking them by
  convention.
- A **leakage-safe seasonal-decomposition feature** (fit only on the
  training split, reusing only the strictly periodic component) —
  tested via ablation, and found to add nothing once fixed. An honest
  negative result, reported rather than dropped quietly.

## A model chosen after clearing a real baseline — including against a neural network

`LightGBM` beats Linear Regression, Random Forest, and a lightweight
`MLPRegressor` neural-network baseline on every metric, and clears the
naive seasonal baseline (~7.8% MAPE) by a wide margin. See
`report/report.qmd`'s Methodology section for the full comparison.

## What this project deliberately doesn't do

No weather data, a different climate/market structure than Vietnam
(Romania's demand is winter-heating-dominant, likely the opposite shape
from Vietnam's), and the 2024-2025 test-period numbers are partly
measured against linearly-interpolated (not fully real) data.
`report/report.qmd`'s Discussion section covers what a real,
Vietnam-specific version would need.

## Get the data

[Kaggle API](https://www.kaggle.com/docs/api) (`pip install kaggle`,
then put your `kaggle.json`/access token in `~/.kaggle/`):

```bash
kaggle datasets download -d stefancomanita/hourly-electricity-consumption-and-production -p data/raw
unzip -o data/raw/hourly-electricity-consumption-and-production.zip -d data/raw
mv data/raw/electricityConsumptionAndProductioction.csv data/raw/Romania_hourly.csv
```

This produces `data/raw/Romania_hourly.csv`. **Note:** this source is
live — re-downloading later will add more recent rows than the snapshot
this project's numbers were computed from.

## Run the notebook

```bash
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

## Train from the command line

```bash
python scripts/train.py                    # default: LightGBM — the measured winner
python scripts/train.py --model "MLP (neural network)"
python scripts/train.py --help
```

## Run the app

A 4-page app: **Predict** — two modes: pick any historical hour for a
single-hour actual-vs-predicted comparison, or forecast forward up to
24 hours ahead (capped there deliberately, not days — see
`config.MAX_FORECAST_HORIZON_HOURS`'s docstring), watching the model
recursively feed its own predictions back in as inputs, with a chart
showing whether error actually compounds with horizon length —
**Dataset Overview** (including the generation mix and renewable-share
trend), **Feature Engineering** (including the stationarity/
autocorrelation checks), and **Model Insights** (naive baseline + model
comparison + live SHAP).

```bash
streamlit run app/streamlit_app.py
```

Or in Docker:

```bash
docker build -t energy-demand-forecasting-app -f app/Dockerfile .
docker run -p 8501:8501 energy-demand-forecasting-app
```

Then open http://localhost:8501.

## Render the research writeup

```bash
quarto render report/report.qmd
```

## Run the tests

```bash
pytest tests/
```

These test the DST/outage cleaning logic, feature engineering
(including dedicated leakage checks for both the rolling-window and the
generation-mix features, and the seasonal-lookup's no-lookahead
property), and the chronological-split/model logic directly with
synthetic data — no download needed, and they already pass without any
real data.

## Deploy

The Streamlit app fetches the live source dataset through the Kaggle API at runtime. Configure either KAGGLE_API_TOKEN or a [kaggle] secrets section containing username and key.

- Repository: <https://github.com/nhamhhung/energy-demand-forecasting>
- Report: <https://nhamhung.github.io/energy-demand-forecasting/>
- Streamlit: <https://energy-demand-forecasting.streamlit.app>
- Fork setup: [docs/SETUP_AND_DEPLOYMENT.md](docs/SETUP_AND_DEPLOYMENT.md)
