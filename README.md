# Energy Demand Forecasting

> **New to Python, or just want to try it?** See [START_HERE.md](START_HERE.md): download the ZIP,
> double-click `start_windows.bat` (Windows) or `start_mac.command` (macOS), and it sets everything up.

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

## Prerequisites

Install once, before Setup below:

| Dependency | Why | Install |
|---|---|---|
| **Python 3.12** | This project's `.venv` is built against 3.12 — a different version may resolve incompatible package versions from `requirements.txt`. | [python.org/downloads](https://www.python.org/downloads/) or a version manager (e.g. `pyenv install 3.12`) |
| **Quarto** | Renders `report/report.qmd` — a standalone binary, not a Python package, so `pip install` never gets it. | [quarto.org/docs/get-started](https://quarto.org/docs/get-started/) |
| **Kaggle API token** | Needed only for the complete dataset; the default Streamlit app uses a bundled sample, and `pytest` uses synthetic data. | Kaggle account → **Account → Create New API Token** → save the downloaded file as `~/.kaggle/kaggle.json` (`%USERPROFILE%\.kaggle\kaggle.json` on Windows). See the [Kaggle API docs](https://www.kaggle.com/docs/api). |
| **Docker** (optional) | Only if you want to run the app in its pre-baked container instead of `streamlit run`. | [docker.com/get-started](https://www.docker.com/get-started/) |

## What's here

| Deliverable | Where |
|---|---|
| A step-by-step guide notebook that walks a beginner through the data, every feature, the models, evaluation, recursive 24-hour forecasting, and SHAP, using the project's own `src/` functions | `notebooks/01_eda_and_modeling.ipynb` |
| A multi-page Streamlit app to explore the data and try the model | `app/streamlit_app.py` + `app/pages_src/` (+ `app/Dockerfile`) |
| A research-style writeup | `report/report.qmd` |
| A script that trains and saves the production model | `scripts/train.py` |

All four share one feature-engineering/model source of truth in
`src/energy_demand_forecasting/`, so the notebook, the app, and the
script can never quietly drift apart.

## Project layout

```
data/raw/                       # downloaded Kaggle CSV (gitignored — see below)
data/processed/                  # any cached intermediate data (gitignored)
notebooks/                       # the main EDA + modeling notebook
src/energy_demand_forecasting/   # shared config, data loading, feature engineering, model code, SHAP interpretability
models/                          # trained pipeline artifact (model.joblib)
app/                              # Streamlit app (multi-page, app/pages_src/) + Dockerfile
scripts/                          # train.py
report/                           # Quarto research writeup
tests/                            # pytest tests for cleaning/feature/model logic (synthetic data — no download needed)
```

## Setup

```bash
python3.12 -m venv .venv          # use the 3.12 interpreter specifically
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Making changes

`src/energy_demand_forecasting/` is the single source of truth —
`config.py` (paths, schema, constants), `data.py` (loading, DST/outage
cleaning), `features.py` (lag/rolling/calendar features, the
leakage-safe seasonal lookup), `model.py` (pipelines, training,
evaluation, `recursive_forecast`), `interpretability.py` (SHAP). The
notebook, the app, and `scripts/train.py` all import from here; nothing
re-derives logic locally, so a change here propagates everywhere
automatically.

The edit loop:

```bash
# 1. Edit src/energy_demand_forecasting/*.py

# 2. Check it against the test suite (fast, synthetic data, no download needed)
PYTHONPATH=src:. pytest tests/

# 3. Retrain, so models/model.joblib reflects your change
PYTHONPATH=src:. python scripts/train.py   # or --model "MLP (neural network)", etc. — see --help
```

`models/model.joblib` is what the notebook, the app, and the report all
load — retraining is the one step that makes a model-code change visible
everywhere else.

## A real leakage trap in the generation-mix columns

`Production` and its 7 source breakdowns are recorded at the **same
hour** as `Consumption`, not before it. Using them directly to predict
that same hour's demand would be circular for a genuine forecasting
task — generation is dispatched *in response to* demand, not fixed in
advance, so a real system forecasting tomorrow's demand doesn't know
tomorrow's actual generation mix either. Every generation-mix feature
in `features.py` is lagged by 24 hours specifically to avoid this.

## Three real data-quality issues, and they are not the same issue

- A **Daylight Saving Time** artifact (1 missing hour every spring, 1
  duplicated hour every autumn) — the same underlying cause as any
  DST-observing country, verified directly.
- A **separate, unrelated real outage problem** concentrated in
  2024-2025: multi-hour gaps up to 83 consecutive hours, consistent
  with a genuine reporting/scraping gap in the live feed this dataset's
  maintainer pulls from. Not present at all in 2019-2023.
- **Fabricated readings in 2025-2026**: stretches of 2-14 hours where
  every column is a scaled-up copy of the hour before, so night-time
  "demand" climbs to 10,000-12,000 MW (the raw maximum, 11,986 MW at
  2025-11-27 06:00, is one of them) and nuclear output to ~2,080 MW,
  beyond the plant's capacity, before dropping back within an hour.
  `data.flag_implausible_readings` catches them with a conservative,
  documented rule (demand above 1.5x the same hour's median over the
  surrounding ±7 days, or nuclear above 1,500 MW, neither of which ever
  happens in 2019-2024; extended over the contiguous run of
  proportional copies) and `clean_frame` interpolates them like gaps:
  123 hours in 16 stretches on the full snapshot. Peak demand after
  cleaning is 9,615 MW (19 January 2021).

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

## Two baselines, and an honest answer about what the model adds

Numbers below are on the full snapshot after cleaning; cross-validation
is a 5-fold `TimeSeriesSplit` on the **training period only**, so it is
independent of the chronological test period (last 15% of hours,
February 2025 to March 2026).

| 1 hour ahead | CV MAPE (training period) | Test MAPE | Test MAE |
|---|---:|---:|---:|
| LightGBM | 3.23% | 4.05% | 239 MW |
| Random Forest | 3.28% | 4.66% | 272 MW |
| Linear Regression | 3.82% | 4.85% | 291 MW |
| MLP (neural network) | 7.94% | 4.42% | 259 MW |
| Persistence ("same as the previous hour") | 3.29% | 3.94% | 244 MW |
| Seasonal naive ("same hour last week") | 5.06% | 7.44% | 443 MW |

- `LightGBM` is the best learned model, ahead of Linear Regression,
  Random Forest and a lightweight `MLPRegressor` neural network (last
  and very unstable in cross-validation), and every model beats the
  seasonal naive baseline.
- **One hour ahead it does not beat persistence**: on the test period
  persistence has the lower MAPE and RMSE (314 vs 342 MW), LightGBM
  only the lower MAE. Call it a tie with the simplest possible rule.
- **The model's value is in multi-hour forecasting.** Forecasting 24
  hours recursively from 410 anchors across the test period, with a
  pipeline trained only on the training period, LightGBM averages
  4.44% MAPE over the 24 steps (3.94% at 1 hour, 4.82% at 19-24 hours),
  against 7.43% for "same hour last week" and 14.16% for "repeat the
  last known value", which ties the model at step 1 but cannot follow
  the daily cycle after that.

See `report/report.qmd`'s Methodology and 24-hour sections for the full
comparison.

## What this project deliberately doesn't do

No weather data, a different climate/market structure than Vietnam
(Romania's demand is winter-heating-dominant, likely the opposite shape
from Vietnam's), and the 2024-2025 test-period numbers are partly
measured against linearly-interpolated (not fully real) data — the
outage gaps and the removed fabricated stretches alike.
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
python -m ipykernel install --user --name energy-demand-forecasting --display-name "Energy Demand Forecasting"  # once
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

The notebook is written as a guide for a junior developer: 13 numbered parts
(setup and module map, the question, the data and its quirks, exploration,
features recomputed by hand for one timestamp, an honest chronological split
with leakage demonstrations, metrics and baselines, the models inside the
`Pipeline`, evaluation, recursive multi-step forecasting, SHAP, using the saved
model like the app, and a recap with exercises). Run it top to bottom on the
`energy-demand-forecasting` kernel; it takes about 3 minutes. It uses
`data/raw/Romania_hourly.csv` when present (falling back to the bundled sample
otherwise), never downloads anything, and never writes to `models/`, `data/`,
`app/` or `report/` — it only reads `models/model.joblib` in Part 11.

## Train from the command line

```bash
PYTHONPATH=src:. python scripts/train.py                    # default: LightGBM — the measured winner
PYTHONPATH=src:. python scripts/train.py --model "MLP (neural network)"
PYTHONPATH=src:. python scripts/train.py --help
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
autocorrelation checks), and **Model Insights** (seasonal naive
+ persistence baselines, model comparison cross-validated on the
training period, live SHAP). The Predict page's error-by-horizon chart
also shows both baselines at every step ahead.

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

`report.qmd` runs on a named Jupyter kernel (`energy-demand-forecasting`),
so register it once from this project's `.venv` before the first render:

```bash
python -m ipykernel install --user --name energy-demand-forecasting --display-name "Energy Demand Forecasting"
quarto render report/report.qmd
```

This regenerates both `report/report.html` and `report/report.pdf` (PDF
needs a LaTeX distribution — if you don't have one, run
`quarto install tinytex` once). Render just one format when you don't need
both:

```bash
quarto render report/report.qmd --to html
quarto render report/report.qmd --to pdf
```

Live-preview while editing (auto-rerenders on save):

```bash
quarto preview report/report.qmd
```

A `.qmd` file is Markdown prose plus fenced Python code chunks
(` ```{python} `/` ``` `), executed top to bottom by the kernel above, same
as a notebook cell. Common per-chunk options (a `#|` comment, first line of
the chunk): `#| echo: false` (hide this chunk's source code),
`#| output: false` (suppress its output, e.g. a setup/import cell),
`#| label: fig-foo` + `#| fig-cap: "..."` (name and caption a figure for
cross-referencing). The [Quarto VS Code
extension](https://marketplace.visualstudio.com/items?itemName=quarto.quarto)
adds syntax highlighting and a one-click Render button if you're doing more
than a one-line edit.

Troubleshooting:

| Symptom | Likely cause |
|---|---|
| `Jupyter engine failed ... kernel not found` | The `ipykernel install --name energy-demand-forecasting` step above hasn't been run yet. |
| `ModuleNotFoundError` inside a code chunk | `quarto render` runs with its working directory set to `report/`, not the project root — check the chunk's `sys.path.insert(0, "../src")` points at the right relative path. |
| Output looks stale after editing | Force a clean re-run: `quarto render report/report.qmd --execute-daemon-restart`. |
| PDF render fails, HTML succeeds | Missing LaTeX — run `quarto install tinytex` once, then retry. |

## Run the tests

```bash
PYTHONPATH=src:. pytest tests/
```

These test the DST/outage cleaning logic, the implausible-reading rule
(on synthetic scaled-copy nights), both baselines, that
cross-validation only touches the rows it is given, feature engineering
(including dedicated leakage checks for both the rolling-window and the
generation-mix features, and the seasonal-lookup's no-lookahead
property), and the chronological-split/model logic directly with
synthetic data — no download needed, and they already pass without any
real data.

## Deploy

The Streamlit app starts immediately from a bundled one-year (8,760-row) sample sourced from Kaggle. Set `USE_FULL_KAGGLE_DATA=true` to fetch and use the complete live dataset through the Kaggle API; configure either `KAGGLE_API_TOKEN` or a `[kaggle]` secrets section containing username and key.

- Repository: <https://github.com/nhamhung/energy-demand-forecasting>
- Report: <https://nhamhung.github.io/energy-demand-forecasting/>
- Beginner's guide: <https://nhamhung.github.io/energy-demand-forecasting/beginners-guide.html>
- Streamlit: <https://energy-demand-forecasting.streamlit.app>
- Fork setup: [docs/SETUP_AND_DEPLOYMENT.md](docs/SETUP_AND_DEPLOYMENT.md)
