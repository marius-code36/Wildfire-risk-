# 🔥 Wildfire Risk Calculator

**Will a wildfire start today?** Enter the weather forecast for one day in California and a
Random Forest trained on 40 years of data (1984–2023) gives the probability that a wildfire
starts that day, a danger level, and whether fire crews should get ready.

Built for the course *AI for Business & Society* at SKEMA Business School.

## What the app shows

- **Fire risk** in very large type (e.g. *Fire risk: 79%*) and a danger level:
  Low (<15%), Moderate (15–30%), High (30–60%), Extreme (≥60%).
- **Alert box**: from 30% risk, *FIRE ALERT – prepare crews*.
- **Gauge** from 0 to 100% with the four zones and the 30% alert marker.
- **The forest votes**: 300 squares, one per tree, coloured from grey (calm) to orange (sees a fire).
- **Main signals** in plain English (warm night, dry week, fire season…).
- Four **example buttons** that fill every input in one click:
  Heatwave in August, Mild spring day, Rainy winter day, Dry windy January.
- **How it works** and **About the model** (cross-validated ROC-AUC, feature importances, limits).

**Flow:** (1) enter the weather or click an example, then **Predict the fire risk**;
(2) a short animation shows the 300 trees voting one row at a time; (3) the result screen shows
the risk, the alert and the forest's vote, with **Change the weather** to go back (values are kept).

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501. The model trains once at start-up (a few seconds) and is cached.

Run the tests:

```bash
python -m pytest -s tests
```

## How it was built

**Data cleaning** (`data.xlsx`, sheet `CA_Weather_Fire_Dataset_1984-20`)
1. Drop rows with missing values: 14,988 → 14,976 rows.
2. Keep only days up to **22 Dec 2023**: fire starts are not recorded after that date, so later
   days would be wrongly labelled "no fire" (label leakage). Result: **14,598 labelled days**,
   about one in three is a fire-start day.

**Features** (10) — the user enters 7 values, the app computes the rest:

| Feature | Meaning |
|---|---|
| PRECIPITATION | rain today (entered in mm on a stepped scale, converted to inches) |
| MAX_TEMP / MIN_TEMP | day maximum / night minimum (°F) |
| AVG_WIND_SPEED | average wind today (mph) |
| TEMP_RANGE | MAX_TEMP − MIN_TEMP (computed) |
| WIND_TEMP_RATIO | AVG_WIND_SPEED / MAX_TEMP (computed) |
| MONTH, DAY_OF_YEAR | from the date (computed) |
| LAGGED_PRECIPITATION | total rain over the last 7 days, today included (entered in mm, converted to inches) |
| LAGGED_AVG_WIND_SPEED | average wind over the last 7 days, today included (mph) |

YEAR, SEASON and DATE are deliberately **not** used, so the model cannot learn trends tied to a
particular year.

**Model**: `RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20, random_state=42)`, trained on
all 14,598 days. Quality is measured with 5-fold stratified cross-validation (ROC-AUC ≈ 0.84,
the exact value is shown in the app). The night temperature (MIN_TEMP) is the most important feature.
Each leaf must contain at least 20 real days (`min_samples_leaf=20`): without it, rare
combinations that never occurred in the data (e.g. a very calm week in February) got absurd
risks of almost 50%.

**Alert threshold: 30%.** In our tests this threshold catches 83% of fire days. It is set low on
purpose: a missed fire costs far more than a false alarm.

**Limit**: winter fires driven by Santa Ana winds are often missed. The model advises, humans decide.

## Files

| File | Role |
|---|---|
| `app.py` | Streamlit interface |
| `model.py` | data loading, cleaning, training, `predict_day()` |
| `tests/test_model.py` | data, model and app tests (incl. `streamlit.testing` AppTest) |
| `.streamlit/config.toml` | colour theme |
| `data.xlsx` | dataset |

## Data source

Yavas, C. E., Kadlec, C., & Kim, J. (2025). *California Weather and Fire Prediction Dataset
(1984–2025) with Engineered Features*. Georgia Southern University. Weather from NOAA Climate Data
Online, fire starts from CAL FIRE. https://zenodo.org/records/14712845
