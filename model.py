"""Data loading, training and prediction logic for the Wildfire Risk Calculator.

Kept free of Streamlit so it can be imported by the tests.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

_HERE = Path(__file__).parent
# Linux file names are case-sensitive: accept data.xlsx or Data.xlsx.
DATA_PATH = next((p for p in (_HERE / "data.xlsx", _HERE / "Data.xlsx") if p.exists()), _HERE / "data.xlsx")
SHEET_NAME = "CA_Weather_Fire_Dataset_1984-20"
# Fire starts are not recorded after this date: later days are wrongly labelled "no fire".
LAST_LABELLED_DATE = pd.Timestamp("2023-12-22")

FEATURES = [
    "PRECIPITATION",
    "MAX_TEMP",
    "MIN_TEMP",
    "AVG_WIND_SPEED",
    "TEMP_RANGE",
    "WIND_TEMP_RATIO",
    "MONTH",
    "LAGGED_PRECIPITATION",
    "LAGGED_AVG_WIND_SPEED",
    "DAY_OF_YEAR",
]
TARGET = "FIRE_START_DAY"

FEATURE_LABELS = {
    "PRECIPITATION": "Rain today",
    "MAX_TEMP": "Max temperature (day)",
    "MIN_TEMP": "Min temperature (night)",
    "AVG_WIND_SPEED": "Wind today",
    "TEMP_RANGE": "Temperature range",
    "WIND_TEMP_RATIO": "Wind / temperature ratio",
    "MONTH": "Month",
    "LAGGED_PRECIPITATION": "Rain, last 7 days",
    "LAGGED_AVG_WIND_SPEED": "Wind, last 7 days",
    "DAY_OF_YEAR": "Day of the year",
}

N_TREES = 300
MAX_DEPTH = 12
# Each leaf must hold at least 20 real days. Without this, rare combinations (e.g. a very
# calm week in February, which never happened in 40 years) end up in leaves built on a
# handful of days and get absurd risks (48% instead of ~20%). CV AUC also improves slightly.
MIN_SAMPLES_LEAF = 20


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and clean the dataset: dropna, then keep only labelled days (<= 2023-12-22)."""
    df = pd.read_excel(path, sheet_name=SHEET_NAME)
    df = df.dropna()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df[df["DATE"] <= LAST_LABELLED_DATE].copy()
    df[TARGET] = _to_bool(df[TARGET]).astype(int)
    return df.reset_index(drop=True)


def make_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=42,
        n_jobs=-1,
    )


def train_model(df: pd.DataFrame) -> RandomForestClassifier:
    model = make_model()
    model.fit(df[FEATURES], df[TARGET])
    return model


def cross_validated_auc(df: pd.DataFrame) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(make_model(), df[FEATURES], df[TARGET], cv=cv, scoring="roc_auc")
    return float(scores.mean())


def feature_importances(model: RandomForestClassifier) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
    imp["label"] = imp["feature"].map(FEATURE_LABELS)
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def build_features(date, max_temp, min_temp, wind, rain, rain_7_days, wind_7_days) -> pd.DataFrame:
    """One-row DataFrame with the exact feature names and order used in training."""
    if isinstance(date, pd.Timestamp):
        date = date.date()
    if isinstance(date, dt.datetime):
        date = date.date()
    max_temp = float(max_temp)
    row = {
        "PRECIPITATION": float(rain),
        "MAX_TEMP": max_temp,
        "MIN_TEMP": float(min_temp),
        "AVG_WIND_SPEED": float(wind),
        "TEMP_RANGE": max_temp - float(min_temp),
        "WIND_TEMP_RATIO": float(wind) / max_temp if max_temp != 0 else 0.0,
        "MONTH": date.month,
        "LAGGED_PRECIPITATION": float(rain_7_days),
        "LAGGED_AVG_WIND_SPEED": float(wind_7_days),
        "DAY_OF_YEAR": date.timetuple().tm_yday,
    }
    return pd.DataFrame([row], columns=FEATURES)


def predict_day(model, date, max_temp, min_temp, wind, rain, rain_7_days, wind_7_days):
    """Return (probability of fire 0-1, number of trees predicting fire, per-tree probabilities)."""
    X = build_features(date, max_temp, min_temp, wind, rain, rain_7_days, wind_7_days)
    probability = float(model.predict_proba(X)[0, 1])
    # Individual trees are fitted on plain arrays inside the forest, so give them arrays
    # (float32, as the forest does internally) to avoid feature-name warnings.
    X_arr = X.to_numpy(dtype=np.float32)
    tree_probs = np.array(
        [est.predict_proba(X_arr, check_input=False)[0, 1] for est in model.estimators_]
    )
    trees_fire = int((tree_probs >= 0.5).sum())
    return probability, trees_fire, tree_probs


def danger_level(risk_pct: float) -> str:
    if risk_pct < 15:
        return "Low"
    if risk_pct < 30:
        return "Moderate"
    if risk_pct < 60:
        return "High"
    return "Extreme"


ALERT_THRESHOLD = 30  # percent


def main_signals(date, min_temp, rain, rain_7_days) -> list[str]:
    signals = []
    if min_temp >= 60:
        signals.append("warm night")
    elif min_temp <= 50:
        signals.append("cool night")
    if rain_7_days >= 0.5 or rain >= 0.2:
        signals.append("recent rain")
    else:
        signals.append("dry week")
    if 6 <= date.month <= 10:
        signals.append("fire season")
    if date.month in (12, 1, 2):
        signals.append("winter: the model sees few winter fires")
    return signals


EXAMPLES = {
    "Heatwave in August": dict(
        date=dt.date(2026, 8, 15), max_temp=92, min_temp=66, wind=8.0,
        rain=0.00, rain_7_days=0.00, wind_7_days=7.5,
    ),
    "Mild spring day": dict(
        date=dt.date(2026, 4, 10), max_temp=70, min_temp=50, wind=9.0,
        rain=0.00, rain_7_days=8 / 25.4, wind_7_days=9.0,  # 8 mm this week
    ),
    "Rainy winter day": dict(
        date=dt.date(2026, 1, 20), max_temp=58, min_temp=45, wind=10.0,
        rain=30 / 25.4, rain_7_days=75 / 25.4, wind_7_days=9.0,  # 30 mm today, 75 mm this week
    ),
    "Dry, windy January": dict(
        date=dt.date(2026, 1, 7), max_temp=70, min_temp=50, wind=14.0,
        rain=0.00, rain_7_days=0.00, wind_7_days=12.0,
    ),
}
DEFAULTS = EXAMPLES["Heatwave in August"]


# --------------------------------------------------------------------------- economics
# Cost of one alert (= cost of a false alarm): a strike team of 5 Type III engines on standby
# for 24 h, Cal OES 2025 rates: engine $156.74/h + 4 firefighters at $44.97/h overtime.
ENGINE_RATE = 156.74
FIREFIGHTER_RATE = 44.97
STANDBY_COST = 5 * 24 * (ENGINE_RATE + 4 * FIREFIGHTER_RATE)  # = $40,394.40
# Default cost of a fire that starts without crews being ready: average firefighting cost per
# fire in California in 2018 (~$1B of suppression for ~8,500 fires). Conservative: no damages.
DEFAULT_MISSED_FIRE_COST = 120_000


def decision_economics(prob: float, missed_fire_cost: float = DEFAULT_MISSED_FIRE_COST,
                       standby_cost: float = STANDBY_COST) -> dict:
    """Money at stake for one day, if we alert vs if we do not.

    Alerting costs `standby_cost` whatever happens. If a fire starts, the alert avoids the cost of
    a fire nobody was ready for, so the saving is `missed_fire_cost - standby_cost`.
    """
    return {
        "cost_if_no_fire": standby_cost,                        # false alarm
        "saved_if_fire": missed_fire_cost - standby_cost,       # fire caught in time
        "expected_value_of_alert": prob * missed_fire_cost - standby_cost,
        "expected_loss_without_alert": prob * missed_fire_cost,
        "break_even_risk": standby_cost / missed_fire_cost,     # alerting pays off above this risk
    }
