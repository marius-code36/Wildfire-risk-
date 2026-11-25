"""Tests for the wildfire model and the Streamlit app.

Run with:  python -m pytest -s tests   (or simply: python tests/test_model.py)
"""
from __future__ import annotations

import re
import sys
import warnings
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import EXAMPLES, FEATURES, cross_validated_auc, load_data, predict_day, train_model  # noqa: E402

EXPECTED_RANGES = {  # percent, inclusive
    "Heatwave in August": (75, 85),
    "Mild spring day": (10, 15),
    "Rainy winter day": (0, 2),
    "Dry, windy January": (10, 25),
}


@lru_cache(maxsize=1)
def data():
    return load_data()


@lru_cache(maxsize=1)
def model():
    return train_model(data())


def test_row_count():
    df = data()
    print(f"\nCleaned rows: {len(df):,}")
    assert len(df) == 14_598
    assert df["DATE"].max() <= __import__("pandas").Timestamp("2023-12-22")
    assert set(df["FIRE_START_DAY"].unique()) <= {0, 1}
    rate = df["FIRE_START_DAY"].mean()
    print(f"Share of fire days: {rate:.1%}")
    assert 0.25 < rate < 0.42


def test_examples_in_expected_ranges():
    m = model()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any sklearn feature-name warning fails the test
        for name, inputs in EXAMPLES.items():
            prob, trees_fire, tree_probs = predict_day(m, **inputs)
            pct = prob * 100
            lo, hi = EXPECTED_RANGES[name]
            print(f"{name:<22} risk = {pct:5.1f}%   trees seeing fire = {trees_fire:3d}/300")
            assert len(tree_probs) == 300
            assert 0 <= trees_fire <= 300
            assert abs(tree_probs.mean() - prob) < 1e-6
            assert lo <= round(pct) <= hi, f"{name}: {pct:.1f}% not in [{lo}, {hi}]"


def test_cross_validated_auc():
    auc = cross_validated_auc(data())
    print(f"\nCross-validated ROC-AUC (5-fold): {auc:.4f}")
    assert auc > 0.80


def test_feature_importance_night_temp_first():
    m = model()
    top = FEATURES[int(m.feature_importances_.argmax())]
    print(f"\nMost important feature: {top}")
    assert top == "MIN_TEMP"


def test_no_absurd_winter_risk():
    """Regression: February, 6 °C night / 18 °C day, very calm week used to give 48%."""
    import datetime as dt
    m = model()
    for wind_7_days in (3.0, 4.0, 4.5, 5.0, 7.5):
        prob, _, _ = predict_day(m, dt.date(2026, 2, 15), 64, 43, 8.0, 0.0, 0.0, wind_7_days)
        print(f"Feb 15, 43/64 °F, 7-day wind {wind_7_days:>4} mph -> {prob:.0%}")
        assert prob < 0.30


def _risk_shown(at) -> int:
    for md in at.markdown:
        found = re.search(r"Fire risk:</span>\s*(\d+)%", md.value)
        if found:
            return int(found.group(1))
    raise AssertionError("risk not displayed")


def test_app_buttons():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    assert not at.exception, at.exception
    print(f"\nApp default risk shown: {_risk_shown(at)}%")
    for i, name in enumerate(EXAMPLES):
        at.button(key=f"ex_{name}").click().run()
        assert not at.exception, at.exception
        shown = _risk_shown(at)
        lo, hi = EXPECTED_RANGES[name]
        alert = any("FIRE ALERT" in md.value for md in at.markdown)
        print(f"Clicked '{name}': risk shown = {shown}%  alert = {alert}")
        assert lo <= shown <= hi
        assert alert == (shown >= 30)
    # moving a slider updates the result instantly
    at.button(key="ex_Heatwave in August").click().run()
    before = _risk_shown(at)
    at.select_slider(key="rain_7_days").set_value(75).run()
    assert not at.exception
    after = _risk_shown(at)
    print(f"Heatwave + 75 mm of rain this week: {before}% -> {after}%")
    assert after < before
    # rain is entered in mm on fixed steps; 4 mm must be selectable
    at.select_slider(key="rain").set_value(4).run()
    assert not at.exception
    assert any(c.value == "= 0.16 in" for c in at.caption)


if __name__ == "__main__":
    test_row_count()
    test_examples_in_expected_ranges()
    test_cross_validated_auc()
    test_feature_importance_night_temp_first()
    test_no_absurd_winter_risk()
    test_app_buttons()
    print("\nALL TESTS PASSED")
