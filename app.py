"""Wildfire Risk Calculator - Streamlit app.

Run locally with:  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import html

import altair as alt
import streamlit as st

from model import (
    ALERT_THRESHOLD,
    DEFAULTS,
    EXAMPLES,
    FEATURE_LABELS,
    FEATURES,
    MAX_DEPTH,
    N_TREES,
    cross_validated_auc,
    danger_level,
    feature_importances,
    load_data,
    main_signals,
    predict_day,
    train_model,
)

st.set_page_config(page_title="Wildfire Risk Calculator", page_icon="🔥", layout="wide")

NAVY = "#0b1f33"
NAVY_2 = "#14283b"
ORANGE = "#f05a28"
SAND = "#f6f3ed"
CALM = "#7898ad"

LEVEL_COLORS = {"Low": "#7898ad", "Moderate": "#e8b04a", "High": "#f0802f", "Extreme": "#d63c12"}


# --------------------------------------------------------------------------- model (cached)
@st.cache_resource(show_spinner="Training the forest on 40 years of weather...")
def get_model():
    """Load + clean the data and train the Random Forest. Runs once per server start."""
    df = load_data()
    model = train_model(df)
    return model, len(df)


@st.cache_resource(show_spinner="Computing cross-validated accuracy...")
def get_model_report():
    """Cross-validated ROC-AUC (5-fold) and feature importances. Runs once per server start."""
    df = load_data()
    model, _ = get_model()
    return cross_validated_auc(df), feature_importances(model)


# --------------------------------------------------------------------------- state
def _coerce(name, value):
    if name in ("max_temp", "min_temp"):
        return int(value)
    if name == "date":
        return value
    return float(value)


def apply_example(name: str) -> None:
    """on_click callback: runs BEFORE the widgets are rendered on the next rerun."""
    for key, value in EXAMPLES[name].items():
        st.session_state[key] = _coerce(key, value)


for _key, _value in DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _coerce(_key, _value)


# --------------------------------------------------------------------------- style
st.markdown(
    f"""
<style>
  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1300px; }}
  h1, h2, h3, h4 {{ color: {NAVY}; }}
  .app-title {{ font-size: 2.9rem; font-weight: 800; color: {NAVY}; line-height: 1.1; margin: 0 0 .4rem 0; }}
  .app-sub {{ font-size: 1.2rem; color: {NAVY_2}; opacity: .85; margin-bottom: 1.4rem; max-width: 60rem; }}
  .card {{ background: #ffffff; border-radius: 18px; padding: 1.6rem 1.8rem;
           box-shadow: 0 2px 14px rgba(11,31,51,.07); margin-bottom: 1.2rem; }}
  .risk-label {{ font-size: 1.3rem; font-weight: 600; color: {NAVY_2}; letter-spacing: .02em; }}
  .risk-big {{ font-size: 5.6rem; font-weight: 900; line-height: 1; margin: .2rem 0 .5rem 0; }}
  .risk-word {{ font-size: 2.2rem; font-weight: 700; color: {NAVY}; vertical-align: middle; }}
  .level-pill {{ display: inline-block; padding: .35rem 1.1rem; border-radius: 999px;
                 color: #fff; font-weight: 700; font-size: 1.35rem; }}
  .alert {{ border-radius: 14px; padding: 1.1rem 1.3rem; font-size: 1.6rem; font-weight: 800;
            margin: 1.2rem 0 .4rem 0; }}
  .alert-on {{ background: linear-gradient(90deg, #d63c12, {ORANGE}); color: #fff; }}
  .alert-off {{ background: #e3ebf1; color: #3e5b70; border: 2px solid {CALM}; }}
  .small-note {{ color: #50657a; font-size: .95rem; }}
  .gauge-wrap {{ position: relative; margin: 2.6rem 0 1.6rem 0; }}
  .gauge {{ display: flex; height: 26px; border-radius: 13px; overflow: hidden; }}
  .gauge div {{ height: 100%; }}
  .gauge-marker {{ position: absolute; top: -9px; width: 10px; height: 44px; margin-left: -5px;
                   background: {NAVY}; border-radius: 5px; border: 2px solid #fff;
                   box-shadow: 0 1px 6px rgba(11,31,51,.45); }}
  .gauge-threshold {{ position: absolute; top: -6px; height: 38px; border-left: 3px dashed {NAVY}; }}
  .gauge-threshold span {{ position: absolute; top: -22px; left: -30px; font-size: .85rem; font-weight: 800;
                           color: {NAVY}; white-space: nowrap; }}
  .gauge-labels {{ display: flex; font-size: .9rem; color: #50657a; margin-top: 4px; }}
  .gauge-labels div {{ text-align: center; }}
  .forest {{ display: grid; grid-template-columns: repeat(20, 1fr); gap: 3px; max-width: 520px; }}
  .forest div {{ aspect-ratio: 1 / 1; border-radius: 2px; }}
  .votes {{ font-size: 1.5rem; font-weight: 700; color: {NAVY}; margin-bottom: .6rem; }}
  .signals {{ font-size: 1.25rem; color: {NAVY_2}; margin-top: .4rem; }}
  .step-num {{ display: inline-block; width: 2.2rem; height: 2.2rem; border-radius: 50%; background: {ORANGE};
               color: #fff; text-align: center; line-height: 2.2rem; font-weight: 800; margin-right: .5rem; }}
  .step-title {{ font-size: 1.3rem; font-weight: 800; color: {NAVY}; }}
  .step-text {{ font-size: 1.05rem; color: {NAVY_2}; margin-top: .5rem; }}
  .footer {{ color: #50657a; font-size: .9rem; text-align: center; margin-top: 2.5rem; }}
  [data-testid="stWidgetLabel"] p {{ font-size: 1.12rem !important; font-weight: 600 !important;
                                     color: {NAVY_2} !important; }}
  [data-testid="stCaptionContainer"] p {{ font-size: .95rem !important; }}
  [data-testid="stExpander"] summary p {{ font-size: 1.25rem !important; font-weight: 700 !important;
                                          color: {NAVY} !important; }}
  [data-testid="stButton"] button p {{ font-weight: 700 !important; font-size: 1.05rem !important; }}
  [data-testid="stButton"] button {{ min-height: 3rem;
                                         border-radius: 12px; }}
  @media (max-width: 640px) {{
    .app-title {{ font-size: 2.1rem; }}
    .risk-big {{ font-size: 4.2rem; }}
    .risk-word {{ font-size: 1.6rem; display: block; }}
    .alert {{ font-size: 1.3rem; }}
  }}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- helpers
def f_to_c(f: float) -> float:
    return (f - 32) * 5 / 9


def mph_to_kmh(mph: float) -> float:
    return mph * 1.609344


def in_to_mm(inches: float) -> float:
    return inches * 25.4


def blend(c1: str, c2: str, t: float) -> str:
    t = max(0.0, min(1.0, float(t)))
    a = [int(c1[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def gauge_html(risk_pct: float) -> str:
    zones = [(15, "Low"), (15, "Moderate"), (30, "High"), (40, "Extreme")]
    bars = "".join(
        f'<div style="width:{w}%;background:{LEVEL_COLORS[name]};"></div>' for w, name in zones
    )
    labels = "".join(f'<div style="width:{w}%;">{name}</div>' for w, name in zones)
    pos = max(0.0, min(100.0, risk_pct))
    return f"""
<div class="gauge-wrap">
  <div class="gauge">{bars}</div>
  <div class="gauge-threshold" style="left:{ALERT_THRESHOLD}%;"><span>Alert {ALERT_THRESHOLD}%</span></div>
  <div class="gauge-marker" style="left:{pos}%;"></div>
  <div class="gauge-labels">{labels}</div>
</div>"""


def forest_html(tree_probs) -> str:
    cells = "".join(
        f'<div style="background:{blend("#d5dde3", ORANGE, p)};" title="Tree {i + 1}: {p:.0%}"></div>'
        for i, p in enumerate(tree_probs)
    )
    return f'<div class="forest">{cells}</div>'


# --------------------------------------------------------------------------- header
st.markdown('<div class="app-title">🔥 Will a wildfire start today?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">Enter the weather forecast for one day in California. Our Random Forest, '
    "trained on 1984-2023, gives the fire-start risk and says if crews should get ready.</div>",
    unsafe_allow_html=True,
)

st.markdown("**Try an example:**")
example_cols = st.columns(len(EXAMPLES))
for col, name in zip(example_cols, EXAMPLES):
    col.button(name, on_click=apply_example, args=(name,), use_container_width=True, key=f"ex_{name}")

model, n_rows = get_model()

st.write("")
left, right = st.columns([1, 1.25], gap="large")

# --------------------------------------------------------------------------- inputs
with left:
    st.markdown("### Weather forecast")
    st.date_input("Date", key="date", format="YYYY-MM-DD")
    st.slider("Max temperature of the day (°F)", 50, 106, step=1, key="max_temp")
    st.caption(f"= {f_to_c(st.session_state.max_temp):.1f} °C")
    st.slider("Min temperature of the night (°F)", 30, 80, step=1, key="min_temp")
    st.caption(f"= {f_to_c(st.session_state.min_temp):.1f} °C")
    if st.session_state.min_temp > st.session_state.max_temp:
        st.caption("Note: the night minimum is above the day maximum - please check the values.")
    st.slider("Average wind today (mph)", 0.0, 20.0, step=0.1, key="wind", format="%.1f")
    st.caption(f"= {mph_to_kmh(st.session_state.wind):.1f} km/h")
    st.slider("Rain today (in)", 0.00, 3.00, step=0.01, key="rain", format="%.2f")
    st.caption(f"= {in_to_mm(st.session_state.rain):.1f} mm")
    st.slider("Total rain over the last 7 days (in)", 0.00, 8.00, step=0.01, key="rain_7_days", format="%.2f")
    st.caption(f"= {in_to_mm(st.session_state.rain_7_days):.1f} mm")
    st.slider("Average wind over the last 7 days (mph)", 3.0, 14.0, step=0.1, key="wind_7_days", format="%.1f")
    st.caption(f"= {mph_to_kmh(st.session_state.wind_7_days):.1f} km/h")

ss = st.session_state
day = ss.date if isinstance(ss.date, dt.date) else DEFAULTS["date"]
prob, trees_fire, tree_probs = predict_day(
    model, day, ss.max_temp, ss.min_temp, ss.wind, ss.rain, ss.rain_7_days, ss.wind_7_days
)
risk_pct = round(prob * 100)
level = danger_level(prob * 100)
alert = prob * 100 >= ALERT_THRESHOLD

# --------------------------------------------------------------------------- output
with right:
    risk_color = ORANGE if alert else CALM
    alert_html = (
        '<div class="alert alert-on">🚒 FIRE ALERT - prepare crews</div>'
        if alert
        else '<div class="alert alert-off">✓ No alert</div>'
    )
    st.markdown(
        f"""
<div class="card">
  <div class="risk-big" style="color:{risk_color};"><span class="risk-word">Fire risk:</span> {risk_pct}%</div>
  <span class="level-pill" style="background:{LEVEL_COLORS[level]};">Danger level: {level}</span>
  {alert_html}
  <div class="small-note">Alert threshold 30%: in our tests it catches 83% of fire days,
  because a missed fire costs far more than a false alarm.</div>
  {gauge_html(prob * 100)}
  <div class="signals"><b>Main signals:</b> {html.escape(", ".join(main_signals(day, ss.min_temp, ss.rain, ss.rain_7_days)))}.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="card">
  <div class="risk-label">The forest votes</div>
  <div class="votes">{trees_fire} of {N_TREES} trees see a fire</div>
  {forest_html(tree_probs)}
  <div class="small-note" style="margin-top:.8rem;">Each square is one tree of the Random Forest.
  The risk is the average of their answers.</div>
</div>
""",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- how it works
st.write("")
st.markdown("## How it works")
steps = [
    (
        "Learn from 40 years",
        f"{n_rows:,} labelled days of NOAA weather and CAL FIRE fire starts, 1984 to Dec 2023. "
        "No year variable, so no data leakage.",
    ),
    (
        f"{N_TREES} trees vote",
        "Each tree asks yes/no questions about the weather (warm night? rain this week?) "
        "and gives a probability; the forest averages them.",
    ),
    ("Alert at 30%", "From 30% risk we alert."),
]
for i, (col, (title, text)) in enumerate(zip(st.columns(3, gap="large"), steps), start=1):
    col.markdown(
        f'<div class="card"><span class="step-num">{i}</span><span class="step-title">{title}</span>'
        f'<div class="step-text">{text}</div></div>',
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- about the model
with st.expander("About the model"):
    auc, importances = get_model_report()
    st.markdown(
        f"""
- **Model:** Random Forest, {N_TREES} trees, max depth {MAX_DEPTH}, trained on {n_rows:,} days.
- **Features used:** {", ".join(FEATURE_LABELS[f] for f in FEATURES)}.
  Temperature range, wind/temperature ratio, month and day of the year are computed from your inputs.
- **Cross-validated ROC-AUC (5-fold):** **{auc:.3f}**
"""
    )
    chart = (
        alt.Chart(importances)
        .mark_bar(color=ORANGE, cornerRadiusEnd=4)
        .encode(
            x=alt.X("importance:Q", title="Importance", axis=alt.Axis(format="%", tickCount=6)),
            y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260, labelFontSize=14)),
            tooltip=[alt.Tooltip("label:N", title="Feature"), alt.Tooltip("importance:Q", format=".1%")],
        )
        .properties(height=380, title="What the forest looks at most")
        .configure_title(fontSize=16, color=NAVY)
    )
    st.altair_chart(chart, use_container_width=True)
    st.markdown(
        "**Limit:** Winter fires driven by Santa Ana winds are often missed. "
        "The model advises, humans decide."
    )

st.markdown(
    '<div class="footer">Data: Yavas et al. (2025), California Weather and Fire Prediction Dataset '
    "(1984-2025), Georgia Southern University (NOAA + CAL FIRE). "
    "AI for Business &amp; Society, SKEMA Business School.</div>",
    unsafe_allow_html=True,
)
