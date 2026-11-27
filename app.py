"""Wildfire Risk Calculator - Streamlit app.

Run locally with:  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import html
import time

import altair as alt
import streamlit as st
import streamlit.components.v1 as components

from model import (
    ALERT_THRESHOLD,
    DEFAULT_MISSED_FIRE_COST,
    DEFAULTS,
    EXAMPLES,
    FEATURE_LABELS,
    FEATURES,
    MAX_DEPTH,
    MIN_SAMPLES_LEAF,
    N_TREES,
    STANDBY_COST,
    cross_validated_auc,
    danger_level,
    decision_economics,
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

ANIMATION_SECONDS = 2.6  # length of the "forest is voting" animation

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
# Rain is entered in millimetres on a stepped scale: fine steps for the small amounts that
# are most common, wider steps for heavy rain. The model itself works in inches.
MM_PER_INCH = 25.4
RAIN_TODAY_MM = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75]
RAIN_WEEK_MM = [0, 1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200]
RAIN_STEPS = {"rain": RAIN_TODAY_MM, "rain_7_days": RAIN_WEEK_MM}


def _nearest(options, value):
    return min(options, key=lambda o: abs(o - value))


def _coerce(name, value):
    if name in ("max_temp", "min_temp"):
        return int(value)
    if name == "date":
        return value
    if name in RAIN_STEPS:  # inches -> nearest millimetre step
        return _nearest(RAIN_STEPS[name], float(value) * MM_PER_INCH)
    return float(value)


INPUT_KEYS = list(DEFAULTS)
ss = st.session_state

# Two screens: "form" (enter the weather) -> short forest animation -> "result".
# Widgets that are not displayed lose their value in Streamlit, so the entered weather is
# also kept in ss.saved and copied back into the widgets when returning to the form.
if "saved" not in ss:
    ss.saved = {k: _coerce(k, v) for k, v in DEFAULTS.items()}
    ss.page = "form"
    ss.animate = False


def apply_example(name: str) -> None:
    """on_click callback: runs BEFORE the widgets are rendered on the next rerun."""
    for key, value in EXAMPLES[name].items():
        ss[key] = _coerce(key, value)
        ss.saved[key] = ss[key]


def go_predict() -> None:
    ss.saved = {k: ss[k] for k in INPUT_KEYS}
    ss.page = "result"
    ss.animate = True


def go_back() -> None:
    for k in INPUT_KEYS:
        ss[k] = ss.saved[k]
    ss.page = "form"


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
  .steps {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.1rem; align-items: stretch;
             margin-bottom: 1.6rem; }}
  .steps .card {{ margin-bottom: 0; }}
  .step-head {{ display: flex; align-items: center; }}
  @media (max-width: 1100px) {{ .steps {{ grid-template-columns: repeat(2, 1fr); }} }}
  @media (max-width: 640px) {{ .steps {{ grid-template-columns: 1fr; }} }}
  .step-num {{ flex: 0 0 auto; display: inline-block; width: 2.2rem; height: 2.2rem; border-radius: 50%; background: {ORANGE};
               color: #fff; text-align: center; line-height: 2.2rem; font-weight: 800; margin-right: .5rem; }}
  .step-title {{ font-size: 1.22rem; line-height: 1.2; font-weight: 800; color: {NAVY}; }}
  .step-text {{ font-size: 1.05rem; color: {NAVY_2}; margin-top: .5rem; }}
  .lvls {{ display: flex; flex-direction: column; gap: .55rem; }}
  .lvl {{ display: flex; gap: .55rem; align-items: flex-start; font-size: .98rem; line-height: 1.3; }}
  .lvl-dot {{ flex: 0 0 auto; width: .95rem; height: .95rem; border-radius: 50%; margin-top: .2rem; }}
  .lvl-range {{ color: #50657a; font-size: .9rem; }}
  .econ {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }}
  .econ-tile {{ border-radius: 14px; padding: 1.1rem 1.2rem; }}
  .econ-head {{ font-size: 1rem; font-weight: 700; opacity: .9; }}
  .econ-value {{ font-size: 2.3rem; font-weight: 900; line-height: 1.15; margin: .3rem 0 .4rem 0; }}
  .econ-text {{ font-size: .95rem; line-height: 1.35; opacity: .92; }}
  .econ-cost {{ background: #ece6dc; color: {NAVY}; }}
  .econ-save {{ background: {ORANGE}; color: #fff; }}
  .econ-avg {{ background: {NAVY}; color: #fff; }}
  @media (max-width: 900px) {{ .econ {{ grid-template-columns: 1fr; }} }}
  [data-testid="stIFrame"], iframe[height="0"] {{ display: block; height: 0 !important; border: 0; }}
  .element-container:has(iframe[height="0"]), [data-testid="stElementContainer"]:has(iframe[height="0"]) {{
      margin: 0 !important; height: 0 !important; }}
  .footer {{ color: #50657a; font-size: .9rem; text-align: center; margin-top: 2.5rem; }}
  [data-testid="stWidgetLabel"] p {{ font-size: 1.12rem !important; font-weight: 600 !important;
                                     color: {NAVY_2} !important; }}
  [data-testid="stCaptionContainer"] p {{ font-size: .95rem !important; }}
  [data-testid="stExpander"] summary p {{ font-size: 1.25rem !important; font-weight: 700 !important;
                                          color: {NAVY} !important; }}
  [data-testid="stButton"] button p {{ font-weight: 700 !important; font-size: 1.05rem !important; }}
  [data-testid="stButton"] button {{ min-height: 3rem;
                                         border-radius: 12px; }}
  [data-testid="stBaseButton-primary"] {{ min-height: 4.2rem !important; border-radius: 16px !important;
                                          box-shadow: 0 4px 16px rgba(240,90,40,.35); }}
  [data-testid="stButton"] button[data-testid="stBaseButton-primary"] p {{ font-size: 1.5rem !important; font-weight: 800 !important; }}
  .section-title {{ font-size: 1.6rem; font-weight: 800; color: {NAVY}; margin: .2rem 0 .8rem 0; }}
  .overlay {{ position: fixed; inset: 0; z-index: 999; background: {SAND};
              display: flex; align-items: center; justify-content: center; padding: 1rem; }}
  .overlay .card {{ width: min(760px, 100%); margin: 0; }}
  .loading {{ text-align: center; padding: 2.4rem 1.5rem; }}
  .loading-title {{ font-size: 2.2rem; font-weight: 800; color: {NAVY}; }}
  .loading-sub {{ font-size: 1.2rem; color: #50657a; margin: .4rem 0 1.4rem 0; }}
  .loading .forest {{ margin: 0 auto; max-width: 560px; }}
  .loading-bar {{ height: 10px; border-radius: 5px; background: #e6e1d8; overflow: hidden;
                  max-width: 560px; margin: 1.4rem auto 0 auto; }}
  .loading-bar div {{ height: 100%; background: {ORANGE}; }}
  .tree-empty {{ background: #efebe4; }}
  .weather-recap {{ font-size: 1.15rem; color: {NAVY_2}; background: #ffffff; border-radius: 14px;
                    padding: .9rem 1.2rem; box-shadow: 0 2px 10px rgba(11,31,51,.06); }}
  .weather-recap b {{ color: {NAVY}; }}
  @media (max-width: 640px) {{
    .loading-title {{ font-size: 1.6rem; }}
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


def forest_loading_html(tree_probs, shown: int) -> str:
    cells = "".join(
        f'<div style="background:{blend("#d5dde3", ORANGE, p)};"></div>' if i < shown
        else '<div class="tree-empty"></div>'
        for i, p in enumerate(tree_probs)
    )
    pct = 100 * shown / len(tree_probs)
    return f"""
<div class="card loading">
  <div class="loading-title">🌲 The forest is voting...</div>
  <div class="loading-sub">{shown} / {len(tree_probs)} trees have given their answer</div>
  <div class="forest">{cells}</div>
  <div class="loading-bar"><div style="width:{pct:.0f}%;"></div></div>
</div>"""


def weather_recap_html(w: dict) -> str:
    return (
        '<div class="weather-recap">'
        f"📅 <b>{w['date']:%d %B %Y}</b> &nbsp;·&nbsp; "
        f"🌡️ <b>{w['max_temp']}°F</b> day / <b>{w['min_temp']}°F</b> night "
        f"({f_to_c(w['max_temp']):.0f} °C / {f_to_c(w['min_temp']):.0f} °C) &nbsp;·&nbsp; "
        f"💨 <b>{w['wind']:.1f} mph</b> today, {w['wind_7_days']:.1f} mph over 7 days &nbsp;·&nbsp; "
        f"🌧️ <b>{w['rain']} mm</b> today, {w['rain_7_days']} mm over 7 days"
        "</div>"
    )


# --------------------------------------------------------------------------- how it works + about
RESPONSE_LEVELS = [
    ("Low", "below 15%", "normal operations"),
    ("Moderate", "15-30%", "closer monitoring, no alert"),
    ("High", "30-60%", "alert: crews ready to intervene"),
    ("Extreme", "60% and more", "alert: reinforcements mobilised"),
]


def levels_html() -> str:
    rows = "".join(
        f'<div class="lvl"><span class="lvl-dot" style="background:{LEVEL_COLORS[name]};"></span>'
        f"<div><b>{name}</b> <span class=\"lvl-range\">{rng}</span><br>{what}</div></div>"
        for name, rng, what in RESPONSE_LEVELS
    )
    return f'<div class="lvls">{rows}</div>'


def render_info() -> None:
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
        ("Graduated response", levels_html()),
    ]
    cards = "".join(
        f'<div class="card step-card"><div class="step-head"><span class="step-num">{i}</span>'
        f'<span class="step-title">{title}</span></div><div class="step-text">{text}</div></div>'
        for i, (title, text) in enumerate(steps, start=1)
    )
    st.markdown(f'<div class="steps">{cards}</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------- about the model
    with st.expander("About the model"):
        auc, importances = get_model_report()
        st.markdown(
            f"- **Model:** Random Forest, {N_TREES} trees, max depth {MAX_DEPTH}, "
            f"at least {MIN_SAMPLES_LEAF} days per leaf, trained on {n_rows:,} days.\n"
            f"- **Features used:** {', '.join(FEATURE_LABELS[f] for f in FEATURES)}. "
            "Temperature range, wind/temperature ratio, month and day of the year "
            "are computed from your inputs.\n"
            f"- **Cross-validated ROC-AUC (5-fold):** **{auc:.3f}**"
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


# --------------------------------------------------------------------------- header
st.markdown('<div class="app-title">🔥 Will a wildfire start today?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">Enter the weather forecast for one day in California. Our Random Forest, '
    "trained on 1984-2023, gives the fire-start risk and says if crews should get ready.</div>",
    unsafe_allow_html=True,
)

model, n_rows = get_model()

# =========================================================================== economics card
def money(x: float, sign: bool = False) -> str:
    s = f"${abs(x):,.0f}"
    if sign:
        return ("+" if x >= 0 else "−") + s
    return ("−" if x < 0 else "") + s


def remember_fire_cost() -> None:
    ss.missed_fire_cost_saved = ss.missed_fire_cost


def tile(kind: str, head: str, value: str, text: str) -> str:
    return (f'<div class="econ-tile econ-{kind}"><div class="econ-head">{head}</div>'
            f'<div class="econ-value">{value}</div><div class="econ-text">{text}</div></div>')


def render_economics(prob: float, alert: bool) -> None:
    """What is at stake today: cost of the alert if no fire, money saved if a fire starts."""
    if "missed_fire_cost" not in ss:  # keep the chosen value across screens
        ss.missed_fire_cost = ss.get("missed_fire_cost_saved", DEFAULT_MISSED_FIRE_COST)
    fire_cost = float(ss.missed_fire_cost)
    e = decision_economics(prob, fire_cost)
    pct = f"{prob:.0%}"
    be = f"{e['break_even_risk']:.0%}"

    if alert:
        title = "💰 What does this alert cost, and what can it save?"
        tiles = [
            tile("cost", "If no fire starts (false alarm)", money(-e["cost_if_no_fire"]),
                 "A strike team of 5 engines on standby for 24 h, for nothing."),
            tile("save", "If a fire does start", f"{money(e['saved_if_fire'], sign=True)} saved",
                 f"Crews are ready: we avoid a {money(fire_cost)} fire nobody was prepared for, "
                 f"minus the {money(STANDBY_COST)} standby cost."),
        ]
        ev = e["expected_value_of_alert"]
        verdict = (f"Alerting pays off on average from {be} risk."
                   if ev >= 0 else
                   f"Just below the {be} break-even: we still alert, because {money(fire_cost)} "
                   "counts firefighting only, not burnt homes and land.")
        tiles.append(tile("avg", "On average, for a day like today", money(ev, sign=True),
                          f"{pct} × {money(fire_cost)} − {money(STANDBY_COST)}. {verdict}"))
    else:
        title = "💰 No alert: what is at stake?"
        loss = e["expected_loss_without_alert"]
        tiles = [
            tile("save", "No crews mobilised", "$0 spent",
                 f"We save the {money(STANDBY_COST)} that a strike team on standby would cost."),
            tile("cost", "If a fire starts anyway", money(-fire_cost),
                 "A fire nobody was prepared for (firefighting cost only)."),
        ]
        verdict = (f"Less than the {money(STANDBY_COST)} an alert would cost: not alerting is the cheaper choice."
                   if loss < STANDBY_COST else
                   f"More than the {money(STANDBY_COST)} an alert would cost: alerting pays off from {be} risk.")
        tiles.append(tile("avg", "Expected loss, for a day like today", money(-loss),
                          f"{pct} × {money(fire_cost)}. {verdict}"))

    st.markdown(
        f'<div class="card"><div class="risk-label" style="margin-bottom:.9rem;">{title}</div>'
        f'<div class="econ">{"".join(tiles)}</div></div>',
        unsafe_allow_html=True,
    )
    c1, _ = st.columns([1.3, 1])
    with c1:
        st.number_input(
            "Assumption: cost of a fire nobody was prepared for ($)",
            min_value=10_000, max_value=10_000_000, step=10_000, format="%d",
            key="missed_fire_cost", on_change=remember_fire_cost,
        )
    st.caption((
        f"Illustrative estimate. Alert cost: Cal OES 2025 rates (Type III engine $156.74/h + "
        f"4 firefighters at $44.97/h overtime) × 5 engines × 24 h = {money(STANDBY_COST)}. "
        f"Default fire cost {money(DEFAULT_MISSED_FIRE_COST)}: average firefighting cost per fire "
        "in California in 2018 (≈ $1B for ≈ 8,500 fires), damages not included."
    ).replace("$", "\\$"))  # "$...$" would otherwise be read as a math formula


# =========================================================================== screens
def render_form_screen() -> None:
    for k in INPUT_KEYS:  # restore values after coming back from the result screen
        if k not in ss:
            ss[k] = ss.saved[k]

    st.markdown("**Try an example:**")
    example_cols = st.columns(len(EXAMPLES))
    for col, name in zip(example_cols, EXAMPLES):
        col.button(name, on_click=apply_example, args=(name,), use_container_width=True, key=f"ex_{name}")

    st.write("")
    _, center, _ = st.columns([1, 2.2, 1])  # one centred column, comfortable to read
    with center:
        st.markdown('<div class="section-title">Weather forecast</div>', unsafe_allow_html=True)
        st.date_input("Date", key="date", format="YYYY-MM-DD")
        st.slider("Max temperature of the day (°F)", 50, 106, step=1, key="max_temp")
        st.caption(f"= {f_to_c(ss.max_temp):.1f} °C")
        st.slider("Min temperature of the night (°F)", 30, 80, step=1, key="min_temp")
        st.caption(f"= {f_to_c(ss.min_temp):.1f} °C")
        if ss.min_temp > ss.max_temp:
            st.caption("Note: the night minimum is above the day maximum - please check the values.")
        st.slider("Average wind today (mph)", 0.0, 20.0, step=0.1, key="wind", format="%.1f")
        st.caption(f"= {mph_to_kmh(ss.wind):.1f} km/h")
        st.select_slider("Rain today (mm)", options=RAIN_TODAY_MM, key="rain", format_func=lambda v: f"{v} mm")
        st.caption(f"= {ss.rain / MM_PER_INCH:.2f} in")
        st.select_slider(
            "Total rain over the last 7 days (mm)", options=RAIN_WEEK_MM, key="rain_7_days",
            format_func=lambda v: f"{v} mm",
        )
        st.caption(f"= {ss.rain_7_days / MM_PER_INCH:.2f} in")
        st.slider("Average wind over the last 7 days (mph)", 3.0, 14.0, step=0.1, key="wind_7_days", format="%.1f")
        st.caption(f"= {mph_to_kmh(ss.wind_7_days):.1f} km/h")

        st.write("")
        st.button("🔥 Predict the fire risk", type="primary", on_click=go_predict,
                  use_container_width=True, key="predict")
    render_info()


def scroll_to_top() -> None:
    """The result screen should open at the top, not where the Predict button was."""
    ss.scroll_nonce = ss.get("scroll_nonce", 0) + 1  # new content -> the script runs again
    components.html(
        f"""<script>/* {ss.scroll_nonce} */
        const doc = window.parent.document;
        doc.querySelectorAll('[data-testid="stMain"], [data-testid="stAppViewContainer"], section.main')
           .forEach(el => el.scrollTo({{top: 0}}));
        window.parent.scrollTo({{top: 0}});
        </script>""",
        height=0,
    )


def render_result() -> None:
    w = ss.saved
    rain_in = w["rain"] / MM_PER_INCH
    rain_7_days_in = w["rain_7_days"] / MM_PER_INCH
    day = w["date"] if isinstance(w["date"], dt.date) else DEFAULTS["date"]
    prob, trees_fire, tree_probs = predict_day(
        model, day, w["max_temp"], w["min_temp"], w["wind"], rain_in, rain_7_days_in, w["wind_7_days"]
    )

    if ss.animate:  # short "the forest is voting" animation, only right after Predict
        ss.animate = False
        # Shown as a full-screen overlay, so the previous screen (which Streamlit keeps on the
        # page, greyed out, until the new one is drawn) stays hidden during the animation.
        overlay = st.empty()
        for shown in range(0, N_TREES + 1, 20):
            overlay.markdown(
                f'<div class="overlay">{forest_loading_html(tree_probs, shown)}</div>',
                unsafe_allow_html=True,
            )
            time.sleep(ANIMATION_SECONDS / (N_TREES / 20))
        time.sleep(0.4)
        overlay.empty()
        scroll_to_top()

    render_result_screen(w, prob, trees_fire, tree_probs, day, rain_in, rain_7_days_in)


def render_result_screen(w, prob, trees_fire, tree_probs, day, rain_in, rain_7_days_in) -> None:
    risk_pct = round(prob * 100)
    level = danger_level(prob * 100)
    alert = prob * 100 >= ALERT_THRESHOLD
    top_left, top_right = st.columns([1, 4], vertical_alignment="center")
    top_left.button("← Change the weather", on_click=go_back, use_container_width=True, key="back")
    top_right.markdown(weather_recap_html(w), unsafe_allow_html=True)
    st.write("")

    res_left, res_right = st.columns([1.15, 1], gap="large")
    with res_left:
        risk_color = ORANGE if alert else CALM
        alert_html = (
            '<div class="alert alert-on">🚒 FIRE ALERT - prepare crews</div>'
            if alert
            else '<div class="alert alert-off">✓ No alert</div>'
        )
        signals = html.escape(", ".join(main_signals(day, w["min_temp"], rain_in, rain_7_days_in)))
        st.markdown(
            f"""
<div class="card">
  <div class="risk-big" style="color:{risk_color};"><span class="risk-word">Fire risk:</span> {risk_pct}%</div>
  <span class="level-pill" style="background:{LEVEL_COLORS[level]};">Danger level: {level}</span>
  {alert_html}
  <div class="small-note">Alert threshold 30%: in our tests it catches 83% of fire days,
  because a missed fire costs far more than a false alarm.</div>
  {gauge_html(prob * 100)}
  <div class="signals"><b>Main signals:</b> {signals}.</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with res_right:
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
    render_economics(risk_pct / 100, alert)  # same rounded risk as displayed, so the sums add up
    render_info()


if ss.page == "form":
    render_form_screen()
else:
    render_result()

st.markdown(
    '<div class="footer">Data: Yavas et al. (2025), California Weather and Fire Prediction Dataset '
    "(1984-2025), Georgia Southern University (NOAA + CAL FIRE). "
    "AI for Business &amp; Society, SKEMA Business School.</div>",
    unsafe_allow_html=True,
)
