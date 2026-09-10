"""Sales & Demand Forecaster — Streamlit app.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import json
import os

from src.data import DataValidationError, generate_sample_sales, guess_columns, load_series
from src.models import compare_models
from src.security import ACCESS_KEY_ENV, MIN_ACCESS_KEY_LENGTH, configured_access_key, is_authorized

# TensorFlow is optional: it powers the LSTM locally, but is too heavy / not always
# available on the hosted free tier. When it is missing, the app still runs the
# classical models live and shows the LSTM's back-test result from offline training.
try:
    import tensorflow  # noqa: F401
    TF_OK = True
except Exception:
    TF_OK = False


def saved_lstm_metric():
    try:
        path = os.path.join(os.path.dirname(__file__), "models", "metrics.json")
        with open(path) as f:
            for r in json.load(f).get("results", []):
                if r.get("model") == "LSTM":
                    return {"MAPE": r.get("MAPE"), "RMSE": r.get("RMSE")}
    except Exception:
        return None
    return None


st.set_page_config(page_title="Sales & Demand Forecaster", layout="wide")


def _configured_access_key() -> str:
    """Read the access key from the environment, falling back to Streamlit secrets."""
    secret_value = ""
    try:
        secret_value = str(st.secrets.get(ACCESS_KEY_ENV, ""))
    except Exception:  # Missing optional Streamlit secrets configuration.
        pass
    return configured_access_key(secret_value)


expected_access_key = _configured_access_key()
if len(expected_access_key) < MIN_ACCESS_KEY_LENGTH:
    st.error(
        f"Set {ACCESS_KEY_ENV} to a random value of at least "
        f"{MIN_ACCESS_KEY_LENGTH} characters before starting the app."
    )
    st.stop()

entered_access_key = st.sidebar.text_input("Access key", type="password")
if not is_authorized(entered_access_key, expected_access_key):
    st.info("Enter the access key to continue.")
    st.stop()

INK, MUTED, ACCENT, LINE = "#16181d", "#626a75", "#0d7d72", "#e4e7eb"
GRAYS = ["#9aa3ad", "#c2c8cf", "#7c8591"]
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_UPLOAD_ROWS = 10_000

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Archivo:wght@600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"] { font-family:'Inter',system-ui,-apple-system,sans-serif; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stHeader"] { display:none !important; }
.block-container { padding-top:2.4rem; padding-bottom:3rem; max-width:1120px; }

.eyebrow { font-size:.72rem; letter-spacing:.16em; text-transform:uppercase;
           color:#0d7d72; font-weight:600; }
.h-title { font-family:'Archivo',sans-serif; font-weight:700; font-size:2.05rem;
           color:#16181d; margin:.2rem 0 .35rem; letter-spacing:-.015em; line-height:1.1; }
.h-sub { color:#626a75; font-size:1rem; max-width:64ch; line-height:1.5; }
.rule { height:1px; background:#e4e7eb; margin:1.2rem 0 1.5rem; }

.stats { display:flex; gap:1px; background:#e4e7eb; border:1px solid #e4e7eb;
         border-radius:12px; overflow:hidden; margin:.1rem 0 1.7rem; }
.stat { flex:1; background:#fff; padding:1.05rem 1.2rem; }
.stat .k { font-size:.68rem; letter-spacing:.09em; text-transform:uppercase;
           color:#626a75; font-weight:600; }
.stat .v { font-family:'IBM Plex Mono',monospace; font-size:1.75rem; font-weight:600;
           color:#16181d; margin-top:.35rem; font-variant-numeric:tabular-nums; line-height:1; }
.stat .v.accent { color:#0d7d72; }
.stat .d { font-size:.8rem; color:#626a75; margin-top:.45rem; }

.tbl { width:100%; border-collapse:collapse; font-size:.93rem; margin-top:.3rem; }
.tbl th { text-align:left; font-size:.68rem; letter-spacing:.07em; text-transform:uppercase;
          color:#626a75; font-weight:600; padding:.55rem .75rem; border-bottom:1px solid #e4e7eb; }
.tbl td { padding:.62rem .75rem; border-bottom:1px solid #e4e7eb;
          font-variant-numeric:tabular-nums; color:#16181d; }
.tbl th.num, .tbl td.num { text-align:right; }
.tbl tr.best td { font-weight:600; }
.tbl tr.best td:first-child { box-shadow:inset 3px 0 0 #0d7d72; }
.section { font-family:'Archivo',sans-serif; font-weight:600; font-size:1.05rem;
           color:#16181d; margin:.4rem 0 .2rem; }
.note { color:#626a75; font-size:.82rem; margin-top:.7rem; line-height:1.5; }

section[data-testid="stSidebar"] { background:#fff; border-right:1px solid #e4e7eb; }
section[data-testid="stSidebar"] .block-container { padding-top:1.6rem; }
.side-label { font-size:.68rem; letter-spacing:.11em; text-transform:uppercase;
              color:#626a75; font-weight:600; margin:.9rem 0 .1rem; }
.side-name { font-family:'Archivo',sans-serif; font-weight:700; font-size:1.05rem; color:#16181d; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300, max_entries=8, show_spinner=False)
def _forecast(values: tuple, dates: tuple, horizon: int, names: tuple):
    series = pd.Series(values, index=pd.to_datetime(dates), name="sales").asfreq("MS")
    return compare_models(series, horizon=horizon, names=list(names))


def run_forecast(series, horizon, names):
    return _forecast(tuple(series.values), tuple(series.index.astype(str)), horizon, tuple(names))


# ---- Sidebar ----
st.sidebar.markdown('<div class="side-name">Forecaster</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="side-label">Data</div>', unsafe_allow_html=True)
source = st.sidebar.radio("Source", ["Sample data", "Upload CSV"], label_visibility="collapsed")

st.sidebar.markdown('<div class="side-label">Forecast horizon</div>', unsafe_allow_html=True)
horizon = st.sidebar.slider("Months", 1, 12, 3, label_visibility="collapsed")

st.sidebar.markdown('<div class="side-label">Models</div>', unsafe_allow_html=True)
chosen = st.sidebar.multiselect(
    "Models", ["LSTM", "Holt-Winters", "SARIMA", "Seasonal Naive"],
    default=["LSTM", "Holt-Winters", "SARIMA", "Seasonal Naive"],
    label_visibility="collapsed")

series = None
if source == "Sample data":
    series = load_series(generate_sample_sales(), "date", "sales")
else:
    up = st.sidebar.file_uploader("CSV file", type=["csv"])
    if up is not None:
        try:
            if getattr(up, "size", 0) > MAX_UPLOAD_BYTES:
                raise DataValidationError("The uploaded CSV must be 20 MB or smaller.")
            raw = pd.read_csv(up, nrows=MAX_UPLOAD_ROWS + 1)
            if len(raw) > MAX_UPLOAD_ROWS:
                raise DataValidationError("The uploaded CSV must contain 10,000 rows or fewer.")
            gd, gv = guess_columns(raw)
            date_col = st.sidebar.selectbox("Date column", raw.columns, index=list(raw.columns).index(gd))
            value_col = st.sidebar.selectbox("Sales column", raw.columns, index=list(raw.columns).index(gv))
            series = load_series(raw, date_col, value_col)
            dropped_rows = series.attrs.get("dropped_rows", 0)
            interpolated_months = series.attrs.get("interpolated_months", 0)
            if dropped_rows:
                st.sidebar.warning(f"Ignored {dropped_rows} invalid row(s).")
            if interpolated_months:
                st.sidebar.warning(f"Interpolated {interpolated_months} missing month(s).")
        except (DataValidationError, pd.errors.EmptyDataError, pd.errors.ParserError,
                UnicodeDecodeError, ValueError) as exc:
            st.sidebar.error(str(exc))


# ---- Header ----
st.markdown(
    """
<div class="eyebrow">Sales &amp; demand forecasting</div>
<div class="h-title">How much will you sell next month?</div>
<div class="h-sub">Read your monthly sales history, project the months ahead, and see which
model the data actually supports — a deep-learning model held against three classical baselines.</div>
<div class="rule"></div>
""",
    unsafe_allow_html=True,
)

if series is None:
    st.markdown('<div class="note">Add a CSV in the sidebar, or switch to the sample data to try it now.</div>',
                unsafe_allow_html=True)
    st.stop()
if len(series) < 18:
    st.markdown('<div class="note">Around 18 months of history are needed for a reliable seasonal forecast. '
                'Add more rows and try again.</div>', unsafe_allow_html=True)
    st.stop()
if not chosen:
    st.markdown('<div class="note">Select at least one model in the sidebar.</div>', unsafe_allow_html=True)
    st.stop()

live = [m for m in chosen if m != "LSTM" or TF_OK]
if "LSTM" in chosen and not TF_OK and source == "Upload CSV":
    st.info("LSTM requires TensorFlow and is unavailable in this environment; the selected classical models will run.")
with st.spinner("Training models and forecasting"):
    results = run_forecast(series, horizon, live) if live else []

# When TF isn't available on the host, surface the LSTM's offline back-test result
# (computed by train.py on the sample data) so the comparison is still complete.
lstm_offline = False
if "LSTM" in chosen and not TF_OK and source == "Sample data":
    m = saved_lstm_metric()
    if m:
        results = results + [{"name": "LSTM", "metrics": m, "future": None, "offline": True}]
        lstm_offline = True

if not results:
    st.markdown('<div class="note">This hosted demo runs the classical baselines live. '
                'Select one of those, or run the project locally for the live LSTM.</div>',
                unsafe_allow_html=True)
    st.stop()

failed = [r for r in results if r.get("error")]
for failure in failed:
    failure_name = escape(str(failure["name"]))
    st.warning(f"{failure_name} could not produce a forecast for this dataset.")

ok = [r for r in results if r.get("future") is not None]
if not ok and not any(r.get("offline") for r in results):
    st.error("No selected model produced a forecast. Check the data and try again.")
    st.stop()

valid = [r for r in ok if r["metrics"]["MAPE"] == r["metrics"]["MAPE"]]
best = min(valid, key=lambda r: r["metrics"]["MAPE"]) if valid else (ok[0] if ok else None)
holdout_months = ok[0].get("test_size", 12) if ok else 12

# ---- Readout strip ----
if best is not None:
    nxt = float(best["future"].iloc[0])
    last = float(series.iloc[-1])
    chg = (nxt - last) / last * 100 if last else 0
    arrow = "+" if chg >= 0 else ""
    best_name = escape(str(best["name"]))
    st.markdown(
        f"""
<div class="stats">
  <div class="stat"><div class="k">Next month — forecast</div>
    <div class="v accent">{nxt:,.0f}</div>
    <div class="d">{arrow}{chg:.1f}% vs last month ({last:,.0f})</div></div>
  <div class="stat"><div class="k">Recommended model</div>
    <div class="v">{best_name}</div>
    <div class="d">{best['metrics']['MAPE']:.2f}% average error on back-test</div></div>
  <div class="stat"><div class="k">Next {horizon} months — total</div>
    <div class="v">{best['future'].sum():,.0f}</div>
    <div class="d">projected with {best_name}</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

# ---- Forecast chart ----
fig = go.Figure()
fig.add_trace(go.Scatter(x=series.index, y=series.values, name="History",
                         line=dict(color=INK, width=2)))
gi = 0
for r in ok:
    is_best = best is not None and r["name"] == best["name"]
    color = ACCENT if is_best else GRAYS[gi % len(GRAYS)]
    if not is_best:
        gi += 1
    link = pd.concat([series.iloc[[-1]], r["future"]])
    fig.add_trace(go.Scatter(
        x=link.index, y=link.values,
        name=f"{r['name']} forecast" + (" (recommended)" if is_best else ""),
        line=dict(color=color, width=3 if is_best else 1.6, dash="solid" if is_best else "dot")))
fig.update_layout(
    height=440, template="plotly_white", hovermode="x unified",
    font=dict(family="Inter, sans-serif", color=INK, size=13),
    margin=dict(t=20, b=10, l=10, r=10), plot_bgcolor="#fff", paper_bgcolor="#fff",
    legend=dict(orientation="h", y=-0.16, x=0, bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(showgrid=False, showline=True, linecolor=LINE, ticks="outside", tickcolor=LINE),
    yaxis=dict(title="Sales", gridcolor="#eef0f2", zeroline=False, tickformat=","))
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ---- Accuracy table ----
st.markdown('<div class="section">Model accuracy</div>', unsafe_allow_html=True)
display_results = [r for r in results if r.get("future") is not None or r.get("offline")]
rows_sorted = sorted(display_results, key=lambda r: (r["metrics"]["MAPE"] != r["metrics"]["MAPE"], r["metrics"]["MAPE"]))
trs = ""
for r in rows_sorted:
    m = r["metrics"]
    cls = "best" if best is not None and r["name"] == best["name"] else ""
    mape = "—" if m["MAPE"] != m["MAPE"] else f"{m['MAPE']:.2f}%"
    rmse = "—" if m["RMSE"] != m["RMSE"] else f"{m['RMSE']:,.0f}"
    name = escape(str(r["name"]))
    if r.get("offline"):
        name += '<span style="color:#9aa3ad;font-size:.78em;font-weight:400;"> · offline</span>'
    trs += f'<tr class="{cls}"><td>{name}</td><td class="num">{mape}</td><td class="num">{rmse}</td></tr>'
lstm_note = (" The LSTM is evaluated during offline training (train.py); this hosted demo "
             "runs the classical models live and runs the LSTM live when the project is run locally."
             ) if lstm_offline else ""
st.markdown(
    f"""
<table class="tbl">
<thead><tr><th>Model</th><th class="num">Avg error (MAPE)</th><th class="num">RMSE</th></tr></thead>
<tbody>{trs}</tbody>
</table>
<div class="note">Each model is scored on the most recent {holdout_months} months, held out during training.
Lower is better. On short monthly series the classical baselines are often hard for an LSTM to beat —
the recommended model is simply whichever earns the lowest error here.{lstm_note}</div>
""",
    unsafe_allow_html=True,
)

with st.expander("Forecast values"):
    out = pd.DataFrame({r["name"]: r["future"] for r in ok})
    out.index = out.index.strftime("%Y-%m")
    st.dataframe(out.style.format("{:,.0f}"), use_container_width=True)
