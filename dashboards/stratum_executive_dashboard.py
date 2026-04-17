from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st


st.set_page_config(page_title="STRATUM v2.3", layout="wide")

BASE = Path("outputs")
TABLES = BASE / "tables"
ALLOC = BASE / "allocations"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def regime_light(value: str) -> str:
    v = str(value).upper()
    if v == "ABSORPTION":
        return "🟢 ABSORPTION"
    if v == "SELECTION":
        return "🟡 SELECTION"
    if v == "PROPAGATION":
        return "🔴 PROPAGATION"
    return "⚪ UNKNOWN"


st.title("STRATUM v2.3 — Institutional Dashboard")
st.caption("Macro regime detection → portfolio translation → optimization")

weekly = load_csv(TABLES / "weekly_regime_monitor.csv")
signals = load_csv(TABLES / "portfolio_signals.csv")
weights = load_csv(ALLOC / "target_weights_by_country.csv")
opt_weights = load_csv(ALLOC / "optimized_weights.csv")
mc = load_csv(TABLES / "monte_carlo_paths.csv")

if weekly.empty:
    st.warning("No weekly_regime_monitor.csv found. Run: python run/run_v2_full_stack.py")
    st.stop()

global_regime = weekly["regime"].mode().iloc[0]
avg_ssi = round(pd.to_numeric(weekly["ssi"], errors="coerce").mean(), 4)

m1, m2, m3 = st.columns(3)
m1.metric("Global Regime", global_regime)
m2.metric("Average SSI", avg_ssi)
m3.metric("Countries", len(weekly))

st.markdown(f"### System State: {regime_light(global_regime)}")

st.divider()

left, right = st.columns([1.3, 1])

with left:
    st.subheader("Weekly Regime Monitor")
    show_cols = [
        c for c in [
            "country", "theta_eff", "cp", "phi", "sg", "ssi", "regime"
        ] if c in weekly.columns
    ]
    st.dataframe(
        weekly[show_cols].sort_values("ssi", ascending=False),
        use_container_width=True,
    )

with right:
    st.subheader("Top Risk Countries")
    top = weekly.sort_values("ssi", ascending=False).head(10)
    st.dataframe(top[["country", "ssi", "regime"]], use_container_width=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Portfolio Signals")
    if not signals.empty:
        st.dataframe(signals, use_container_width=True)
    else:
        st.info("No portfolio_signals.csv found.")

with col2:
    st.subheader("Optimized Weights")
    if not opt_weights.empty:
        st.dataframe(opt_weights, use_container_width=True)
    else:
        st.info("No optimized_weights.csv found.")

st.divider()

st.subheader("Target Weights by Country")
if not weights.empty:
    country_selected = st.selectbox("Country", sorted(weights["country"].dropna().unique()))
    sub = weights[weights["country"] == country_selected].copy()
    st.dataframe(sub.sort_values("target_weight", ascending=False), use_container_width=True)
else:
    st.info("No target_weights_by_country.csv found.")

st.divider()

st.subheader("Monte Carlo Snapshot")
if not mc.empty:
    latest_week = mc["week
