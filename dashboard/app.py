from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="stratum", layout="wide")
st.title("stratum — institutional dashboard")

TAB = Path("output/tables")
ALC = Path("output/alloc")


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


state = load(TAB / "state.csv")
sig = load(TAB / "signals.csv")
wgt = load(ALC / "weights_by_country.csv")
opt = load(ALC / "opt_weights.csv")
mc = load(TAB / "mc_paths.csv")

if state.empty:
    st.warning("run: python run/run_all.py")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("regime", state["regime"].mode().iloc[0])
m2.metric("avg ssi", round(pd.to_numeric(state["ssi"], errors="coerce").mean(), 4))
m3.metric("countries", len(state))

st.subheader("state")
st.dataframe(state.sort_values("ssi", ascending=False), use_container_width=True)

st.subheader("signals")
if not sig.empty:
    st.dataframe(sig, use_container_width=True)

st.subheader("country weights")
if not wgt.empty:
    c = st.selectbox("country", sorted(wgt["country"].unique()))
    st.dataframe(wgt[wgt["country"] == c].sort_values("target_weight", ascending=False), use_container_width=True)

st.subheader("optimized weights")
if not opt.empty:
    st.dataframe(opt, use_container_width=True)

st.subheader("monte carlo")
if not mc.empty:
    st.line_chart(mc.groupby("week")[["cp", "sg", "phi", "theta"]].mean(), use_container_width=True)
