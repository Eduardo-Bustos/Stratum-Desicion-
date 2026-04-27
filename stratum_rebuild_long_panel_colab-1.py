import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.api import VAR

MASTER_FILE = "master_v2_enriched_colab_ready.csv"
START_DATE = "1990-01-01"
ROLL_Z = 252
MIN_OBS_PER_INSTRUMENT = 260
BATCH_SIZE = 25


def safe_zscore(s, window=252, min_periods=60):
    m = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    return (s - m) / sd.replace(0, np.nan)


def clean_master(path=MASTER_FILE):
    master = pd.read_csv(path)
    master.columns = master.columns.str.lower()
    for c in ["instrument_id", "ticker", "yahoo_symbol"]:
        if c in master.columns:
            master[c] = master[c].astype(str).str.strip()
    if "is_active" in master.columns:
        active = master["is_active"].astype(str).str.lower().isin(["true", "1", "yes", "y", "activo"])
        master = master[active].copy()
    master["yahoo_symbol"] = master["yahoo_symbol"].replace({"nan": np.nan, "": np.nan}).fillna(master["ticker"])
    master = master.dropna(subset=["yahoo_symbol"]).drop_duplicates("instrument_id")
    return master


def download_yahoo_panel(master, start=START_DATE, batch_size=BATCH_SIZE):
    frames = []
    rows = master.to_dict("records")
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        symbols = [r["yahoo_symbol"] for r in batch if str(r.get("primary_source", "Yahoo Finance")) == "Yahoo Finance"]
        symbols = [s for s in symbols if isinstance(s, str) and s and s.lower() != "nan"]
        if not symbols:
            continue
        print(f"Downloading batch {i//batch_size+1}: {len(symbols)} symbols")
        data = yf.download(symbols, start=start, auto_adjust=True, group_by="ticker", threads=True, progress=False)
        for r in batch:
            sym = r["yahoo_symbol"]
            iid = r["instrument_id"]
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if sym in data.columns.get_level_values(0):
                        raw = data[sym].copy()
                    else:
                        continue
                else:
                    raw = data.copy()
                if raw.empty or "Close" not in raw.columns:
                    continue
                cols = [c for c in ["Close", "High", "Low", "Volume"] if c in raw.columns]
                g = raw[cols].copy().dropna(subset=["Close"])
                if len(g) < MIN_OBS_PER_INSTRUMENT:
                    continue
                g = g.rename(columns={"Close":"close", "High":"high", "Low":"low", "Volume":"volume"})
                g["date"] = pd.to_datetime(g.index)
                g["instrument_id"] = iid
                g["ticker"] = r.get("ticker", iid)
                g["yahoo_symbol"] = sym
                frames.append(g.reset_index(drop=True))
            except Exception as e:
                print(f"Skipped {iid}/{sym}: {e}")
                continue
    if not frames:
        raise RuntimeError("No market data downloaded. Check symbols / connection.")
    return pd.concat(frames, ignore_index=True).sort_values(["instrument_id", "date"])


def compute_factors(panel):
    outs=[]
    for iid, g in panel.groupby("instrument_id", sort=False):
        g = g.sort_values("date").copy()
        g["return"] = np.log(g["close"] / g["close"].shift(1))
        if "high" in g.columns and "low" in g.columns:
            g["spread_proxy"] = np.log(g["high"] / g["low"].replace(0, np.nan)).rolling(2, min_periods=2).mean()
        else:
            g["spread_proxy"] = np.nan
        dollar_vol = g["close"] * g.get("volume", np.nan)
        g["amihud"] = (g["return"].abs() / dollar_vol.replace(0, np.nan)).rolling(5, min_periods=3).mean()
        g["amihud"] = g["amihud"].replace([np.inf, -np.inf], np.nan).clip(lower=1e-12)
        g["depth_proxy"] = 1.0 / g["amihud"]
        g["adoption_proxy"] = g["return"].rolling(20, min_periods=10).std()
        g["vol_cluster"] = g["return"].abs().rolling(20, min_periods=10).mean()
        g["spread_z"] = safe_zscore(g["spread_proxy"], ROLL_Z)
        g["amihud_z"] = safe_zscore(g["amihud"] * 1e12, ROLL_Z)
        g["vol_cluster_z"] = safe_zscore(g["vol_cluster"], ROLL_Z)
        g["adoption_z"] = safe_zscore(g["adoption_proxy"], ROLL_Z)
        g["depth_z"] = safe_zscore(np.log(g["depth_proxy"].replace(0, np.nan)), ROLL_Z)
        g["fragility"] = 0.5*g["vol_cluster_z"] + 0.3*g["amihud_z"] + 0.2*g["spread_z"]
        g["absorption"] = 0.6*g["depth_z"] + 0.4*g["adoption_z"]
        g["sg"] = g["fragility"] - g["absorption"]
        g["liquidity_stress"] = 0.5*g["spread_z"] + 0.5*g["amihud_z"]
        g["isi_extended"] = 0.30*g["fragility"] + 0.25*g["liquidity_stress"] + 0.20*g["vol_cluster_z"] + 0.25*g["sg"]
        outs.append(g)
    fac = pd.concat(outs, ignore_index=True)
    fac = fac.dropna(subset=["sg", "fragility", "absorption"])
    fac["regime"] = np.select([fac["sg"]>1.5, fac["sg"]>0.5, fac["sg"]<-0.5], ["Acute Stress", "Fragile", "Absorbent"], default="Transitional")
    keep=["date","instrument_id","ticker","yahoo_symbol","return","spread_proxy","amihud","depth_proxy","adoption_proxy","vol_cluster","fragility","absorption","sg","liquidity_stress","isi_extended","regime"]
    return fac[keep].sort_values(["instrument_id","date"]).reset_index(drop=True)


def build_system_indices(factors):
    daily = factors.groupby("date").agg(sg_level=("sg","mean"), sg_dispersion=("sg","std"), fragility=("fragility","mean"), absorption=("absorption","mean"), liquidity_stress=("liquidity_stress","mean"), isi_extended=("isi_extended","mean"), n_instruments=("instrument_id","nunique")).dropna()
    daily["sg_ext"] = daily["sg_level"] + 0.5*daily["sg_dispersion"]
    daily["fai"] = safe_zscore(daily["sg_ext"].diff().abs().rolling(20, min_periods=10).mean(), 252)
    daily["ssi"] = safe_zscore(daily["sg_dispersion"], 252)
    daily["isi_tdif"] = 1/(1+np.exp(-(1.2*daily["sg_ext"] + 0.8*daily["fai"].fillna(0) + 1.0*daily["ssi"].fillna(0))))
    tau = 0.3492
    daily["state_class"] = np.select([(daily["sg_ext"]>=tau) & (daily["fai"]>0.7), daily["sg_ext"]>=tau, daily["sg_ext"]<tau], ["fragmented", "binding", "coherent"], default="fragile")
    return daily.reset_index()


def compute_persistence(system):
    s = system.dropna(subset=["state_class"]).copy()
    s["state_lag"] = s["state_class"].shift(1)
    s = s.dropna(subset=["state_lag"])
    order = ["coherent", "fragile", "binding", "fragmented"]
    P = pd.crosstab(s["state_lag"], s["state_class"], normalize="index").reindex(index=order, columns=order).fillna(0)
    duration = pd.Series(np.where(1-np.diag(P)>1e-12, 1/(1-np.diag(P)), np.inf), index=order, name="expected_duration")
    return P, duration


def main():
    master = clean_master()
    master.to_csv("master_v2_enriched_verified.csv", index=False)
    panel = download_yahoo_panel(master)
    panel.to_csv("daily_market_snapshot_long.csv", index=False)
    factors = compute_factors(panel)
    factors.to_csv("daily_stratum_factors_long.csv", index=False)
    system = build_system_indices(factors)
    system.to_csv("stratum_system_indices_long.csv", index=False)
    P, dur = compute_persistence(system)
    P.to_csv("regime_transition_matrix_long.csv")
    dur.to_csv("regime_expected_duration_long.csv")
    print("master", master.shape)
    print("panel", panel.shape)
    print("factors", factors.shape)
    print("system", system.shape)
    print(P.round(3))
    print(dur.round(2))

if __name__ == "__main__":
    main()
