from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml

from engine.stratum_core_metrics import (
    compute_theta_eff,
    compute_cp,
    compute_phi,
    compute_sg,
    compute_ssi,
    classify_regime,
)


SETTINGS_PATH = Path("configs/settings.yaml")
WEO_PATH = Path("data/weo/IMF_WEO_2026.csv")
MARKET_PATH = Path("data/market/market_data.csv")
OUTPUT_PATH = Path("outputs/tables/weekly_regime_monitor.csv")


def load_settings() -> dict:
    with SETTINGS_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_state() -> pd.DataFrame:
    settings = load_settings()

    weo = pd.read_csv(WEO_PATH)
    market = pd.read_csv(MARKET_PATH)

    df = weo.merge(market, on="country", how="inner")

    theta_w = settings["weights"]["theta"]
    cp_w = settings["weights"]["cp"]
    th = settings["thresholds"]

    df["theta_eff"] = compute_theta_eff(
        df,
        w_liquidity=theta_w["liquidity_score"],
        w_reserves=theta_w["fx_reserves_score"],
        w_policy=theta_w["policy_stability_score"],
    )

    df["cp"] = compute_cp(
        df,
        w_energy=cp_w["energy_dependency"],
        w_inflation=cp_w["inflation"],
        w_debt=cp_w["debt_ratio"],
        w_external=cp_w["current_account_gap"],
    )

    df["phi"] = compute_phi(df)
    df["sg"] = compute_sg(df["theta_eff"], df["cp"])
    df["ssi"] = compute_ssi(df["sg"], df["cp"], df["phi"])

    df["regime"] = classify_regime(
        df["sg"],
        sg_absorption_max=th["sg_absorption_max"],
        sg_selection_max=th["sg_selection_max"],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    return df


if __name__ == "__main__":
    state = build_state()
    print(state[["country", "theta_eff", "cp", "phi", "sg", "ssi", "regime"]])
