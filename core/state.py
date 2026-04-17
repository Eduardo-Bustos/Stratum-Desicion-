from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml

from core.metrics import theta, cp, phi, sg, ssi, regime

CFG = Path("config/settings.yaml")
WEO = Path("data/macro/weo_2026.csv")
MKT = Path("data/market/market_data.csv")
OUT = Path("output/tables/state.csv")


def load_cfg() -> dict:
    with CFG.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_csv(path: Path) -> None:
    first = path.read_text(encoding="utf-8").splitlines()[0]
    assert "," in first, f"invalid csv: {path}"


def build_state() -> pd.DataFrame:
    validate_csv(WEO)
    validate_csv(MKT)

    cfg = load_cfg()
    weo = pd.read_csv(WEO)
    mkt = pd.read_csv(MKT)

    df = weo.merge(mkt, on="country", how="inner")

    w_t = cfg["weights"]["theta"]
    w_c = cfg["weights"]["cp"]
    th = cfg["thresholds"]

    df["theta"] = theta(
        df,
        w_liq=w_t["liquidity_score"],
        w_res=w_t["fx_reserves_score"],
        w_pol=w_t["policy_stability_score"],
    )

    df["cp"] = cp(
        df,
        w_eng=w_c["energy_dependency"],
        w_inf=w_c["inflation"],
        w_debt=w_c["debt_ratio"],
        w_ext=w_c["current_account_gap"],
    )

    df["phi"] = phi(df)
    df["sg"] = sg(df["theta"], df["cp"])
    df["ssi"] = ssi(df["sg"], df["cp"], df["phi"])
    df["regime"] = regime(
        df["sg"],
        a_max=th["sg_absorption_max"],
        s_max=th["sg_selection_max"],
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    return df


if __name__ == "__main__":
    print(build_state())
