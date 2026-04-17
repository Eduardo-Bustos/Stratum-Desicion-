from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml

CFG = Path("config/settings.yaml")
STATE = Path("output/tables/state.csv")
SIG_OUT = Path("output/tables/signals.csv")
WGT_OUT = Path("output/alloc/weights_by_country.csv")


def load_cfg() -> dict:
    with CFG.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in df.iterrows():
        reg = r["regime"]

        if reg == "absorption":
            eq = "overweight"
            ra = "neutral"
            hd = "light"
        elif reg == "selection":
            eq = "underweight"
            ra = "overweight"
            hd = "medium"
        else:
            eq = "strong_underweight"
            ra = "strong_overweight"
            hd = "high"

        rows.append(
            {
                "country": r["country"],
                "regime": reg,
                "equity_signal": eq,
                "real_asset_signal": ra,
                "hedge_signal": hd,
                "ssi": r["ssi"],
            }
        )

    out = pd.DataFrame(rows)
    SIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SIG_OUT, index=False)
    return out


def build_weights(df: pd.DataFrame) -> pd.DataFrame:
    cfg = load_cfg()
    alloc = cfg["portfolio"]
    rows = []

    for _, r in df.iterrows():
        bucket = alloc[r["regime"]]
        for asset, w in bucket.items():
            rows.append(
                {
                    "country": r["country"],
                    "regime": r["regime"],
                    "asset": asset,
                    "target_weight": w,
                    "ssi": r["ssi"],
                }
            )

    out = pd.DataFrame(rows)
    WGT_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(WGT_OUT, index=False)
    return out


if __name__ == "__main__":
    state = pd.read_csv(STATE)
    build_signals(state)
    build_weights(state)
