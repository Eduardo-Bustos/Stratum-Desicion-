from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml


SETTINGS_PATH = Path("configs/settings.yaml")
INPUT_PATH = Path("outputs/tables/weekly_regime_monitor.csv")
OUTPUT_SIGNALS = Path("outputs/tables/portfolio_signals.csv")
OUTPUT_WEIGHTS = Path("outputs/allocations/target_weights_by_country.csv")


def load_settings() -> dict:
    with SETTINGS_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in df.iterrows():
        regime = r["regime"]
        if regime == "ABSORPTION":
            eq_signal = "OVERWEIGHT"
            real_asset_signal = "NEUTRAL"
            hedge_signal = "LIGHT"
        elif regime == "SELECTION":
            eq_signal = "UNDERWEIGHT"
            real_asset_signal = "OVERWEIGHT"
            hedge_signal = "MEDIUM"
        else:
            eq_signal = "STRONG_UNDERWEIGHT"
            real_asset_signal = "STRONG_OVERWEIGHT"
            hedge_signal = "HIGH"

        rows.append(
            {
                "country": r["country"],
                "regime": regime,
                "equities_signal": eq_signal,
                "real_assets_signal": real_asset_signal,
                "hedge_signal": hedge_signal,
                "ssi": r["ssi"],
            }
        )

    out = pd.DataFrame(rows)
    OUTPUT_SIGNALS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_SIGNALS, index=False)
    return out


def build_target_weights(df: pd.DataFrame) -> pd.DataFrame:
    settings = load_settings()
    alloc = settings["portfolio"]

    rows = []

    for _, r in df.iterrows():
        regime = r["regime"].lower()
        weights = alloc[regime]

        for asset, weight in weights.items():
            rows.append(
                {
                    "country": r["country"],
                    "regime": r["regime"],
                    "asset": asset,
                    "target_weight": weight,
                    "ssi": r["ssi"],
                }
            )

    out = pd.DataFrame(rows)
    OUTPUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_WEIGHTS, index=False)
    return out


if __name__ == "__main__":
    state = pd.read_csv(INPUT_PATH)
    generate_signals(state)
    build_target_weights(state)
