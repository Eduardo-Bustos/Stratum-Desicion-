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
    market = pd.read_csv(M
