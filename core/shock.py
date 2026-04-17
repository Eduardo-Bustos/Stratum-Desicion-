from __future__ import annotations

import pandas as pd
import yaml
from pathlib import Path

SCEN = Path("config/scenarios.yaml")


def load_scenarios() -> dict:
    with SCEN.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_scenario(df: pd.DataFrame, name: str) -> pd.DataFrame:
    cfg = load_scenarios()[name]
    out = df.copy()

    if "energy_dependency_multiplier" in cfg:
        out["energy_dependency"] *= cfg["energy_dependency_multiplier"]
    if "inflation_multiplier" in cfg:
        out["inflation"] *= cfg["inflation_multiplier"]
    if "liquidity_score_shift" in cfg:
        out["liquidity_score"] += cfg["liquidity_score_shift"]
    if "policy_stability_shift" in cfg:
        out["policy_stability_score"] += cfg["policy_stability_shift"]
    if "fertilizer_stress_shift" in cfg:
        out["fertilizer_stress"] += cfg["fertilizer_stress_shift"]

    return out
