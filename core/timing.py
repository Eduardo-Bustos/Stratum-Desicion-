from __future__ import annotations

import pandas as pd


def trade_timing(row: pd.Series) -> str:
    if float(row.get("sg", 0)) >= 1.0:
        return "immediate"
    if float(row.get("cp", 0)) >= 0.8:
        return "staggered"
    return "wait"
