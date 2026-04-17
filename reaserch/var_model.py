from __future__ import annotations

from pathlib import Path
import pandas as pd
from statsmodels.tsa.api import VAR

DATA = Path("data/core/stratum_universe.csv")
OUT = Path("output/tables/var_summary.txt")

COLS = ["cp", "sg", "phi", "theta", "inflation", "credit_spread", "energy_price"]


def fit_var() -> object:
    df = pd.read_csv(DATA)
    x = df[COLS].apply(pd.to_numeric, errors="coerce").dropna()

    model = VAR(x)
    res = model.fit(maxlags=4, ic="aic")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        f.write(str(res.summary()))

    return res


if __name__ == "__main__":
    print(fit_var().summary())
