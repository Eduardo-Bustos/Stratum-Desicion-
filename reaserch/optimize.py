from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

RET = Path("data/market/asset_returns.csv")
OUT = Path("output/alloc/opt_weights.csv")


def max_dd(r: pd.Series) -> float:
    wealth = (1 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1
    return float(dd.min())


def score_fn(w: np.ndarray, x: pd.DataFrame, penalty: float = 0.50) -> float:
    port = x.mul(w, axis=1).sum(axis=1)
    mean = float(port.mean())
    vol = float(port.std(ddof=0))
    sharpe = mean / (vol + 1e-9)
    dd = abs(max_dd(port))
    return sharpe - penalty * dd


def optimize(regime: str = "selection", n_trials: int = 20000, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(RET)
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    x = df.apply(pd.to_numeric, errors="coerce").dropna()
    assets = list(x.columns)

    rng = np.random.default_rng(seed)
    best_s = -1e9
    best_w = None

    for _ in range(n_trials):
        w = rng.random(len(assets))
        w /= w.sum()

        if regime == "propagation":
            for i, a in enumerate(assets):
                if a in ["gold", "usd", "energy", "fertilizers"]:
                    w[i] *= 1.25
                if a in ["equities_em", "credit_hy"]:
                    w[i] *= 0.50
            w /= w.sum()

        elif regime == "selection":
            for i, a in enumerate(assets):
                if a in ["energy", "fertilizers", "grains", "gold"]:
                    w[i] *= 1.15
            w /= w.sum()

        s = score_fn(w, x)
        if s > best_s:
            best_s = s
            best_w = w.copy()

    out = pd.DataFrame({"asset": assets, "weight": best_w})
    out = out.sort_values("weight", ascending=False).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    return out


if __name__ == "__main__":
    print(optimize("selection"))
