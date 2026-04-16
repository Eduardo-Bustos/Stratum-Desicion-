from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yaml


RETURNS_PATH = Path("data/market/asset_returns.csv")
SETTINGS_PATH = Path("configs/settings.yaml")
OUTPUT_PATH = Path("outputs/allocations/optimized_weights.csv")


def load_settings() -> dict:
    with SETTINGS_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1
    return float(dd.min())


def objective(weights: np.ndarray, rets: pd.DataFrame, penalty: float = 0.50) -> float:
    port = rets.mul(weights, axis=1).sum(axis=1)
    mean_ret = float(port.mean())
    vol = float(port.std(ddof=0))
    sharpe = mean_ret / (vol + 1e-9)
    mdd = abs(max_drawdown(port))
    return sharpe - penalty * mdd


def optimize_weights(regime: str = "SELECTION", n_trials: int = 20000, seed: int = 42) -> pd.DataFrame:
    rets = pd.read_csv(RETURNS_PATH)
    if "date" in rets.columns:
        rets = rets.drop(columns=["date"])

    rets = rets.apply(pd.to_numeric, errors="coerce").dropna()
    assets = list(rets.columns)

    rng = np.random.default_rng(seed)

    best_score = -1e9
    best_w = None

    for _ in range(n_trials):
        w = rng.random(len(assets))
        w = w / w.sum()

        # regime tilt
        if regime.upper() == "PROPAGATION":
            for i, a in enumerate(assets):
                if a in ["gold", "usd", "energy", "fertilizers"]:
                    w[i] *= 1.25
                if a in ["equities_em", "credit_hy"]:
                    w[i] *= 0.50
            w = w / w.sum()

        elif regime.upper() == "SELECTION":
            for i, a in enumerate(assets):
                if a in ["energy", "fertilizers", "grains", "gold"]:
                    w[i] *= 1.15
            w = w / w.sum()

        score = objective(w, rets)

        if score > best_score:
            best_score = score
            best_w = w.copy()

    out = pd.DataFrame({"asset": assets, "weight": best_w})
    out = out.sort_values("weight", ascending=False).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    return out


if __name__ == "__main__":
    df = optimize_weights("SELECTION")
    print(df)
