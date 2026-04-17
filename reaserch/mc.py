from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yaml

CFG = Path("config/settings.yaml")
OUT = Path("output/tables/mc_paths.csv")


def load_cfg() -> dict:
    with CFG.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_mc() -> pd.DataFrame:
    cfg = load_cfg()["monte_carlo"]
    n = int(cfg["n_paths"])
    h = int(cfg["horizon_weeks"])
    seed = int(cfg["seed"])

    rng = np.random.default_rng(seed)
    rows = []

    for p in range(n):
        cp = 0.55
        sg = 0.20
        phi = 0.30
        theta = 0.50

        for t in range(h):
            cp = max(0, cp + 0.02 + rng.normal(0, 0.05))
            phi = max(0, phi + 0.15 * cp + rng.normal(0, 0.04))
            sg = max(-1, sg + 0.10 * phi - 0.08 * theta + rng.normal(0, 0.03))
            theta = max(0, theta - 0.05 * sg + 0.01)

            rows.append(
                {
                    "path": p,
                    "week": t,
                    "cp": round(cp, 4),
                    "sg": round(sg, 4),
                    "phi": round(phi, 4),
                    "theta": round(theta, 4),
                }
            )

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    return out


if __name__ == "__main__":
    print(run_mc().head())
