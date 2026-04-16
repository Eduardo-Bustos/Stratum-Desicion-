from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yaml


SETTINGS_PATH = Path("configs/settings.yaml")
OUTPUT_PATH = Path("outputs/tables/monte_carlo_paths.csv")


def load_settings() -> dict:
    with SETTINGS_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_mc() -> pd.DataFrame:
    cfg = load_settings()["monte_carlo"]

    n_paths = int(cfg["n_paths"])
    horizon = int(cfg["horizon_weeks"])
    seed = int(cfg["seed"])

    rng = np.random.default_rng(seed)
    rows = []

    for path in range(n_paths):
        cp = 0.55
        sg = 0.20
        phi = 0.30
        theta = 0.50

        for t in range(horizon):
            e1 = rng.normal(0, 0.05)
            e2 = rng.normal(0, 0.04)
            e3 = rng.normal(0, 0.03)

            cp = max(0, cp + 0.02 + e1)
            phi = max(0, phi + 0.15 * cp + e2)
            sg = max(-1, sg + 0.10 * phi - 0.08 * theta + e3)
            theta = max(0, theta - 0.05 * sg + 0.01)

            rows.append(
                {
                    "path": path,
                    "week": t,
                    "cp": round(cp, 4),
                    "sg": round(sg, 4),
                    "phi": round(phi, 4),
                    "theta_eff": round(theta, 4),
                }
            )

    out = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    return out


if __name__ == "__main__":
    df = run_mc()
    print(df.head())
