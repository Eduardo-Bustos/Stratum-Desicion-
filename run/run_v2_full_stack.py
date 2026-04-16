from __future__ import annotations

import pandas as pd

from engine.weekly_state_builder import build_state
from engine.portfolio_layer import generate_signals, build_target_weights
from research.estimation_var import fit_var
from research.monte_carlo_engine import run_mc
from research.optimizer_engine import optimize_weights


def main() -> None:
    state = build_state()
    generate_signals(state)
    build_target_weights(state)

    fit_var()
    run_mc()

    global_regime = state["regime"].mode().iloc[0]
    optimize_weights(global_regime)

    print("STRATUM v2.3 READY")


if __name__ == "__main__":
    main()
