from __future__ import annotations

from core.state import build_state
from core.signals import build_signals, build_weights
from research.var_model import fit_var
from research.mc import run_mc
from research.optimize import optimize


def main() -> None:
    state = build_state()
    build_signals(state)
    build_weights(state)

    fit_var()
    run_mc()

    reg = state["regime"].mode().iloc[0]
    optimize(reg)

    print("stratum ready")


if __name__ == "__main__":
    main()
