from __future__ import annotations

import numpy as np
import pandas as pd


def minmax_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)


def compute_theta_eff(
    df: pd.DataFrame,
    w_liquidity: float = 0.40,
    w_reserves: float = 0.30,
    w_policy: float = 0.30,
) -> pd.Series:
    liquidity = minmax_series(df["liquidity_score"])
    reserves = minmax_series(df["fx_reserves_score"])
    policy = minmax_series(df["policy_stability_score"])

    theta = (
        w_liquidity * liquidity
        + w_reserves * reserves
        + w_policy * policy
    )
    return theta.round(4)


def compute_cp(
    df: pd.DataFrame,
    w_energy: float = 0.45,
    w_inflation: float = 0.20,
    w_debt: float = 0.15,
    w_external: float = 0.20,
) -> pd.Series:
    energy = minmax_series(df["energy_dependency"])
    inflation = minmax_series(df["inflation"])
    debt = minmax_series(df["debt_ratio"])

    # current account: más negativo = peor
    current_account_gap = minmax_series(-pd.to_numeric(df["current_account"], errors="coerce"))

    cp = (
        w_energy * energy
        + w_inflation * inflation
        + w_debt * debt
        + w_external * current_account_gap
    )
    return cp.round(4)


def compute_phi(df: pd.DataFrame) -> pd.Series:
    # Estrés latente: inflación + spreads + fertilizantes
    infl = minmax_series(df["inflation"])
    spreads = minmax_series(df["credit_spread"])
    fert = minmax_series(df["fertilizer_stress"])

    phi = 0.35 * infl + 0.35 * spreads + 0.30 * fert
    return phi.round(4)


def compute_sg(theta_eff: pd.Series, cp: pd.Series) -> pd.Series:
    return (cp - theta_eff).round(4)


def compute_ssi(sg: pd.Series, cp: pd.Series, phi: pd.Series) -> pd.Series:
    ssi = (0.45 * sg) + (0.35 * cp) + (0.20 * phi)
    return ssi.round(4)


def classify_regime(
    sg: pd.Series,
    sg_absorption_max: float = 0.0,
    sg_selection_max: float = 0.5,
) -> pd.Series:
    out = []
    for x in sg:
        if pd.isna(x):
            out.append("UNKNOWN")
        elif x <= sg_absorption_max:
            out.append("ABSORPTION")
        elif x <= sg_selection_max:
            out.append("SELECTION")
        else:
            out.append("PROPAGATION")
    return pd.Series(out, index=sg.index)
