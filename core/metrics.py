from __future__ import annotations

import pandas as pd


def scale(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    lo, hi = x.min(), x.max()
    if pd.isna(lo) or pd.isna(hi) or lo == hi:
        return pd.Series(0.5, index=x.index)
    return (x - lo) / (hi - lo)


def theta(
    df: pd.DataFrame,
    w_liq: float = 0.40,
    w_res: float = 0.30,
    w_pol: float = 0.30,
) -> pd.Series:
    liq = scale(df["liquidity_score"])
    res = scale(df["fx_reserves_score"])
    pol = scale(df["policy_stability_score"])
    return (w_liq * liq + w_res * res + w_pol * pol).round(4)


def cp(
    df: pd.DataFrame,
    w_eng: float = 0.45,
    w_inf: float = 0.20,
    w_debt: float = 0.15,
    w_ext: float = 0.20,
) -> pd.Series:
    eng = scale(df["energy_dependency"])
    inf = scale(df["inflation"])
    debt = scale(df["debt_ratio"])
    ext = scale(-pd.to_numeric(df["current_account"], errors="coerce"))
    return (w_eng * eng + w_inf * inf + w_debt * debt + w_ext * ext).round(4)


def phi(df: pd.DataFrame) -> pd.Series:
    inf = scale(df["inflation"])
    spr = scale(df["credit_spread"])
    fer = scale(df["fertilizer_stress"])
    return (0.35 * inf + 0.35 * spr + 0.30 * fer).round(4)


def sg(theta_s: pd.Series, cp_s: pd.Series) -> pd.Series:
    return (cp_s - theta_s).round(4)


def ssi(sg_s: pd.Series, cp_s: pd.Series, phi_s: pd.Series) -> pd.Series:
    return (0.45 * sg_s + 0.35 * cp_s + 0.20 * phi_s).round(4)


def regime(sg_s: pd.Series, a_max: float = 0.0, s_max: float = 0.5) -> pd.Series:
    out = []
    for x in sg_s:
        if pd.isna(x):
            out.append("unknown")
        elif x <= a_max:
            out.append("absorption")
        elif x <= s_max:
            out.append("selection")
        else:
            out.append("propagation")
    return pd.Series(out, index=sg_s.index)
