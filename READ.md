# DECISION STRATUM v2.3

Institutional macro-financial regime engine with:

- structural metrics: Theta_eff, CP, Phi, SG, SSI
- regime classification: Absorption / Selection / Propagation
- portfolio signals by country
- Monte Carlo simulation
- portfolio optimization
- Streamlit dashboard

## Repo structure

- `configs/` → settings and scenario rules
- `data/` → WEO, market and research datasets
- `engine/` → regime, metrics and portfolio logic
- `research/` → VAR, Monte Carlo, optimizer
- `run/` → orchestration scripts
- `dashboards/` → Streamlit UI
- `outputs/` → generated tables, figures and allocations

## Run full system

```bash
python run/run_v2_full_stack.py
