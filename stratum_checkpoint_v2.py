# ============================================================
# STRATUM RESEARCH — CHECKPOINT COMPLETO v2
# Panel con Regímenes Económicos de Markov & Índices Sistémicos
# Compatible con Google Colab + Google Drive
# ============================================================

# ── CELDA 0: Instalaciones (ejecutar una sola vez si es necesario) ──
# !pip install -q statsmodels linearmodels scikit-learn seaborn

import os
import sys
import json
import shutil
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Visualización
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Econometría
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy import stats

# Panel data (opcional)
try:
    from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects
    from linearmodels.panel import compare
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False
    warnings.warn("linearmodels no disponible — se omiten modelos de panel FE/RE.")

# Clustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ============================================================
# 0. RUTAS Y VERIFICACIÓN DE ARCHIVOS
# ============================================================

# Detección robusta de Colab
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")

if IN_COLAB:
    try:
        from google.colab import drive
        if not os.path.exists("/content/drive/MyDrive"):
            drive.mount("/content/drive")
            print("✔ Drive montado")
        else:
            print("✔ Drive ya estaba montado")
    except Exception as e:
        print(f"⚠ Error montando Drive: {e}")
    BASE = "/content/drive/MyDrive/stratum"
else:
    BASE = "./stratum"

CHECKPOINT = f"{BASE}/checkpoints"
FIGURES    = f"{BASE}/figures"
TABLES     = f"{BASE}/tables"

for d in [BASE, CHECKPOINT, FIGURES, TABLES]:
    os.makedirs(d, exist_ok=True)

# Verificar archivos requeridos
REQUIRED = [
    f"{BASE}/panel_with_system_regimes_economic.csv",
    f"{BASE}/system_with_markov_regimes_economic.csv",
]

print(f"\nBASE activa : {BASE}")
print(f"IN_COLAB    : {IN_COLAB}\n")

all_ok = True
for fp in REQUIRED:
    exists = os.path.exists(fp)
    print(f"  {'✔' if exists else '❌  FALTANTE'}  {fp}")
    if not exists:
        all_ok = False

if not all_ok:
    print(f"\nContenido actual de {BASE}:")
    try:
        for item in sorted(os.listdir(BASE)):
            print(f"    {item}")
    except Exception:
        print("    (carpeta vacía o no existe)")
    raise FileNotFoundError(
        f"\n❌ Coloca los CSV en: {BASE}\n"
        f"   Verifica con: !ls \"{BASE}\""
    )

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"\n✅ Archivos verificados  |  timestamp: {timestamp}")

# ============================================================
# 1. CARGAR INPUTS BASE
# ============================================================
print("\n" + "="*60)
print("1. CARGA DE DATOS")
print("="*60)

panel = pd.read_csv(
    f"{BASE}/panel_with_system_regimes_economic.csv",
    parse_dates=["date"]
)

system = pd.read_csv(
    f"{BASE}/system_with_markov_regimes_economic.csv",
    parse_dates=["date"]
)

print(f"PANEL  shape : {panel.shape}")
print(f"SYSTEM shape : {system.shape}")
print(f"\nPANEL  columnas : {panel.columns.tolist()}")
print(f"SYSTEM columnas : {system.columns.tolist()}")
print(f"\nPANEL  rango temporal : {panel['date'].min()} → {panel['date'].max()}")
print(f"SYSTEM rango temporal : {system['date'].min()} → {system['date'].max()}")

# ============================================================
# 2. CREAR DATASET FINAL DEL PAPER
# ============================================================
print("\n" + "="*60)
print("2. CONSTRUCCIÓN DEL DATASET FINAL")
print("="*60)

SYSTEM_COLS_DESIRED = [
    "date", "SG", "Fragility", "Absorption", "LiquidityStress",
    "ISI_extended", "FAI", "SSI", "ISI", "Regime_economic"
]

system_cols = [c for c in SYSTEM_COLS_DESIRED if c in system.columns]
missing_sys = set(SYSTEM_COLS_DESIRED) - set(system_cols)
if missing_sys:
    print(f"  ⚠ Columnas sistémicas ausentes en system.csv: {missing_sys}")

final = panel.merge(
    system[system_cols],
    on="date",
    how="left",
    suffixes=("_panel", "_system")
)

print(f"Dataset merged shape: {final.shape}")

# ============================================================
# 3. NORMALIZAR COLUMNA DE RÉGIMEN
# ============================================================
print("\n" + "="*60)
print("3. NORMALIZACIÓN DEL RÉGIMEN ECONÓMICO")
print("="*60)

def resolve_regime_column(df: pd.DataFrame) -> pd.DataFrame:
    """Consolida Regime_economic independientemente del sufijo post-merge."""
    if "Regime_economic" in df.columns:
        pass
    elif "Regime_economic_system" in df.columns:
        df["Regime_economic"] = df["Regime_economic_system"]
        print("  → Régimen tomado de '_system'")
    elif "Regime_economic_panel" in df.columns:
        df["Regime_economic"] = df["Regime_economic_panel"]
        print("  → Régimen tomado de '_panel'")
    else:
        raise KeyError(
            "No se encontró columna de régimen económico. "
            "Se esperaba 'Regime_economic', 'Regime_economic_system' "
            "o 'Regime_economic_panel'."
        )
    return df

final = resolve_regime_column(final)

# Detectar columna de entidad
entity_col = next(
    (c for c in ["entity", "country", "firm", "id", "code"] if c in final.columns),
    None
)
print(f"  Columna de entidad detectada: {entity_col}")

# Forward-fill por entidad o global
if entity_col:
    final["Regime_economic"] = (
        final.sort_values("date")
             .groupby(entity_col)["Regime_economic"]
             .ffill()
    )
    print("  → ffill aplicado por entidad")
else:
    final = final.sort_values("date")
    final["Regime_economic"] = final["Regime_economic"].ffill()
    print("  → ffill aplicado globalmente")

# Etiquetas legibles
REGIME_LABELS = {0: "Recesión", 1: "Expansión", 2: "Transición"}
unique_regimes = sorted(final["Regime_economic"].dropna().unique())
if len(unique_regimes) <= len(REGIME_LABELS):
    final["Regime_label"] = final["Regime_economic"].map(REGIME_LABELS).fillna("Desconocido")
else:
    final["Regime_label"] = "R_" + final["Regime_economic"].astype(str)

print(f"\nDistribución de regímenes:")
print(final["Regime_economic"].value_counts(dropna=False).sort_index().to_string())

# ============================================================
# 4. LIMPIEZA Y VALIDACIÓN
# ============================================================
print("\n" + "="*60)
print("4. LIMPIEZA Y VALIDACIÓN")
print("="*60)

SYSTEMIC_VARS = [c for c in [
    "SG", "Fragility", "Absorption", "LiquidityStress",
    "ISI_extended", "FAI", "SSI", "ISI"
] if c in final.columns]

print(f"\nVariables sistémicas disponibles: {SYSTEMIC_VARS}")

# Reporte de nulos
null_report = final[SYSTEMIC_VARS + ["Regime_economic"]].isnull().sum()
print(f"\nNulos por columna (antes de limpiar):\n{null_report.to_string()}")

# Eliminar filas donde TODAS las vars sistémicas sean nulas
before = len(final)
final = final.dropna(subset=SYSTEMIC_VARS, how="all")
print(f"\nFilas eliminadas (todas sistémicas nulas): {before - len(final)}")

# Eliminar duplicados
dup_keys = [entity_col, "date"] if entity_col else ["date"]
dupes = final.duplicated(subset=dup_keys).sum()
print(f"Duplicados (entidad-fecha): {dupes}")
if dupes > 0:
    final = final.drop_duplicates(subset=dup_keys)
    print(f"  → Duplicados eliminados.")

print(f"\nDataset final limpio: {final.shape}")

# Estadísticas descriptivas
print("\n--- Estadísticas descriptivas (variables sistémicas) ---")
desc = final[SYSTEMIC_VARS].describe().T
desc["cv"] = desc["std"] / desc["mean"].abs()
print(desc.round(4).to_string())

# ============================================================
# 5. INGENIERÍA DE VARIABLES
# ============================================================
print("\n" + "="*60)
print("5. INGENIERÍA DE VARIABLES")
print("="*60)

sort_keys = [entity_col, "date"] if entity_col else ["date"]
final = final.sort_values(sort_keys).reset_index(drop=True)

grp = final.groupby(entity_col) if entity_col else None

def _transform(col, fn):
    if grp:
        return grp[col].transform(fn)
    return final[col].transform(fn)

# Lags y primeras diferencias
for var in SYSTEMIC_VARS:
    final[f"{var}_lag1"] = _transform(var, lambda s: s.shift(1))
    final[f"{var}_lag2"] = _transform(var, lambda s: s.shift(2))
    final[f"d_{var}"]    = _transform(var, lambda s: s.diff())

# Dummies de régimen (one-hot)
regime_dummies = pd.get_dummies(
    final["Regime_economic"].astype(int),
    prefix="regime",
    drop_first=False
)
final = pd.concat([final, regime_dummies], axis=1)
REGIME_DUMMY_COLS = regime_dummies.columns.tolist()
print(f"Dummies de régimen: {REGIME_DUMMY_COLS}")

# Indicador de cambio de régimen
final["regime_change"] = (
    final["Regime_economic"] != _transform("Regime_economic", lambda s: s.shift(1))
).astype(int)
print(f"Cambios de régimen detectados: {final['regime_change'].sum()}")

# Z-scores
scaler = StandardScaler()
scaled_vars = [f"{v}_z" for v in SYSTEMIC_VARS]
final[scaled_vars] = scaler.fit_transform(final[SYSTEMIC_VARS].fillna(0))
print(f"Variables estandarizadas: {scaled_vars}")

# ============================================================
# 6. ANÁLISIS EXPLORATORIO (EDA)
# ============================================================
print("\n" + "="*60)
print("6. ANÁLISIS EXPLORATORIO")
print("="*60)

REGIME_COLORS  = {0: "#d62728", 1: "#2ca02c", 2: "#ff7f0e"}
DEFAULT_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]

# ─ 6.1 Series temporales por variable ───────────────────────
fig, axes = plt.subplots(
    len(SYSTEMIC_VARS), 1,
    figsize=(14, 3.5 * len(SYSTEMIC_VARS)),
    sharex=True
)
if len(SYSTEMIC_VARS) == 1:
    axes = [axes]

for ax, var in zip(axes, SYSTEMIC_VARS):
    dates   = final["date"].values
    regimes = final["Regime_economic"].values
    for rv in sorted(final["Regime_economic"].dropna().unique()):
        ax.fill_between(
            dates, 0, 1,
            where=(regimes == rv),
            transform=ax.get_xaxis_transform(),
            alpha=0.15,
            color=REGIME_COLORS.get(int(rv), "#aec7e8"),
            label=f"R{int(rv)}: {REGIME_LABELS.get(int(rv), '')}"
        )
    if entity_col:
        for i, ent in enumerate(final[entity_col].unique()[:5]):
            sub = final[final[entity_col] == ent]
            ax.plot(sub["date"], sub[var],
                    lw=1.2, alpha=0.75,
                    color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
                    label=str(ent))
    else:
        ax.plot(final["date"], final[var], lw=1.5, color="#1f77b4")
    ax.set_ylabel(var, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

handles, labels_ = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels_, handles))
axes[0].legend(by_label.values(), by_label.keys(),
               loc="upper left", fontsize=8, ncol=3)
axes[0].set_title("Series temporales de variables sistémicas por régimen", fontsize=13)
axes[-1].set_xlabel("Fecha")
plt.tight_layout()
fig_path_series = f"{FIGURES}/systemic_series_{timestamp}.png"
plt.savefig(fig_path_series, dpi=150, bbox_inches="tight")
plt.show()
print(f"  ✔ Series guardadas: {fig_path_series}")

# ─ 6.2 Correlaciones por régimen ────────────────────────────
for rv in sorted(final["Regime_economic"].dropna().unique()):
    subset = final[final["Regime_economic"] == rv][SYSTEMIC_VARS].dropna()
    if len(subset) < 5:
        continue
    fig_c, ax_c = plt.subplots(figsize=(8, 6))
    sns.heatmap(subset.corr(), annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, ax=ax_c,
                linewidths=.5, square=True)
    ax_c.set_title(
        f"Correlaciones — Régimen {int(rv)}: {REGIME_LABELS.get(int(rv), '')}",
        fontsize=12
    )
    plt.tight_layout()
    fp = f"{FIGURES}/corr_regime{int(rv)}_{timestamp}.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✔ Correlaciones R{int(rv)}: {fp}")

# ─ 6.3 Boxplots por régimen ──────────────────────────────────
fig_b, ax_b = plt.subplots(
    1, len(SYSTEMIC_VARS),
    figsize=(4 * len(SYSTEMIC_VARS), 5)
)
if len(SYSTEMIC_VARS) == 1:
    ax_b = [ax_b]

for ax, var in zip(ax_b, SYSTEMIC_VARS):
    rv_sorted = sorted(final["Regime_economic"].dropna().unique())
    data_box  = [final[final["Regime_economic"] == rv][var].dropna().values for rv in rv_sorted]
    labels_bx = [f"R{int(rv)}\n{REGIME_LABELS.get(int(rv),'')}" for rv in rv_sorted]
    bp = ax.boxplot(data_box, labels=labels_bx, patch_artist=True, notch=True)
    for patch, rv in zip(bp["boxes"], rv_sorted):
        patch.set_facecolor(REGIME_COLORS.get(int(rv), "#aec7e8"))
        patch.set_alpha(0.6)
    ax.set_title(var, fontsize=10)
    ax.set_xlabel("Régimen")

plt.suptitle("Distribución de variables sistémicas por régimen", y=1.02, fontsize=13)
plt.tight_layout()
fig_path_box = f"{FIGURES}/boxplot_regimes_{timestamp}.png"
plt.savefig(fig_path_box, dpi=150, bbox_inches="tight")
plt.show()
print(f"  ✔ Boxplots guardados: {fig_path_box}")

# ============================================================
# 7. TESTS ESTADÍSTICOS ENTRE REGÍMENES
# ============================================================
print("\n" + "="*60)
print("7. TESTS ESTADÍSTICOS ENTRE REGÍMENES")
print("="*60)

regime_vals  = sorted(final["Regime_economic"].dropna().unique())
stat_results = []

for var in SYSTEMIC_VARS:
    groups       = [final[final["Regime_economic"] == rv][var].dropna().values for rv in regime_vals]
    groups_valid = [g for g in groups if len(g) >= 3]

    if len(groups_valid) >= 2:
        stat_kw, p_kw = stats.kruskal(*groups_valid)
        row = {
            "variable"          : var,
            "test"              : "Kruskal-Wallis",
            "statistic"         : round(stat_kw, 4),
            "p_value"           : round(p_kw, 6),
            "significativo_5pct": p_kw < 0.05
        }
        for rv, g in zip(regime_vals, groups):
            row[f"media_R{int(rv)}"] = round(g.mean(), 4) if len(g) else np.nan
        stat_results.append(row)

df_stats    = pd.DataFrame(stat_results)
stats_path  = f"{TABLES}/kruskal_wallis_{timestamp}.csv"
df_stats.to_csv(stats_path, index=False)
print(df_stats.to_string(index=False))
print(f"\n  ✔ Tests guardados: {stats_path}")

# ============================================================
# 8. MODELOS ECONOMÉTRICOS
# ============================================================
print("\n" + "="*60)
print("8. MODELOS ECONOMÉTRICOS")
print("="*60)

# Identificar variable dependiente
CANDIDATE_DEPVARS = ["Fragility", "ISI", "ISI_extended", "FAI"]
dep_var = next((v for v in CANDIDATE_DEPVARS if v in final.columns), None)
if dep_var is None:
    dep_var = SYSTEMIC_VARS[0]

indep_vars = [v for v in SYSTEMIC_VARS if v != dep_var]
print(f"  Variable dependiente  : {dep_var}")
print(f"  Variables regresoras  : {indep_vars}")

# ─ 8.1 OLS por régimen ──────────────────────────────────────
ols_results = {}

for rv in regime_vals:
    sub       = final[final["Regime_economic"] == rv].copy()
    sub_clean = sub[[dep_var] + indep_vars].dropna()
    min_obs   = max(10, len(indep_vars) + 2)

    if len(sub_clean) < min_obs:
        print(f"\n  ⚠ Régimen {int(rv)}: {len(sub_clean)} obs < mínimo {min_obs}. Omitido.")
        continue

    y     = sub_clean[dep_var]
    X     = add_constant(sub_clean[indep_vars])
    model = OLS(y, X).fit(cov_type="HC3")
    ols_results[int(rv)] = model

    print(f"\n── OLS Régimen {int(rv)} ({REGIME_LABELS.get(int(rv), '')}) ──")
    print(f"   N={model.nobs:.0f}  R²={model.rsquared:.4f}  "
          f"R²adj={model.rsquared_adj:.4f}  "
          f"F={model.fvalue:.2f} (p={model.f_pvalue:.4f})")
    print(model.summary2().tables[1].round(4).to_string())

    dw_stat        = durbin_watson(model.resid)
    bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
    print(f"   Durbin-Watson : {dw_stat:.4f}")
    print(f"   Breusch-Pagan : stat={bp_stat:.4f}  p={bp_p:.4f}")

    vif_df = pd.DataFrame({
        "variable": X.columns,
        "VIF"     : [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    print(f"   VIF:\n{vif_df.to_string(index=False)}")

    coef_path = f"{TABLES}/ols_regime{int(rv)}_{timestamp}.csv"
    model.summary2().tables[1].to_csv(coef_path)
    print(f"   ✔ Coeficientes: {coef_path}")

# ─ 8.2 Panel FE ─────────────────────────────────────────────
if HAS_LINEARMODELS and entity_col and len(indep_vars) > 0:
    print("\n── Panel OLS — Efectos Fijos ──")
    try:
        pan = final.set_index([entity_col, "date"])
        pan = pan[[dep_var] + indep_vars + REGIME_DUMMY_COLS].dropna()

        y_pan = pan[dep_var]
        X_pan = add_constant(pan[indep_vars + REGIME_DUMMY_COLS[1:]])

        fe_model = PanelOLS(
            y_pan, X_pan,
            entity_effects=True,
            time_effects=False,
            drop_absorbed=True
        ).fit(cov_type="clustered", cluster_entity=True)

        print(fe_model.summary)
        fe_path = f"{TABLES}/panelFE_{timestamp}.csv"
        fe_model.params.to_frame("coef").join(
            fe_model.pvalues.to_frame("p_value")
        ).to_csv(fe_path)
        print(f"  ✔ Panel FE: {fe_path}")

    except Exception as e:
        print(f"  ⚠ Panel FE falló: {e}")

# ============================================================
# 9. DURACIÓN Y MATRIZ DE TRANSICIÓN
# ============================================================
print("\n" + "="*60)
print("9. DURACIÓN DE EPISODIOS Y TRANSICIONES")
print("="*60)

def compute_durations(series: pd.Series) -> pd.DataFrame:
    records, current, start = [], None, None
    for i, val in enumerate(series):
        if val != current:
            if current is not None:
                records.append({
                    "regime"  : current,
                    "label"   : REGIME_LABELS.get(int(current), str(current)),
                    "duration": i - start
                })
            current, start = val, i
    if current is not None:
        records.append({
            "regime"  : current,
            "label"   : REGIME_LABELS.get(int(current), str(current)),
            "duration": len(series) - start
        })
    return pd.DataFrame(records)

regime_series = final.sort_values("date")["Regime_economic"].dropna()
dur_df        = compute_durations(regime_series)

print("Duración de episodios por régimen:")
print(dur_df.groupby("label")["duration"].describe().round(2).to_string())

dur_path = f"{TABLES}/regime_durations_{timestamp}.csv"
dur_df.to_csv(dur_path, index=False)

# Matriz de transición
r_shift  = regime_series.shift(1)
trans_df = pd.crosstab(
    r_shift.rename("From"),
    regime_series.rename("To"),
    normalize="index"
).round(4)
trans_df.index   = [REGIME_LABELS.get(int(i), i) for i in trans_df.index]
trans_df.columns = [REGIME_LABELS.get(int(c), c) for c in trans_df.columns]

print("\nMatriz de transición empírica (probabilidades):")
print(trans_df.to_string())

trans_path = f"{TABLES}/transition_matrix_{timestamp}.csv"
trans_df.to_csv(trans_path)

fig_t, ax_t = plt.subplots(figsize=(6, 5))
sns.heatmap(trans_df.astype(float), annot=True, fmt=".2f",
            cmap="Blues", linewidths=.5, ax=ax_t, vmin=0, vmax=1)
ax_t.set_title("Matriz de Transición Empírica entre Regímenes", fontsize=12)
ax_t.set_xlabel("Régimen destino")
ax_t.set_ylabel("Régimen origen")
plt.tight_layout()
fig_path_t = f"{FIGURES}/transition_matrix_{timestamp}.png"
plt.savefig(fig_path_t, dpi=150, bbox_inches="tight")
plt.show()
print(f"  ✔ Heatmap de transición: {fig_path_t}")

# ============================================================
# 10. PERSISTENCIA DE RESULTADOS
# ============================================================
print("\n" + "="*60)
print("10. PERSISTENCIA DE RESULTADOS")
print("="*60)

final_csv = f"{BASE}/final_dataset_{timestamp}.csv"
final.to_csv(final_csv, index=False)
print(f"  ✔ Dataset CSV    : {final_csv}  {final.shape}")

try:
    final_pq = f"{BASE}/final_dataset_{timestamp}.parquet"
    final.to_parquet(final_pq, index=False)
    print(f"  ✔ Dataset Parquet: {final_pq}")
except Exception as e:
    print(f"  ⚠ Parquet no disponible: {e}")

checkpoint_meta = {
    "timestamp"       : timestamp,
    "panel_shape"     : list(panel.shape),
    "system_shape"    : list(system.shape),
    "final_shape"     : list(final.shape),
    "dep_var"         : dep_var,
    "indep_vars"      : indep_vars,
    "systemic_vars"   : SYSTEMIC_VARS,
    "regime_vals"     : [int(v) for v in regime_vals],
    "regime_labels"   : REGIME_LABELS,
    "entity_col"      : entity_col,
    "ols_regimes_fit" : {
        str(rv): {
            "N"      : int(m.nobs),
            "R2"     : round(m.rsquared, 6),
            "R2_adj" : round(m.rsquared_adj, 6),
            "F_pvalue": round(m.f_pvalue, 6)
        }
        for rv, m in ols_results.items()
    },
    "figures"         : [fig_path_series, fig_path_box, fig_path_t],
    "tables"          : [stats_path, trans_path, dur_path]
}

ck_path = f"{CHECKPOINT}/checkpoint_{timestamp}.json"
with open(ck_path, "w") as f:
    json.dump(checkpoint_meta, f, indent=2, default=str)
print(f"  ✔ Metadata JSON  : {ck_path}")

# ============================================================
# 11. RESUMEN EJECUTIVO
# ============================================================
print("\n" + "="*60)
print("RESUMEN EJECUTIVO DEL CHECKPOINT")
print("="*60)
print(f"  Período analizado    : {final['date'].min().date()} → {final['date'].max().date()}")
print(f"  Observaciones totales: {len(final):,}")
if entity_col:
    print(f"  Entidades únicas     : {final[entity_col].nunique()}")
print(f"  Regímenes detectados : {len(regime_vals)}  →  {list(REGIME_LABELS.values())}")
print(f"  Variable dependiente : {dep_var}")
print(f"  Variables sistémicas : {SYSTEMIC_VARS}")
print(f"\nModelos OLS estimados:")
for rv, m in ols_results.items():
    print(f"  Régimen {rv} ({REGIME_LABELS.get(rv, '')}) — "
          f"N={int(m.nobs)}  R²={m.rsquared:.4f}  p(F)={m.f_pvalue:.4f}")
print(f"\nArtefactos generados:")
print(f"  Figuras  → {FIGURES}/")
print(f"  Tablas   → {TABLES}/")
print(f"  Dataset  → {final_csv}")
print(f"  Metadata → {ck_path}")
print("\n✅  Checkpoint v2 completado exitosamente.")
