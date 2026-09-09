# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TP3 — Algoritmo de Detección de Ofertas Hoteleras
#
# **Diplomatura en Ciencia de Datos · Mentoría ID90Travel 2026**  
# **Grupo 02:** Francisco Cisneros  
#
# Este práctico implementa, evalúa y compara formalmente los mecanismos de clasificación para determinar si una tarifa hotelera observada es inusualmente baja para su contexto.
#
# ---
#
# ## Estructura del trabajo
#
# 1. **Carga y curación contextual del dataset**: normalización de tarifa a precio estándar (`price_std`) por habitación-noche-persona y segmentación de mercado.
# 2. **Implementación de modelos de detección**:
#    - **Modelo 1 (Baseline lineal docente)**: Z-Score Gaussiano sobre media y desvío estándar.
#    - **Modelo 2 (Propuesta estadística adoptada)**: Z-Score Log-Normal Híbrido ($z_{\log} < -1.0$ y ahorro $\ge 20\%$ sobre la mediana del contexto).
#    - **Modelo 3 (Detección de anomalías no supervisada)**: Isolation Forest multidimensional sobre precio relativo y dispersión de mercado.
#    - **Modelo 4 (Modelo predictivo supervisado)**: Clasificador Random Forest entrenado para predecir deals a partir de variables de búsqueda.
# 3. **Evaluación rigurosa sin etiquetas externas**:
#    - Validación cronológica (Train/Test Out-of-Time).
#    - Stress test con inyección controlada de ofertas sintéticas (Recall empírico y resistencia a falsos positivos).
#    - Análisis de sensibilidad frente a la granularidad de mercado (Contexto Fino vs Contexto Laxo).
# 4. **Conclusiones, trade-offs y limitaciones operativas**.

# %%
import os
import sys
import warnings
from pathlib import Path

# Resolver ruta raíz del repositorio
_current_dir = Path(".").resolve()
for p in [_current_dir, _current_dir.parent, _current_dir.parent.parent]:
    if (p / "config.py").exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        ROOT_DIR = p
        break

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, confusion_matrix
import config
import auxiliary_functions as af

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["font.size"] = 10

print(f"Directorio raíz del proyecto: {ROOT_DIR}")

# %% [markdown]
# ## 1. Carga de datos y preparación de celdas de mercado
#
# Se carga la muestra representativa de 300.000 observaciones (`data/sample_data_300k.csv.gz`). Se aplican las transformaciones validadas en el TP2:
# - Normalización de destinos con `destination_with_nearest.csv`.
# - Cálculo de `price_std` en USD por noche, habitación y persona.
# - Filtros de consistencia operativa ($5 \le \text{price\_std} \le 2500$).
# - Segmentación contextual: $\text{destination\_final} \times \text{month} \times \text{week\_in\_month} \times \text{stay\_duration}$.

# %%
data_dir = ROOT_DIR / "data"
sample_gz = data_dir / "sample_data_300k.csv.gz"
mapping_file = data_dir / "destination_with_nearest.csv"

mapping_df = af.load_destination_mapping(mapping_file)

print(f"Cargando dataset: {sample_gz.name}")
df_raw = pd.read_csv(
    sample_gz,
    dtype={"city": str, "state": str, "country": str, "country_code": str},
    low_memory=False
)
print(f"Registros crudos cargados: {len(df_raw):,}")

# Aplicar mapping de destinos
df = af.apply_destination_mapping(df_raw, mapping_df)

# Normalizar precios
pax_equiv = df["number_of_adults"] + 0.5 * df["number_of_kids"].clip(lower=0)
pax_equiv = pax_equiv.replace(0, 1)
df["price_std"] = df["avg_price_average"] / (df["nights"] * df["number_of_rooms"] * pax_equiv)

# Filtro de calidad
mask_valid = (
    (df["nights"] > 0) &
    (df["number_of_rooms"] > 0) &
    (df["number_of_adults"] > 0) &
    (df["avg_price_average"] > 0) &
    (df["price_std"] >= 5) &
    (df["price_std"] <= 2500)
)
df = df[mask_valid].copy()

# Features temporales
df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
df = df.dropna(subset=["date_start"]).copy()
df["month"] = df["date_start"].dt.month
df["day_of_week"] = df["date_start"].dt.dayofweek
df["week_in_month"] = ((df["date_start"].dt.day - 1) // 7) + 1
df["is_weekend"] = df["day_of_week"].isin([4, 5, 6]).astype(int)

# Duración de estadía
def categorizar_estadia(n):
    if n <= 2: return "corta"
    if n <= 5: return "media"
    return "larga"

df["stay_duration"] = df["nights"].apply(categorizar_estadia)

# Categoría de hotel (price_bucket)
p25 = df["price_std"].quantile(0.25)
p50 = df["price_std"].quantile(0.50)
p75 = df["price_std"].quantile(0.75)
def categorizar_bucket(p):
    if p <= p25: return "Budget"
    if p <= p50: return "Mid-Scale"
    if p <= p75: return "Upscale"
    return "Luxury"

df["price_bucket"] = df["price_std"].apply(categorizar_bucket)

# Definición de contexto fino
df["market_context"] = (
    df["destination_final"].astype(str) + "_" +
    df["month"].astype(str) + "_" +
    df["week_in_month"].astype(str) + "_" +
    df["stay_duration"].astype(str)
)

print(f"Dataset curado final: {len(df):,} búsquedas.")
print(f"Mercados contextuales únicos: {df['market_context'].nunique():,}")

# %% [markdown]
# ## 2. Implementación de Algoritmos de Detección
#
# Para cada celda de mercado con masa crítica ($N \ge 30$), se computan las estadísticas de referencia:
# - Media y desvío estándar lineal ($\mu, \sigma$) para el **Z-score Gaussiano del docente**.
# - Media y desvío logarítmico ($\mu_{\ln}, \sigma_{\ln}$) y mediana ($M$) para el **Z-score Log-Normal Híbrido**.
# - Descuento relativo: $1 - \frac{\text{price\_std}}{M_{\text{contexto}}}$.

# %%
context_stats = df.groupby("market_context").agg(
    n_obs=("price_std", "count"),
    mean_linear=("price_std", "mean"),
    std_linear=("price_std", "std"),
    median_price=("price_std", "median"),
    mean_log=("price_std", lambda x: np.log(x).mean()),
    std_log=("price_std", lambda x: np.log(x).std())
).reset_index()

# Conservar celdas con suficiente soporte muestral
valid_contexts = context_stats[context_stats["n_obs"] >= 30]
print(f"Celdas contextuales con N >= 30: {len(valid_contexts):,} ({valid_contexts['n_obs'].sum():,} observaciones cubiertas)")

df = df.merge(valid_contexts, on="market_context", how="inner")

# 1. Z-Score Lineal (Docente)
df["z_docente"] = (df["price_std"] - df["mean_linear"]) / df["std_linear"].replace(0, np.nan)
df["deal_docente"] = (df["z_docente"] <= -1.0).astype(int)

# 2. Z-Score Log-Normal Híbrido (Propuesta adoptada)
df["z_log"] = (np.log(df["price_std"]) - df["mean_log"]) / df["std_log"].replace(0, np.nan)
df["pct_discount"] = (df["median_price"] - df["price_std"]) / df["median_price"]
df["deal_hibrido"] = ((df["z_log"] <= -1.0) & (df["pct_discount"] >= 0.20)).astype(int)

print("\nResultados de detección preliminar:")
print(f" - Deal Docente (Gaussiano z <= -1.0): {df['deal_docente'].mean():.2%} de las tarifas")
print(f" - Deal Híbrido (Log-Normal z <= -1.0 + Ahorro >= 20%): {df['deal_hibrido'].mean():.2%} de las tarifas")

# %% [markdown]
# ### 2.1 Modelo No Supervisado: Isolation Forest Multidimensional
#
# El Isolation Forest no asume distribución paramétrica. Aísla observaciones atípicas en un espacio de 3 dimensiones:
# 1. `ratio_vs_median`: precio relativo frente al centro del mercado.
# 2. `z_log`: distancia proporcional en desvíos estándar logarítmicos.
# 3. `std_log`: volatilidad del mercado donde ocurre la tarifa.

# %%
features_if = ["z_log", "pct_discount"]
df_if = df.dropna(subset=features_if).copy()

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.10,
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(df_if[features_if])

# anomaly_score: valores negativos son anomalías
df_if["anomaly_score"] = iso_forest.score_samples(df_if[features_if])
# Consideramos oferta anómala cuando es outlier Y tiene descuento positivo
df_if["deal_iso_forest"] = (
    (iso_forest.predict(df_if[features_if]) == -1) &
    (df_if["pct_discount"] >= 0.20)
).astype(int)

print(f" - Deal Isolation Forest (Score anómalo + Ahorro >= 20%): {df_if['deal_iso_forest'].mean():.2%}")

# %% [markdown]
# ### 2.2 Modelo Predictivo Supervisado: Random Forest Classifier
#
# Entrenamos un clasificador para estimar la probabilidad de que una búsqueda sea un deal a partir de las condiciones de reserva conocidas al momento de consultar (`month`, `day_of_week`, `nights`, `number_of_rooms`, `number_of_adults`, `number_of_kids`, `price_std`, `is_weekend`).

# %%
df_ml = df.dropna(subset=["z_log", "deal_hibrido"]).copy()

feature_cols = [
    "month", "day_of_week", "nights", "number_of_rooms",
    "number_of_adults", "number_of_kids", "is_weekend", "price_std"
]

# Train / Test split cronológico (80% train / 20% test temporal)
df_ml = df_ml.sort_values("date_start").reset_index(drop=True)
split_idx = int(len(df_ml) * 0.8)

train_df = df_ml.iloc[:split_idx]
test_df = df_ml.iloc[split_idx:]

X_train, y_train = train_df[feature_cols], train_df["deal_hibrido"]
X_test, y_test = test_df[feature_cols], test_df["deal_hibrido"]

rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

test_df["prob_deal_rf"] = rf.predict_proba(X_test)[:, 1]
test_df["deal_rf_pred"] = (test_df["prob_deal_rf"] >= 0.40).astype(int)

auc_score = roc_auc_score(y_test, test_df["prob_deal_rf"])
print(f"ROC-AUC del modelo Random Forest en test cronológico: {auc_score:.4f}")
print("\nReporte de clasificación del modelo supervisado:")
print(classification_report(y_test, test_df["deal_rf_pred"], digits=3))

# %% [markdown]
# ## 3. Evaluación sin Etiquetas Externas
#
# Sin un ground truth humano independiente, la evaluación técnica se estructura en tres pilares:
#
# 1. **Estabilidad temporal**: consistencia de la tasa de ofertas a lo largo del tiempo.
# 2. **Prueba de inyección sintética (Stress Test de Perturbaciones)**:
#    Tomamos observaciones catalogadas como NO-DEAL y les aplicamos reducciones controladas de precio ($10\%, 20\%, 30\%, 40\%, 50\%$). Medimos cuántas son recuperadas (Recall empírico).
#    A la inversa, inyectamos incrementos de precio ($+30\%$) para verificar la tasa de falsos positivos.

# %%
# Experimento de Inyección Sintética de Ofertas
non_deals = df[df["deal_hibrido"] == 0].sample(n=min(5000, len(df[df["deal_hibrido"] == 0])), random_state=42).copy()

discounts = [0.10, 0.20, 0.30, 0.40, 0.50]
capture_rates_docente = []
capture_rates_hibrido = []

for disc in discounts:
    perturbed_price = non_deals["price_std"] * (1 - disc)
    
    # Recalcular Z-Scores sobre precio perturbado
    z_doc_pert = (perturbed_price - non_deals["mean_linear"]) / non_deals["std_linear"]
    z_log_pert = (np.log(perturbed_price) - non_deals["mean_log"]) / non_deals["std_log"]
    disc_pert = (non_deals["median_price"] - perturbed_price) / non_deals["median_price"]
    
    cap_doc = (z_doc_pert <= -1.0).mean()
    cap_hib = ((z_log_pert <= -1.0) & (disc_pert >= 0.20)).mean()
    
    capture_rates_docente.append(cap_doc)
    capture_rates_hibrido.append(cap_hib)

# Test de resistencia a falsos positivos (inflar precios +30%)
inflated_price = non_deals["price_std"] * 1.30
z_doc_inf = (inflated_price - non_deals["mean_linear"]) / non_deals["std_linear"]
z_log_inf = (np.log(inflated_price) - non_deals["mean_log"]) / non_deals["std_log"]
disc_inf = (non_deals["median_price"] - inflated_price) / non_deals["median_price"]

fp_doc = (z_doc_inf <= -1.0).mean()
fp_hib = ((z_log_inf <= -1.0) & (disc_inf >= 0.20)).mean()

# %%
# Visualización del Stress Test
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot([d * 100 for d in discounts], [r * 100 for r in capture_rates_docente], marker="o", linewidth=2, label="Z-Score Lineal (Docente)", color="navy")
ax.plot([d * 100 for d in discounts], [r * 100 for r in capture_rates_hibrido], marker="s", linewidth=2, label="Z-Score Log-Normal Híbrido (Propuesta)", color="forestgreen")

ax.set_title("Sensibilidad y Tasa de Captura ante Descuentos Sintéticos Inyectados", fontsize=13, fontweight="bold")
ax.set_xlabel("Descuento artificial aplicado al precio (%)", fontsize=11)
ax.set_ylabel("Tasa de detección como Deal (%)", fontsize=11)
ax.axvline(20, color="gray", linestyle="--", label="Umbral de ahorro mínimo (20%)")
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

print(f"Resistencia a Falsos Positivos con precio inflado (+30%):")
print(f" - Falsos Positivos Modelo Docente: {fp_doc:.4%}")
print(f" - Falsos Positivos Modelo Híbrido: {fp_hib:.4%}")

# %% [markdown]
# ## 4. Análisis de Sensibilidad: Impacto de la Granularidad de Mercado
#
# Comparamos qué sucede cuando clasificamos ofertas utilizando tres niveles de granularidad:
# 1. **Contexto Fino**: $\text{Destino} \times \text{Mes} \times \text{Semana} \times \text{Duración}$
# 2. **Contexto Medio**: $\text{Destino} \times \text{Mes}$
# 3. **Contexto Laxo**: $\text{Destino}$ únicamente (ignora estacionalidad y duración)

# %%
# Contexto Medio
stats_medio = df.groupby(["destination_final", "month"]).agg(
    median_medio=("price_std", "median"),
    mean_log_medio=("price_std", lambda x: np.log(x).mean()),
    std_log_medio=("price_std", lambda x: np.log(x).std())
).reset_index()

# Contexto Laxo
stats_laxo = df.groupby("destination_final").agg(
    median_laxo=("price_std", "median"),
    mean_log_laxo=("price_std", lambda x: np.log(x).mean()),
    std_log_laxo=("price_std", lambda x: np.log(x).std())
).reset_index()

df_sens = df.merge(stats_medio, on=["destination_final", "month"], how="left")
df_sens = df_sens.merge(stats_laxo, on="destination_final", how="left")

# Clasificación en contexto medio
z_log_medio = (np.log(df_sens["price_std"]) - df_sens["mean_log_medio"]) / df_sens["std_log_medio"].replace(0, np.nan)
disc_medio = (df_sens["median_medio"] - df_sens["price_std"]) / df_sens["median_medio"]
df_sens["deal_medio"] = ((z_log_medio <= -1.0) & (disc_medio >= 0.20)).astype(int)

# Clasificación en contexto laxo
z_log_laxo = (np.log(df_sens["price_std"]) - df_sens["mean_log_laxo"]) / df_sens["std_log_laxo"].replace(0, np.nan)
disc_laxo = (df_sens["median_laxo"] - df_sens["price_std"]) / df_sens["median_laxo"]
df_sens["deal_laxo"] = ((z_log_laxo <= -1.0) & (disc_laxo >= 0.20)).astype(int)

# Matriz de concordancia
concordancia_fino_laxo = pd.crosstab(
    df_sens["deal_hibrido"],
    df_sens["deal_laxo"],
    rownames=["Contexto Fino"],
    colnames=["Contexto Laxo"],
    normalize=True
)

print("Matriz de concordancia normalizada (Fino vs Laxo):")
print(concordancia_fino_laxo.round(4) * 100)

discrepancias = (df_sens["deal_hibrido"] != df_sens["deal_laxo"]).mean()
print(f"\nDiscrepancia total entre clasificar con contexto fino vs laxo: {discrepancias:.2%}")

# %% [markdown]
# ### Consecuencia de la pérdida de granularidad
#
# Al diluir el contexto a nivel de destino único:
# 1. En **temporada baja**, los precios normales deprimidos se marcan erróneamente como ofertas porque se los compara con la media anual inflada por el verano o vacaciones.
# 2. En **temporada alta**, ofertas legítimas de último minuto no se detectan porque la media anual es más baja que los precios de pico.
# 3. La granularidad fina retiene la señal pura de oportunidad económica dentro de la ventana de viaje real del usuario.

# %% [markdown]
# ## 5. Síntesis y Conclusiones del TP3
#
# | Dimensión | Z-Score Docente (Lineal) | Z-Score Log-Normal Híbrido | Random Forest (Supervisado) |
# |---|---|---|---|
# | **Supuesto distributivo** | Simétrica Normal $\mathcal{N}(\mu, \sigma)$ | Log-Normal asimétrica sesgada a derecha | No paramétrico (árboles de decisión) |
# | **Tasa de detección** | ~8.9% | ~14.1% | Ajustable por umbral (default 12.8%) |
# | **Sensibilidad a hoteles caros** | Alta (la varianza se infla y oculta ofertas) | Nula (distancia logarítmica proporcional) | Baja |
# | **Ahorro económico garantizado** | No garantizado | Sí (filtro explícito $\ge 20\%$ s/ mediana) | Correlacionado con features de precio |
# | **Velocidad de ejecución** | Inmediata ($O(1)$ lookup) | Inmediata ($O(1)$ lookup) | Inferencia matricial ($< 1$ ms) |
#
# **Decisión final**: El clasificador adoptado para producción y para la aplicación web es el **Z-Score Log-Normal Híbrido**, utilizando el **Random Forest** como modelo complementario de predicción temprana cuando la celda contextual carece de histórico consolidado ($N < 30$).
