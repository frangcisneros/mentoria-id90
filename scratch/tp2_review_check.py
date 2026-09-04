# Verificación de la comparativa de estadísticos del TP2.
# Reproduce el pipeline del notebook y cuantifica:
#   1) sigma_ln típico en mercados sólidos (para el ejemplo log de la sección 7)
#   2) qué fracción de las filas evaluadas cae en mercados degenerados/pequeños
#   3) cómo cambian los porcentajes de detección si la comparativa se restringe
#      a filas de mercados con N >= 30 (donde los percentiles son estimables)
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import config
import auxiliary_functions as af

data_dir = ROOT_DIR / "data"
mapping_df = af.load_destination_mapping(data_dir / "destination_with_nearest.csv")
df_raw = pd.read_csv(
    data_dir / "sample_data_300k.csv.gz",
    dtype={"city": str, "state": str, "country": str, "country_code": str},
    low_memory=False,
)

df = af.standardize_prices(af.validate_data(df_raw))
df["date_start"] = pd.to_datetime(df["date_start"])
df["month"] = df["date_start"].dt.month
days = df["date_start"].dt.day.values
w = np.ones(len(days), dtype=int)
w[(days >= 8) & (days <= 15)] = 2
w[(days >= 16) & (days <= 22)] = 3
w[days >= 23] = 4
df["week_in_month"] = w
df["stay_duration"] = np.where(df["nights"] <= 2, "corta",
                               np.where(df["nights"] <= 5, "media", "larga"))
df["price_std"] = df["avg_price_average_std"]

df = af.apply_destination_mapping(df, mapping_df)
price_dist = af.calculate_price_distribution_by_destination(df)
df = af.classify_observations_into_buckets(df, price_dist)

context_cols = ["destination_final", "month", "week_in_month", "stay_duration"]
df["log_price_std"] = np.log(df["price_std"].clip(lower=0.01))

grouped = df.groupby(context_cols)
tp2_base = grouped.agg(
    n_records=("price_std", "count"),
    demand_weight=("count_repeated", "sum"),
    mean_price=("price_std", "mean"),
    std_price=("price_std", "std"),
    median_price=("price_std", "median"),
    mean_log_price=("log_price_std", "mean"),
    std_log_price=("log_price_std", "std"),
).reset_index()
p_df = grouped["price_std"].quantile([0.10, 0.25, 0.50, 0.75]).unstack()
p_df.columns = ["p10", "p25", "p50", "p75"]
p_df = p_df.reset_index()
tp2_contexts = tp2_base.merge(p_df, on=context_cols)
tp2_contexts["q_lo"] = tp2_contexts["p25"] - 1.5 * (tp2_contexts["p75"] - tp2_contexts["p25"])

# --- 1) sigma_ln típico en mercados sólidos -------------------------------
solid = tp2_contexts[tp2_contexts["demand_weight"] >= 30]
print("std_log_price en mercados N>=30:")
print(solid["std_log_price"].describe().round(3).to_string())
med_sl = solid["std_log_price"].median()
print(f"\nEjemplo 25 vs 90 USD: diff log = {np.log(25) - np.log(90):.2f}; "
      f"con sigma_ln mediana {med_sl:.2f} -> z_log = {(np.log(25) - np.log(90)) / med_sl:.2f}")

# --- 2) eval_df igual que el notebook --------------------------------------
stats_to_merge = tp2_contexts[[
    "destination_final", "month", "week_in_month", "stay_duration",
    "mean_price", "std_price", "median_price", "mean_log_price", "std_log_price",
    "p10", "p25", "p75", "q_lo", "demand_weight", "n_records",
]].copy()
sample_eval = df.sample(n=min(150_000, len(df)), random_state=42).copy()
eval_df = sample_eval.merge(stats_to_merge, on=context_cols, how="left")

std_safe = eval_df["std_price"].copy()
zero_std = (std_safe <= 0) | (std_safe.isna())
std_safe[zero_std] = (eval_df.loc[zero_std, "mean_price"] * 0.10).clip(lower=5.0)
eval_df["z_docente"] = (eval_df["price_std"] - eval_df["mean_price"]) / std_safe
eval_df["flag_deal_docente"] = eval_df["z_docente"] < config.THRESHOLDS["deal"]

std_log_safe = eval_df["std_log_price"].copy()
zero_std_log = (std_log_safe <= 0) | (std_log_safe.isna())
std_log_safe[zero_std_log] = 0.20
eval_df["z_log"] = (eval_df["log_price_std"] - eval_df["mean_log_price"]) / std_log_safe
eval_df["flag_deal_log"] = eval_df["z_log"] < -1.0
eval_df["discount_ratio"] = 1.0 - (eval_df["price_std"] / eval_df["median_price"].clip(lower=1.0))
eval_df["flag_deal_hibrido"] = (eval_df["z_log"] < -1.0) & (eval_df["discount_ratio"] >= 0.20)
eval_df["flag_discount_30pct"] = eval_df["discount_ratio"] >= 0.30
eval_df["flag_p10"] = eval_df["price_std"] <= eval_df["p10"]
eval_df["flag_iqr"] = eval_df["price_std"] <= eval_df["q_lo"]

flags = ["flag_deal_docente", "flag_deal_log", "flag_deal_hibrido",
         "flag_discount_30pct", "flag_p10", "flag_iqr"]

print("\n--- % detectado: TODAS las filas vs solo filas en mercados N>=30 ---")
solid_mask = eval_df["demand_weight"] >= 30
print(f"filas en mercados solidos: {solid_mask.mean() * 100:.1f}% de la muestra evaluada")
for f in flags:
    print(f"  {f:22s} all={eval_df[f].mean() * 100:6.2f}%   solid={eval_df.loc[solid_mask, f].mean() * 100:6.2f}%")

print("\n--- filas en contextos pequenos (n_records<=2) que disparan flags ---")
tiny = eval_df["n_records"] <= 2
print(f"filas en contextos n_records<=2: {tiny.mean() * 100:.1f}%")
for f in ["flag_p10", "flag_iqr"]:
    print(f"  {f}: en tiny={eval_df.loc[tiny, f].mean() * 100:.1f}%  vs  en no-tiny={eval_df.loc[~tiny, f].mean() * 100:.1f}%")

print("\n--- IQR==0 entre filas evaluadas ---")
iqr0 = (eval_df["p75"] - eval_df["p25"]) <= 0
print(f"filas con IQR==0: {iqr0.mean() * 100:.1f}%  | flag_iqr ahi: {eval_df.loc[iqr0, 'flag_iqr'].mean() * 100:.1f}%")
