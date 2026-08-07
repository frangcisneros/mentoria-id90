"""
07 - Validación del mapping por patrones de precio
Genera un score compuesto (CV + correlación) para evaluar calidad del mapping.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from auxiliary_functions import load_destination_mapping, apply_destination_mapping

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
DATA_DIR = OUTPUT_DIR / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

print('Cargando datos...')
df_hist = pd.read_csv(str(Path(__file__).resolve().parent.parent.parent / 'data' / 'datos_historicos_2024.csv'), nrows=200000)
df_hist['date_start'] = pd.to_datetime(df_hist['date_start'])
df_hist['month'] = df_hist['date_start'].dt.month

df_hist['price_std'] = df_hist['avg_price_average'] / (
    df_hist['nights'] * df_hist['number_of_rooms'] * (df_hist['number_of_adults'] + df_hist['number_of_kids']).clip(lower=1)
)

df_hist['country_code'] = df_hist['country_code'].str.upper()
mapping_df = load_destination_mapping()
df_hist = apply_destination_mapping(df_hist, mapping_df)

print(f'Registros con mapping: {df_hist["destination_name"].notna().sum():,} / {len(df_hist):,}')
print(f'Destinos únicos: {df_hist["destination_name"].nunique()}')
print()

destinos_validar = df_hist.groupby('destination_name').filter(lambda x: len(x) >= 100)

resultados = []
for dest_name, grp in destinos_validar.groupby('destination_name'):
    if grp['city'].nunique() < 2:
        continue

    precio_ciudad_mes = grp.groupby(['city', 'month'])['price_std'].mean().unstack(fill_value=np.nan)

    if precio_ciudad_mes.shape[0] < 2 or precio_ciudad_mes.shape[1] < 3:
        continue

    cv_por_mes = precio_ciudad_mes.std() / precio_ciudad_mes.mean()
    cv_promedio = cv_por_mes.mean()

    corr_matrix = precio_ciudad_mes.T.corr()
    n = corr_matrix.shape[0]
    corr_promedio = (corr_matrix.values.sum() - n) / (n * (n - 1))

    resultados.append({
        'destino': dest_name,
        'n_ciudades': grp['city'].nunique(),
        'cv_precio': cv_promedio,
        'corr_precio': corr_promedio,
        'precio_promedio': grp['price_std'].mean()
    })

res = pd.DataFrame(resultados)

# Score compuesto: combina CV bajo + correlación alta
# cv_score: 1 / (1 + cv) → 1 cuando CV=0, ~0 cuando CV es alto
# corr_score: (corr + 1) / 2 → 1 cuando corr=1, 0 cuando corr=-1
res['cv_score'] = 1 / (1 + res['cv_precio'])
res['corr_score'] = (res['corr_precio'] + 1) / 2
res['score'] = (res['cv_score'] + res['corr_score']) / 2

res = res.sort_values('score', ascending=False)

print(f'Destinos analizados (>=2 ciudades, >=100 obs): {len(res)}')
print()

print('='*80)
print('TOP 20 - MEJOR SCORE (mapping más confiable)')
print('='*80)
print(res[['destino','n_ciudades','cv_precio','corr_precio','score']].head(20).to_string(index=False))
print()

print('='*80)
print('TOP 20 - PEOR SCORE (mapping cuestionable)')
print('='*80)
print(res[['destino','n_ciudades','cv_precio','corr_precio','score']].tail(20).to_string(index=False))
print()

print('='*80)
print('DISTRIBUCIÓN DEL SCORE')
print('='*80)
print(f'  Score mediana: {res["score"].median():.3f}')
print(f'  Score P25:     {res["score"].quantile(0.25):.3f}')
print(f'  Score P75:     {res["score"].quantile(0.75):.3f}')
print(f'  Score mínimo:  {res["score"].min():.3f}')
print(f'  Score máximo:  {res["score"].max():.3f}')
print()

buenos = res[res['score'] >= 0.6]
regulares = res[(res['score'] >= 0.4) & (res['score'] < 0.6)]
malos = res[res['score'] < 0.4]
print(f'  Score >= 0.6 (bueno):      {len(buenos)} destinos ({len(buenos)/len(res)*100:.0f}%)')
print(f'  Score 0.4-0.6 (regular):   {len(regulares)} destinos ({len(regulares)/len(res)*100:.0f}%)')
print(f'  Score < 0.4 (problema):    {len(malos)} destinos ({len(malos)/len(res)*100:.0f}%)')

res.to_csv(DATA_DIR / '07_validacion_mapping.csv', index=False)
print(f'\n✓ Resultados guardados en {DATA_DIR / "07_validacion_mapping.csv"}')
