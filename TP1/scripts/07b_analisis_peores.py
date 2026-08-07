"""
Análisis de peores mappings: para cada destino con score bajo,
muestra el patrón de precio mensual de cada ciudad para ver
si hay una mejor opción de agrupación.
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from auxiliary_functions import load_destination_mapping, apply_destination_mapping

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
IMAGES_DIR = OUTPUT_DIR / 'images'
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

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

# Calcular scores
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
    })

res = pd.DataFrame(resultados)
res['cv_score'] = 1 / (1 + res['cv_precio'])
res['corr_score'] = (res['corr_precio'] + 1) / 2
res['score'] = (res['cv_score'] + res['corr_score']) / 2
res = res.sort_values('score')

# Top 10 peores
peores = res.head(10)['destino'].tolist()

print(f'\n Analizando los 10 peores scores:')
print(res.head(10)[['destino','n_ciudades','cv_precio','corr_precio','score']].to_string(index=False))

# Para cada destino malo, ver patrones de precio por ciudad
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i, dest in enumerate(peores):
    ax = axes[i]
    grp = destinos_validar[destinos_validar['destination_name'] == dest]
    precio_ciudad_mes = grp.groupby(['city', 'month'])['price_std'].mean().unstack(fill_value=np.nan)
    
    for city in precio_ciudad_mes.index:
        ax.plot(precio_ciudad_mes.columns, precio_ciudad_mes.loc[city], marker='o', label=city, alpha=0.7)
    
    ax.set_title(f'{dest}\n(score={res[res["destino"]==dest]["score"].values[0]:.2f})', fontsize=9)
    ax.set_xlabel('Mes')
    ax.set_ylabel('Precio std')
    ax.legend(fontsize=6, loc='best')
    ax.set_xticks(range(1, 13))

plt.suptitle('Patrones de precio por ciudad en destinos con PEOR score', fontsize=13)
plt.tight_layout()
plt.savefig(IMAGES_DIR / '07_peores_mappings.png', dpi=150)
plt.show()

# Para cada destino malo, sugerir candidatos basados en precio promedio similar
print('\n' + '='*80)
print('ANÁLISIS POR DESTINO: posibles mejores candidatos')
print('='*80)

for dest in peores:
    grp = destinos_validar[destinos_validar['destination_name'] == dest]
    ciudades = grp['city'].unique()
    precio_prom = grp.groupby('city')['price_std'].mean()
    precio_prom_grupo = grp['price_std'].mean()
    
    print(f'\n--- {dest} (score={res[res["destino"]==dest]["score"].values[0]:.2f}) ---')
    print(f'  Ciudades: {list(ciudades)}')
    print(f'  Precio promedio grupo: ${precio_prom_grupo:.2f}')
    for city, p in precio_prom.items():
        print(f'    {city}: ${p:.2f}')
    
    # Buscar otros destinos con precio similar
    otros = res[res['destino'] != dest].copy()
    otros['diff_precio'] = abs(otros['destino'].map(
        destinos_validar.groupby('destination_name')['price_std'].mean()
    ) - precio_prom_grupo)
    candidatos = otros.nsmallest(3, 'diff_precio')
    
    print(f'  Candidatos con precio similar:')
    for _, cand in candidatos.iterrows():
        print(f'    → {cand["destino"]} (score={cand["score"]:.2f}, precio similar)')
