"""
07c - Análisis detallado de mappings problemáticos
Para cada destino con score bajo, muestra contexto para decidir:
mantener, separar o reasignar.
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

# --- Cargar datos ---
df_hist = pd.read_csv(str(Path(__file__).resolve().parent.parent.parent / 'data' / 'datos_historicos_2024.csv'), nrows=200000)
df_hist['date_start'] = pd.to_datetime(df_hist['date_start'])
df_hist['month'] = df_hist['date_start'].dt.month
df_hist['price_std'] = df_hist['avg_price_average'] / (
    df_hist['nights'] * df_hist['number_of_rooms'] * (df_hist['number_of_adults'] + df_hist['number_of_kids']).clip(lower=1)
)
df_hist['country_code'] = df_hist['country_code'].str.upper()
mapping_df = load_destination_mapping()
df_hist = apply_destination_mapping(df_hist, mapping_df)
destinos_validar = df_hist.groupby('destination_name').filter(lambda x: len(x) >= 100)

# --- Calcular scores ---
resultados = []
for dest_name, grp in destinos_validar.groupby('destination_name'):
    if grp['city'].nunique() < 2:
        continue
    pcm = grp.groupby(['city', 'month'])['price_std'].mean().unstack(fill_value=np.nan)
    if pcm.shape[0] < 2 or pcm.shape[1] < 3:
        continue
    cv = (pcm.std() / pcm.mean()).mean()
    cm = pcm.T.corr()
    n = cm.shape[0]
    corr = (cm.values.sum() - n) / (n * (n - 1))
    resultados.append({
        'destino': dest_name,
        'n_ciudades': grp['city'].nunique(),
        'cv': cv,
        'corr': corr,
        'precio_prom': grp['price_std'].mean(),
        'n_obs': len(grp)
    })

res = pd.DataFrame(resultados)
res['score'] = ((1/(1+res['cv'])) + ((res['corr']+1)/2)) / 2
res = res.sort_values('score')

# --- Analizar cada uno ---
peores = res[res['score'] < 0.65].copy()  # Umbral: score < 0.65

reporte = []
for _, row in peores.iterrows():
    dest = row['destino']
    grp = destinos_validar[destinos_validar['destination_name'] == dest]
    
    # Info por ciudad
    info_ciudades = grp.groupby('city').agg(
        n_obs=('price_std', 'count'),
        precio_mean=('price_std', 'mean'),
        precio_std=('price_std', 'std'),
        hotel_count_mean=('avg_hotel_count', 'mean')
    ).sort_values('precio_mean')
    
    ciudades = info_ciudades.index.tolist()
    precios = info_ciudades['precio_mean'].values
    ratio_precio = precios.max() / precios.min() if precios.min() > 0 else float('inf')
    
    # Buscar candidatos alternativos (destinos cercanos con precio similar)
    precio_grupo = grp['price_std'].mean()
    otros_destinos = res[res['destino'] != dest].copy()
    otros_destinos['diff_precio'] = abs(otros_destinos['precio_prom'] - precio_grupo)
    candidatos = otros_destinos.nsmallest(3, 'diff_precio')
    
    reporte.append({
        'destino': dest,
        'score': row['score'],
        'cv': row['cv'],
        'corr': row['corr'],
        'ciudades': ciudades,
        'precios': [f'${p:.2f}' for p in precios],
        'ratio_precio': ratio_precio,
        'n_obs_total': int(row['n_obs']),
        'candidatos': candidatos['destino'].tolist()
    })

# --- Imprimir reporte ---
print('='*90)
print('REPORTE DE MAPPINGS PROBLEMÁTICOS (score < 0.65)')
print('='*90)

for r in reporte:
    print(f"\n{'─'*90}")
    print(f"DESTINO: {r['destino']}")
    print(f"  Score: {r['score']:.2f}  |  CV: {r['cv']:.2f}  |  Corr: {r['corr']:.2f}")
    print(f"  Observaciones totales: {r['n_obs_total']}")
    print(f"  Ciudades ({len(r['ciudades'])}):")
    for c, p in zip(r['ciudades'], r['precios']):
        print(f"    - {c:30s}  precio prom: {p}")
    print(f"  Ratio max/min: {r['ratio_precio']:.2f}x")
    print(f"  Candidatos alternativos: {', '.join(r['candidatos'])}")
    
    # Diagnóstico
    if r['ratio_precio'] > 2.0:
        diagnostico = "ALTA diferencia de precio → probablemente segmentos distintos"
    elif r['corr'] < -0.3:
        diagnostico = "Correlación negativa → patrones estacionales opuestos"
    elif r['cv'] > 0.4:
        diagnostico = "Alta variabilidad → ciudades con dinámicas distintas"
    else:
        diagnostico = "Diferencia moderada"
    print(f"  Diagnóstico: {diagnostico}")

# --- Guardar reporte ---
with open(DATA_DIR / '07c_reporte_mappings.txt', 'w') as f:
    f.write('REPORTE DE MAPPINGS PROBLEMÁTICOS\n')
    f.write('='*90 + '\n\n')
    for r in reporte:
        f.write(f"DESTINO: {r['destino']}\n")
        f.write(f"  Score: {r['score']:.2f}  |  CV: {r['cv']:.2f}  |  Corr: {r['corr']:.2f}\n")
        f.write(f"  Ciudades: {', '.join(r['ciudades'])}\n")
        f.write(f"  Precios: {', '.join(r['precios'])}\n")
        f.write(f"  Ratio max/min: {r['ratio_precio']:.2f}x\n")
        f.write(f"  Candidatos: {', '.join(r['candidatos'])}\n\n")

print(f'\n\n✓ Reporte guardado en {DATA_DIR / "07c_reporte_mappings.txt"}')
print(f'  Total mappings problemáticos: {len(reporte)}')
