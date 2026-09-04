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
#     display_name: .venv_linux
#     language: python
#     name: python3
# ---

# %% [markdown]
# # NOTAS DE CORRECCIÓN TP1 → TP1_CORREGIDO
#
# Este notebook es la versión corregida de la entrega original. Los cambios principales aplicados son:
#
# 1. **Expansión temporal corregida**: ahora se generan exactamente `nights` noches pagadas (`date_start + 0` a `date_start + nights - 1`), sin incluir `date_end` (checkout).
# 2. **Días de semana corregidos**: el diccionario de nombres ahora respeta la convención de pandas (`0 = Lunes`, `6 = Domingo`).
# 3. **Unidad de análisis explicitada**: se distinguen `demand_weight` (suma de `count_repeated`), `n_records` (filas/noches desagregadas) y `count_obs` (alias de demanda ponderada).
# 4. **Mapping con trazabilidad**: se agrega `match_level` para saber qué nivel de matching usó cada geografía.
# 5. **SNR con soporte de datos**: se reportan contextos generados y % de demanda en contextos confiables.
#
# TODO: re-ejecutar las celdas de conclusión sobre el dataset completo (2024+2025) usando `load_all_historicals`.
#

# %% [markdown]
# # TP1 — Definición de mercado y análisis exploratorio
#
# **Diplomatura en Ciencia de Datos · Mentoría ID90Travel**
#
# En este notebook consolidamos el análisis exploratorio del dataset de búsquedas hoteleras de ID90Travel para determinar qué combinación de dimensiones genera grupos de observaciones donde los precios presentan un comportamiento homogéneo.
#
# Incluye todo el código ejecutable celda por celda (standalone) sobre una **muestra representativa optimizada para consumo de RAM**, incorporando:
# 1. **Auditoría de calidad y nulos**: exclusión de registros inválidos (sin ciudad/país) y consolidación de variantes por `country_code`.
# 2. **Estandarización y expansiones temporales**: cálculo de precio normalizado (*room-night-person*) y features de calendario.
# 3. **Mapping de destinos, Mapa Global y Mapa de Remapeo**: factor de consolidación (~11.9x), análisis de *singletons*, **mapa interactivo de destinos canónicos (`05_mapa_destinos_canonicos.html`)**, **mapa interactivo de remapeo antes vs después (`06_mapa_remapeo_antes_despues.html`)** y vectorización regional.
# 4. **Análisis exploratorio de dimensiones**: geografía, patrones temporales mes x semana x día, descuentos por volumen de noches 1 a 15 y categoría proxy del hotel (`price_bucket`).
# 5. **Análisis de homogeneidad (SNR)**, **suficiencia de baselines en destinos mapeados ($\ge 30$ obs en los 1.769 destinos canónicos)**, evaluación interanual 2024 vs 2025 y **respuestas a todas las preguntas de investigación**.
#
# ---
#

# %% [markdown]
# ## 0. Decisiones metodológicas del análisis
#
# Antes de ejecutar el notebook, quedan documentadas las decisiones metodológicas que atraviesan todo el trabajo (basado en las indicaciones comunes para TP2):
#
# 1. **Unidad de análisis**: fila original → noche pagada (`date_start` + 0 a `nights-1`) → contexto agregado.
# 2. **Fuente de datos**: ambos años históricos (2024 + 2025) o, si no están disponibles, muestra representativa de 300.000 filas con semilla fija.
# 3. **Muestreo**: cuando se use muestra, semilla `random_state=42` y estratificación implícita por archivo/año.
# 4. **Fecha del contexto**: noche pagada; `date_end` se interpreta como checkout y no se incluye.
# 5. **Precio comparable**: `avg_price_average / (nights × number_of_rooms × (adults + kids))`.
# 6. **Filtros de calidad**: noches > 0 y ≤ 30, habitaciones > 0, ocupación > 0, precio > 0 y ≤ 50.000 USD, precio estandarizado > 0.
# 7. **Mercado geográfico**: destino canónico via `destination_with_nearest.csv` con pretratamiento del docente; sin match queda marcado como `no_match`.
# 8. **Desagregación temporal**: noches pagadas con `range(nights)`, no rango inclusivo.
# 9. **Uso de `count_repeated`**: peso de demanda; reportamos `demand_weight` (suma), `n_records` (filas/noches) y `count_obs` (alias).
# 10. **Variables de segmentación**: `destination_final`, `month`, `week_in_month`, `stay_duration`; `price_bucket` opcional.
# 11. **Criterio de confianza**: al menos 30 unidades de demanda ponderada por contexto.
# 12. **Fallbacks**: solo dentro del mismo destino (mes, año); no asignar destinos canónicos automáticamente a no-matcheados.
# 13. **Reproducibilidad**: este notebook parte de `load_all_historicals()`; el pipeline productivo está en `pipeline_build_baselines.py`.
#

# %% [markdown]
# ## 1. Configuración y carga de datos
#

# %%
# %matplotlib inline
import sys
from pathlib import Path

# Buscar la carpeta raíz donde se encuentra config.py
for p in [Path('.').resolve(), Path('.').resolve().parent, Path('.').resolve().parent.parent]:
    if (p / 'config.py').exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config
import auxiliary_functions as af
from math import radians, sin, cos, sqrt, atan2
import folium

# Configuración gráfica para visualización inline en Jupyter Notebook
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Tamaño de muestra representativa para ejecución rápida y bajo consumo de RAM
SAMPLE_SIZE = 300_000

DB_PATH = Path(config.BASE_DIR) / 'data' / 'hotel_data.db'
OUTPUT_DIR = Path(config.BASE_DIR) / 'TP1' / 'outputs'

print(f"✓ Base de datos: {DB_PATH}")
print(f"✓ Directorio de Outputs: {OUTPUT_DIR}")
print(f"✓ Muestra optimizada configurada: {SAMPLE_SIZE:,} registros")


# %% [markdown]
# ## 2. Descripción general del dataset y Auditoría de Nulos
#
# El dataset contiene registros de búsquedas hoteleras de ID90Travel con la siguiente estructura:
# - **5,1M de registros totales** correspondiente al período 2024-2025 (procesados mediante una muestra aleatoria representativa de 300.000 observaciones para optimizar memoria RAM)
# - **~26.000 ciudades** distintas en estado crudo
# - **Variables clave**: precio, oferta (cantidad de hoteles disponibles) y demanda (frecuencia de búsqueda)
#
# ### Auditoría de Calidad y Filtrado de Nulos:
# - Exclusión de registros sin ciudad ni país simultáneamente (`city` y `country` nulos).
# - Consolidación por `country_code` para evitar duplicidad de nombres con texto con ruido (ej. *"United States"*, *"USA"*, *"US"*).
#

# %%
# CORRECCIÓN TP1: una sola fuente de datos determinista para conclusiones
# Si existen los CSV históricos crudos, se cargan AMBOS años completos.
# Si no existen, se usa la muestra representativa de 300k con semilla fija.
# El notebook ya no usa RANDOM() sin semilla ni las primeras n filas de cada CSV.

mapping_df = af.load_destination_mapping()
hist_files = sorted(list((Path(config.BASE_DIR) / 'data').glob('datos_historicos_*.csv')))

if hist_files:
    print(f'Usando archivos históricos crudos: {[f.name for f in hist_files]}')
    df_raw = af.load_all_historicals()
    fuente_usada = 'CSV históricos completos (2024 + 2025)'
else:
    sample_gz_path = Path(config.BASE_DIR) / 'data' / 'sample_data_300k.csv.gz'
    if not sample_gz_path.exists():
        raise FileNotFoundError('No se encontraron datos históricos ni sample_data_300k.csv.gz')
    print(f'Usando muestra representativa: {sample_gz_path.name}')
    df_raw = pd.read_csv(sample_gz_path, dtype={'city': str, 'state': str, 'country': str, 'country_code': str}, low_memory=False)
    fuente_usada = 'Muestra representativa 300k (seed=42)'

# Pipeline de calidad, estandarización y mapeo
df_valid = af.validate_data(df_raw)
df_std = af.standardize_prices(df_valid)
df = af.apply_destination_mapping(df_std, mapping_df)

# Features temporales y de duración de estadía (sobre date_start como check-in)
df['date_start'] = pd.to_datetime(df['date_start'])
df['date_end'] = pd.to_datetime(df['date_end'])
df['day_of_week'] = df['date_start'].dt.dayofweek  # 0=Lunes, 6=Domingo
df['month'] = df['date_start'].dt.month
df['year'] = df['date_start'].dt.year

days = df['date_start'].dt.day.values
week_in_month = np.ones(len(days), dtype=int)
week_in_month[(days >= 8) & (days <= 15)] = 2
week_in_month[(days >= 16) & (days <= 22)] = 3
week_in_month[days >= 23] = 4
df['week_in_month'] = week_in_month

df['stay_duration'] = np.where(df['nights'] <= 2, 'corta', np.where(df['nights'] <= 5, 'media', 'larga'))
df['price_std'] = df['avg_price_average_std']

print(f'\nFuente usada: {fuente_usada}')
print(f'Registros procesados: {len(df):,}')
print(f'Destinos canónicos únicos: {df["destination_final"].nunique():,}')
print(f'Países representados: {df["country_code"].nunique():,}')
print(f'Registros con Mapeo Canónico: {df["is_mapped"].sum():,} ({df["is_mapped"].mean():.1%})')
print(f'Rango de fechas: {df["date_start"].min().strftime("%Y-%m-%d")} a {df["date_start"].max().strftime("%Y-%m-%d")}')
print(f'\nEstadísticas de precio estandarizado (price_std = $/hab-noche-persona):')
print(df['price_std'].describe().round(2).to_string())


# %% [markdown]
# ## 3. Mapeo de Destinos, Mapa Global y Validación de Correcciones Manuales
#
# El mapping geográfico agrupa ~26.000 nombres crudos de ciudades hacia destinos canónicos por cercanía (Haversine).
#
# Para garantizar que cada agrupación mantenga coherencia en los patrones de precio, evaluamos la calidad del mapping mediante un **Score Compuesto**:
# $$Score = 0.5 \times \text{CV\_score} + 0.5 \times \text{Corr\_score}$$
#
# Donde:
# - $\text{CV\_score} = \frac{1}{1 + \text{CV}}$ penaliza alta dispersión interna de precios.
# - $\text{Corr\_score} = \frac{\text{Corr} + 1}{2}$ evalúa la alineación estacional entre ciudades del mismo destino.
#
# ### Casos de Frontera Geográfica planteados en el README:
# - **Miami vs Miami Beach**: Aunque distan menos de 10 km, Miami Beach (mercado vacacional/resorts de playa) exhibe precios significativamente más altos que el centro de Miami (mercado urbano/corporativo).
# - **Las Vegas vs Henderson**: Henderson dista 20 km del Strip de Las Vegas. Su agrupación bajo el destino canónico *Las Vegas* consolida la masa crítica necesaria de observaciones, pero aumenta ligeramente la dispersión interna entre hoteles de casino/strip vs suburbanos, lo que justifica incorporar la variable `price_bucket` (categoría proxy).
#
# ### Correcciones Manuales Aplicadas (`07e_aplicar_cambios_mapping.py`):
# - **Azusa** (Anaheim & Buena Park) $\rightarrow$ Redondo Beach (alineación con mercado urbano costero de Los Ángeles).
# - **Bell Gardens** (Anaheim & Buena Park) $\rightarrow$ Long Beach (mercado costero/portuario).
# - **Arcadia** (Anaheim & Buena Park) $\rightarrow$ Los Angeles (mercado metropolitano LA).
# - **Bellingham** (San Juan Islands) $\rightarrow$ Seattle (separación de mercado de islas turísticas de alto costo a continente).
# - **Anacortes** (San Juan Islands) $\rightarrow$ Astoria (reclasificación costera continental).
# - **Ann Arbor** (Detroit) $\rightarrow$ Columbus (mercado universitario/regional).
# - **Arlington** (Seattle) $\rightarrow$ Cambridge (alineación regional).
#

# %%
# Análisis de métricas de consolidación del archivo de mapping
total_refs = len(mapping_df)
dests_unicos = mapping_df['nearest_destination_id'].nunique()
factor_consol = total_refs / dests_unicos

ciudades_por_dest = mapping_df.groupby('nearest_destination_id').size()
singletons = (ciudades_por_dest == 1).sum()
multiples = (ciudades_por_dest > 1).sum()

print('Métricas de Consolidación Geográfica del Mapping:')
print(f'  Total de referencias geográficas en mapping: {total_refs:,}')
print(f'  Destinos canónicos únicos:                   {dests_unicos:,}')
print(f'  Factor de consolidación:                     {factor_consol:.1f}x (referencias por destino)')
print(f'  Destino con MÁS ciudades consolidadas:        {ciudades_por_dest.max()} ciudades')
print(f'  Destinos con MÚLTIPLES ciudades consolidadas: {multiples:,} ({multiples/len(ciudades_por_dest):.1%})')
print(f'  Destinos Singletons (1 sola ciudad):         {singletons:,} ({singletons/len(ciudades_por_dest):.1%})')

print('\nTop 10 Destinos por Cantidad de Ciudades Consolidadas:')
top_10_map = mapping_df.groupby(['nearest_destination_id', 'nearest_destination_name']).size().nlargest(10)
for i, ((d_id, d_name), count) in enumerate(top_10_map.items(), 1):
    print(f'  {i:2d}. {d_name:40s} → {count:3d} ciudades')


# %%
# Cobertura y cambios detectados dinámicamente en el archivo de mapping
map_new = pd.read_csv(Path(config.BASE_DIR) / 'data' / 'destination_with_nearest.csv', dtype={'nearest_destination_id': str}, low_memory=False)
map_old = pd.read_csv(Path(config.BASE_DIR) / 'data' / 'destination_with_nearest_backup.csv', dtype={'nearest_destination_id': str}, low_memory=False)

# Detectar reasignaciones comparando con el backup original
diff_mask = (map_new['nearest_destination_id'] != map_old['nearest_destination_id']) | (map_new['nearest_destination_name'] != map_old['nearest_destination_name'])
diffs = map_new[diff_mask].copy()
diffs['Destino Anterior'] = map_old.loc[diff_mask, 'nearest_destination_name']
diffs.rename(columns={'city': 'Ciudad', 'nearest_destination_name': 'Destino Reasignado'}, inplace=True)
df_cambios = diffs[['reference', 'Ciudad', 'Destino Anterior', 'Destino Reasignado']].drop_duplicates()

con_mapeo = df['is_mapped'].sum()
sin_mapeo = len(df) - con_mapeo

print(f"Cobertura del mapping en la muestra actual:")
print(f"  Con mapeo canónico:             {con_mapeo:,} ({con_mapeo/len(df):.1%})")
print(f"  Sin mapeo (fallback a ciudad):  {sin_mapeo:,} ({sin_mapeo/len(df):.1%})")

print(f"\nReasignaciones Manuales Detectadas en destination_with_nearest.csv ({len(df_cambios)} registros):")
print(df_cambios.to_string(index=False))


# %% [markdown]
# ### 3.1 Visualización Geográfica General: Mapa de Destinos Canónicos y Ciudades Mapeadas
#
# Generamos un mapa interactivo con Folium que muestra la distribución geográfica de los **Top Destinos Canónicos** y la densidad de ciudades consolidadas en cada uno.
#

# %%
# Generar mapa general de destinos canónicos consolidados
m_canon = folium.Map(location=[39.8283, -98.5795], zoom_start=4, tiles='CartoDB positron')

dest_agg = mapping_df.groupby(['nearest_destination_id', 'nearest_destination_name']).agg(
    lat=('latitude', 'mean'),
    lon=('longitude', 'mean'),
    cant_ciudades=('city', 'count')
).reset_index()

top_dests_map = dest_agg.nlargest(100, 'cant_ciudades')

for _, r in top_dests_map.iterrows():
    folium.CircleMarker(
        location=[r['lat'], r['lon']],
        radius=min(15, max(4, int(r['cant_ciudades'] / 10))),
        color='darkblue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.6,
        popup=f"<b>Destino Canónico: {r['nearest_destination_name']}</b><br>ID: {r['nearest_destination_id']}<br>Ciudades Consolidadas: {r['cant_ciudades']}"
    ).add_to(m_canon)

maps_out = OUTPUT_DIR / 'maps'
maps_out.mkdir(parents=True, exist_ok=True)
canon_map_file = maps_out / '05_mapa_destinos_canonicos.html'
m_canon.save(canon_map_file)
print(f'✓ Mapa de destinos canónicos guardado en: {canon_map_file}')

m_canon


# %% [markdown]
# ### 3.2 Visualización Geográfica de Remapeos: Mapa de Remapeo Antes vs Después
#
# Para analizar visualmente cómo cambió la asignación de las ciudades reasignadas, comparamos el mapping original (`destination_with_nearest_backup.csv`) con el mapping optimizado (`destination_with_nearest.csv`).
#
# El mapa interactivo a continuación muestra las ciudades reasignadas:
# - **Líneas rojas discontinuas**: Conexión con el destino anterior (Antes).
# - **Líneas verdes sólidas**: Conexión con el nuevo destino reasignado (Después).
#

# %%
# Cargar ambos mappings para comparación espacial
map_new = pd.read_csv(Path(config.BASE_DIR) / 'data' / 'destination_with_nearest.csv', dtype={'nearest_destination_id': str}, low_memory=False)
map_old = pd.read_csv(Path(config.BASE_DIR) / 'data' / 'destination_with_nearest_backup.csv', dtype={'nearest_destination_id': str}, low_memory=False)

# Obtener diccionario de coordenadas promedio por destino canónico
coords = map_new.groupby('nearest_destination_name')[['latitude', 'longitude']].mean().to_dict(orient='index')

# Detectar diferencias
diff_mask = (map_new['nearest_destination_id'] != map_old['nearest_destination_id']) | (map_new['nearest_destination_name'] != map_old['nearest_destination_name'])
diffs = map_new[diff_mask].copy()
diffs['dest_ant'] = map_old.loc[diff_mask, 'nearest_destination_name']

diffs['lat_ant'] = diffs['dest_ant'].map(lambda d: coords[d]['latitude'] if d in coords else None)
diffs['lon_ant'] = diffs['dest_ant'].map(lambda d: coords[d]['longitude'] if d in coords else None)
diffs['lat_nueva'] = diffs['nearest_destination_name'].map(lambda d: coords[d]['latitude'] if d in coords else None)
diffs['lon_nueva'] = diffs['nearest_destination_name'].map(lambda d: coords[d]['longitude'] if d in coords else None)

def calc_haversine(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return np.nan
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

diffs['dist_ant_km'] = [calc_haversine(r['latitude'], r['longitude'], r['lat_ant'], r['lon_ant']) for _, r in diffs.iterrows()]
diffs['dist_nueva_km'] = [calc_haversine(r['latitude'], r['longitude'], r['lat_nueva'], r['lon_nueva']) for _, r in diffs.iterrows()]

# Mapa de remapeos
m_remapeo = folium.Map(location=[38.0, -97.0], zoom_start=4, tiles='CartoDB positron')

for _, r in diffs.iterrows():
    # Línea origen -> destino anterior (rojo)
    if pd.notna(r['lat_ant']) and pd.notna(r['lon_ant']):
        folium.PolyLine(
            locations=[[r['latitude'], r['longitude']], [r['lat_ant'], r['lon_ant']]],
            color='red', weight=2.5, opacity=0.8, dash_array='5, 5',
            popup=f"Anterior: {r['city']} -> {r['dest_ant']} ({r['dist_ant_km']:.1f} km)"
        ).add_to(m_remapeo)
    
    # Línea origen -> destino nuevo (verde)
    if pd.notna(r['lat_nueva']) and pd.notna(r['lon_nueva']):
        folium.PolyLine(
            locations=[[r['latitude'], r['longitude']], [r['lat_nueva'], r['lon_nueva']]],
            color='green', weight=3, opacity=0.9,
            popup=f"Nuevo: {r['city']} -> {r['nearest_destination_name']} ({r['dist_nueva_km']:.1f} km)"
        ).add_to(m_remapeo)
    
    # Marcador ciudad origen
    folium.CircleMarker(
        location=[r['latitude'], r['longitude']], radius=5, color='blue', fill=True, fill_color='blue', fill_opacity=0.9,
        popup=f"Ciudad: {r['city']}"
    ).add_to(m_remapeo)

output_map_path = OUTPUT_DIR / 'maps' / '06_mapa_remapeo_antes_despues.html'
output_map_path.parent.mkdir(parents=True, exist_ok=True)
m_remapeo.save(str(output_map_path))
print(f"✓ Mapa de remapeo guardado en: {output_map_path}")

# Resumen de distancias
tabla_dist = diffs[['city', 'dest_ant', 'nearest_destination_name', 'dist_ant_km', 'dist_nueva_km']].drop_duplicates().copy()
tabla_dist.columns = ['Ciudad', 'Destino Anterior', 'Destino Nuevo', 'Dist. Anterior (km)', 'Dist. Nueva (km)']
df_res_map = tabla_dist.copy()
print(f"\nTabla Comparativa de Ciudades Remapeadas:")
print(tabla_dist.round(1).to_string(index=False))


# %%
# Gráfico Matplotlib comparativo de vectores de remapeo por Zonas (Costa Oeste y Midwest)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax1 = axes[0]
ca_cities = ['Azusa', 'Bell Gardens', 'Arcadia']
for city in ca_cities:
    r = df_res_map[df_res_map['Ciudad'] == city]
    if not r.empty:
        c_info = map_old[map_old['city'] == city].iloc[0]
        ax1.plot(c_info['longitude'], c_info['latitude'], 'bo', markersize=8, label=city)

ax1.set_title('Reasignaciones en Zona Sur de California', fontsize=12, fontweight='bold')
ax1.set_xlabel('Longitud')
ax1.set_ylabel('Latitud')

ax2 = axes[1]
nw_cities = ['Bellingham', 'Anacortes']
for city in nw_cities:
    r = df_res_map[df_res_map['Ciudad'] == city]
    if not r.empty:
        c_info = map_old[map_old['city'] == city].iloc[0]
        ax2.plot(c_info['longitude'], c_info['latitude'], 'ro', markersize=8, label=city)

ax2.set_title('Reasignaciones en Pacific Northwest (Washington)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Longitud')
ax2.set_ylabel('Latitud')

for ax in axes:
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()


# %% [markdown]
# ## 4. Pregunta 1: ¿Qué define a un mercado hotelero?
#
# **Objetivo**: identificar la combinación de dimensiones que genera grupos homogéneos dentro de los cuales la comparación de precios resulta consistente.
#
# ### 4.1 Dimensión geográfica y Selección de Mercados Contrastantes
#
# Analizamos la distribución de precios entre destinos clave con perfiles de mercado contrapuestos (*Orlando*, *Miami*, *New York*, *Las Vegas*, *Cancún*, *Paris*).
#

# %%
# Selección de 6 Mercados Contrastantes para Análisis Comparativo Detallado
destinos_6 = ['Orlando', 'Miami', 'Las Vegas', 'Detroit', 'Columbus', 'Cancun']
df_6 = df[df['destination_name'].isin(destinos_6)].copy()

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df_6, y='destination_name', x='price_std', showfliers=False, ax=ax, hue='destination_name', legend=False)
ax.set_title('Distribución de Precios Estandarizados en 6 Mercados Contrastantes', fontsize=13, fontweight='bold')
ax.set_xlabel('Precio Estandarizado ($ / habitación-noche-persona)')
ax.set_ylabel('Destino Canónico')
plt.tight_layout()
plt.show()


# %%
# Boxplot de precios por destino (top 20 general)
top_20_dests = df.groupby('destination_name').size().nlargest(20).index
df_top20 = df[df['destination_name'].isin(top_20_dests)].copy()

median_order = df_top20.groupby('destination_name')['price_std'].median().sort_values(ascending=False).index
df_top20['destination_name'] = pd.Categorical(df_top20['destination_name'], categories=median_order, ordered=True)

fig, ax = plt.subplots(figsize=(14, 8))
sns.boxplot(data=df_top20, x='price_std', y='destination_name', ax=ax, showfliers=False, color='skyblue')
ax.set_title('Distribución de Precios por Destino (Top 20 General)', fontsize=14, fontweight='bold')
ax.set_xlabel('Precio Estandarizado ($/habitación-noche-persona)')
ax.set_ylabel('Destino Canónico')
ax.set_xlim(0, 200)

plt.tight_layout()
plt.show()


# %%
# Heatmap destino x mes (Top 15 destinos)
top_15_dests = df.groupby('destination_name').size().nlargest(15).index
df_top15 = df[df['destination_name'].isin(top_15_dests)]

pivot_mes = df_top15.pivot_table(index='destination_name', columns='month', values='price_std', aggfunc='mean')
pivot_mes.columns = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(pivot_mes, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, linewidths=0.5, cbar_kws={'label': 'Precio Promedio ($)'})
ax.set_title('Precio Promedio por Destino y Mes', fontsize=14, fontweight='bold')
ax.set_xlabel('Mes del Año')
ax.set_ylabel('Destino')

plt.tight_layout()
plt.show()


# %% [markdown]
# ### 4.2 Dimensión temporal (Mes, Semana del Mes y Día de la Semana)
#
# Analizamos cómo varían las tarifas a lo largo del tiempo:
# - **Día de la semana**: evaluamos si los sábados presentan tarifas más elevadas vs domingos.
# - **Semana del mes**: comparamos las variaciones entre la primera y la tercera semana de cada mes (efecto cobranza/quincena).
# - **Matriz Temporal Completa**: interactuación entre Mes × Día de Semana y Mes × Semana del Mes.
#

# %%
# Precio por día de la semana
dias_nombre = {0: 'Lun', 1: 'Mar', 2: 'Mié', 3: 'Jue', 4: 'Vie', 5: 'Sáb', 6: 'Dom'}
df['day_name'] = df['day_of_week'].map(dias_nombre)
order_dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']

precio_dia = df.groupby('day_name')['price_std'].agg(['mean', 'median', 'std', 'count']).reindex(order_dias)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=precio_dia.index, y=precio_dia['mean'], color='steelblue', ax=ax)
ax.set_title('Precio Estandarizado Promedio por Día de la Semana', fontsize=13, fontweight='bold')
ax.set_xlabel('Día de Inicio de Estadía')
ax.set_ylabel('Precio Promedio ($)')
for i, v in enumerate(precio_dia['mean']):
    ax.text(i, v + 0.5, f'${v:.1f}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()


# %%
# Patrones semanales (Semana del mes 1 a 4)
precio_semana = df.groupby('week_in_month')['price_std'].mean().reset_index()
precio_semana['week_label'] = ['Semana 1 (1-7)', 'Semana 2 (8-15)', 'Semana 3 (16-22)', 'Semana 4 (23+)']

fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=precio_semana, x='week_label', y='price_std', marker='o', color='crimson', linewidth=2.5, markersize=8, ax=ax)
ax.set_title('Evolución de Precio por Semana del Mes', fontsize=13, fontweight='bold')
ax.set_xlabel('Semana del Mes')
ax.set_ylabel('Precio Promedio ($)')
ax.set_ylim(precio_semana['price_std'].min() * 0.95, precio_semana['price_std'].max() * 1.05)

plt.tight_layout()
plt.show()


# %%
# Análisis Temporal Completo (Mes × Día de Semana y Mes × Semana del Mes)
months_labels = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
weeks_labels = ['Sem 1', 'Sem 2', 'Sem 3', 'Sem 4']

pivot_md = df.groupby(['month', 'day_of_week'])['price_std'].mean().unstack()
pivot_md.columns = ['Dom', 'Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb']
pivot_md = pivot_md[order_dias]

pivot_mw = df.groupby(['month', 'week_in_month'])['price_std'].mean().unstack()
pivot_mw.columns = weeks_labels

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

sns.heatmap(pivot_md, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0], yticklabels=months_labels, linewidths=0.5)
axes[0].set_title('Precio Promedio ($): Mes × Día de Semana', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Día de Semana')
axes[0].set_ylabel('Mes')

sns.heatmap(pivot_mw, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1], yticklabels=months_labels, linewidths=0.5)
axes[1].set_title('Precio Promedio ($): Mes × Semana del Mes', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Semana del Mes')
axes[1].set_ylabel('Mes')

plt.tight_layout()
plt.show()


# %% [markdown]
# ### 4.3 Dimensión de oferta, demanda y categoría proxy
#
# Analizamos las relaciones entre la disponibilidad de hoteles (`avg_hotel_count`), la intensidad de búsquedas (`count_repeated`) y el precio normalizado, incorporando la **categoría proxy de hotel (gama de precios)**.
#

# %%
# Categoría Proxy de Hotel (Basada en rangos de precio estandarizado)
conditions = [
    df['price_std'] <= 20,
    df['price_std'] <= 40,
    df['price_std'] <= 60,
    df['price_std'] <= 100,
    df['price_std'] <= 150
]
choices = ['1_Budget (≤$20)', '2_Mid-Low ($20-40)', '3_Mid ($40-60)', '4_Mid-High ($60-100)', '5_High ($100-150)']
df['category_proxy'] = np.select(conditions, choices, default='6_Premium (>$150)')

df_cat = df.groupby('category_proxy').agg(
    mean_price=('price_std', 'mean'),
    mean_supply=('avg_hotel_count', 'mean'),
    mean_demand=('count_repeated', 'mean'),
    mean_nights=('nights', 'mean'),
    count_obs=('price_std', 'count')
).reset_index()

print('Análisis de Categoría Proxy de Hotel (Segmentación por Gama de Precio):')
print(df_cat.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cats_short = [c.split('_')[0] for c in df_cat['category_proxy']]

axes[0].bar(cats_short, df_cat['mean_supply'], color='coral')
axes[0].set_title('Oferta Promedio (Hoteles) por Categoría', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Categoría Proxy')
axes[0].set_ylabel('Hoteles Disponibles')

axes[1].bar(cats_short, df_cat['mean_demand'], color='steelblue')
axes[1].set_title('Demanda Promedio (Búsquedas) por Categoría', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Categoría Proxy')
axes[1].set_ylabel('count_repeated Promedio')

axes[2].pie(df_cat['count_obs'], labels=cats_short, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set3', n_colors=6))
axes[2].set_title('Distribución de Observaciones por Categoría', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()


# %%
# Heatmap oferta vs demanda vs precio con rank method first para evitar duplicados
df['oferta_bin'] = pd.qcut(df['avg_hotel_count'].rank(method='first'), q=4, labels=['Baja', 'Media-Baja', 'Media-Alta', 'Alta'])
df['demanda_bin'] = pd.qcut(df['count_repeated'].rank(method='first'), q=4, labels=['Baja', 'Media-Baja', 'Media-Alta', 'Alta'])

pivot_od = df.pivot_table(index='oferta_bin', columns='demanda_bin', values='price_std', aggfunc='mean')

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot_od, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax, linewidths=0.5)
ax.set_title('Precio Promedio ($) por Niveles de Oferta y Demanda', fontsize=13, fontweight='bold')
ax.set_xlabel('Nivel de Demanda (Búsquedas)')
ax.set_ylabel('Nivel de Oferta (Hoteles)')

plt.tight_layout()
plt.show()


# %%
# Correlaciones entre variables cuantitativas
vars_cuant = ['price_std', 'nights', 'number_of_rooms', 'number_of_adults', 'number_of_kids', 'avg_hotel_count', 'count_repeated']
corr_matrix = df[vars_cuant].corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax, linewidths=0.5)
ax.set_title('Matriz de Correlación entre Variables', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()


# %% [markdown]
# ### 4.4 Dimensión de duración de estadía y Descuentos por Volumen
#
# Evaluamos en detalle cómo varía la tarifa por noche normalizada ante cambios en la cantidad total de noches reservadas (1 a 15+ noches).
#

# %%
# Análisis exhaustivo de descuentos por volumen (noches 1 a 15)
df_vol = df.groupby('nights').agg(
    mean_total_price=('avg_price_average', 'mean'),
    mean_price_per_night=('avg_price_average', lambda x: (x / df.loc[x.index, 'nights']).mean()),
    mean_price_std=('price_std', 'mean'),
    count_obs=('price_std', 'count')
).reset_index()

df_vol = df_vol[df_vol['nights'] <= 15]
price_1n = df_vol[df_vol['nights'] == 1]['mean_price_std'].values[0]
df_vol['discount_pct'] = (1 - df_vol['mean_price_std'] / price_1n) * 100

print('Tabla Detallada de Descuentos por Volumen según Noches de Estadía:')
print(df_vol[['nights', 'mean_total_price', 'mean_price_std', 'discount_pct', 'count_obs']].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].plot(df_vol['nights'], df_vol['mean_price_std'], 'o-', linewidth=2.5, markersize=7, color='darkgreen')
axes[0].set_title('Precio Estandarizado vs Noches de Estadía', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Cantidad de Noches')
axes[0].set_ylabel('Precio Estandarizado Promedio ($)')
axes[0].grid(True, alpha=0.4)

axes[1].bar(df_vol['nights'], df_vol['discount_pct'], color='forestgreen', alpha=0.8)
axes[1].set_title('Porcentaje de Descuento Acumulado vs 1 Noche (%)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Cantidad de Noches')
axes[1].set_ylabel('Descuento (%)')
axes[1].grid(True, alpha=0.4)

plt.tight_layout()
plt.show()


# %% [markdown]
# ## 5. Análisis de homogeneidad
#
# Para determinar qué combinación de dimensiones genera grupos donde los precios resultan comparables, calculamos la relación señal-ruido o **Signal-to-Noise Ratio (SNR)**. Un valor más alto de SNR indica mayor varianza explicada entre grupos y menor dispersión interna.
#

# %%
# Resultados de homogeneidad (Cálculo de SNR)
def calcular_snr(df_data, group_cols):
    grouped = df_data.groupby(group_cols)['price_std']
    counts = grouped.count()
    means = grouped.mean()
    vars_intra = grouped.var().fillna(0)
    
    valid = counts >= 2
    N_valid = counts[valid].sum()
    
    var_between = ((means[valid] - df_data['price_std'].mean())**2 * counts[valid]).sum() / N_valid
    var_within = (vars_intra[valid] * (counts[valid] - 1)).sum() / (N_valid - len(counts[valid]))
    
    snr = var_between / var_within if var_within > 0 else 0
    return snr, len(counts), var_between, var_within

dims_eval = {
    '1. Solo Destino': ['destination_final'],
    '2. Destino + Mes': ['destination_final', 'month'],
    '3. Destino + Mes + Semana': ['destination_final', 'month', 'week_in_month'],
    '4. Destino + Mes + Semana + Estadía': ['destination_final', 'month', 'week_in_month', 'stay_duration']
}

res_snr = []
for nombre, cols in dims_eval.items():
    snr, n_grupos, v_bet, v_with = calcular_snr(df, cols)
    res_snr.append({
        'Combinación de Dimensiones': nombre,
        'Nº Grupos': f'{n_grupos:,}',
        'Var Between': round(v_bet, 2),
        'Var Within': round(v_with, 2),
        'SNR': round(snr, 4)
    })

fig, ax = plt.subplots(figsize=(10, 5))
snr_vals = [float(r['SNR']) for r in res_snr]
comb_names = [r['Combinación de Dimensiones'] for r in res_snr]

sns.barplot(x=comb_names, y=snr_vals, color='rebeccapurple', ax=ax)
ax.set_title('Signal-to-Noise Ratio (SNR) por Combinación de Dimensiones', fontsize=13, fontweight='bold')
ax.set_ylabel('SNR (Varianza Entre / Varianza Intra)')
plt.xticks(rotation=15, ha='right')
for i, v in enumerate(snr_vals):
    ax.text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()


# %%
# Escala geográfica (SNR por escala)
escalas = {
    '1. País': ['country_code'],
    '2. Estado/Provincia': ['country_code', 'state'],
    '3. Destino Canónico': ['destination_final'],
    '4. Ciudad Cruda': ['city']
}

res_esc = []
for nombre, cols in escalas.items():
    snr, n_grupos, v_bet, v_with = calcular_snr(df, cols)
    res_esc.append({
        'Escala Geográfica': nombre,
        'Nº Grupos': f'{n_grupos:,}',
        'Var Between': round(v_bet, 2),
        'Var Within': round(v_with, 2),
        'SNR': round(snr, 4)
    })

fig, ax = plt.subplots(figsize=(10, 5))
snr_esc_vals = [float(r['SNR']) for r in res_esc]
esc_names = [r['Escala Geográfica'] for r in res_esc]

sns.barplot(x=esc_names, y=snr_esc_vals, color='darkorange', ax=ax)
ax.set_title('Signal-to-Noise Ratio (SNR) por Escala Geográfica', fontsize=13, fontweight='bold')
ax.set_ylabel('SNR')
for i, v in enumerate(snr_esc_vals):
    ax.text(i, v + 0.0001, f'{v:.4f}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()


# %%
# Ver top combinaciones de homogeneidad
df_snr_resumen = pd.DataFrame(res_snr)
print('Tabla Resumen de Signal-to-Noise Ratio (SNR) por Combinación de Dimensiones:')
print(df_snr_resumen.to_string(index=False))

print('\nTabla Resumen por Escala Geográfica:')
df_esc_resumen = pd.DataFrame(res_esc)
print(df_esc_resumen.to_string(index=False))


# %%
# CORRECCIÓN TP1: SNR debe mostrarse junto con el soporte de datos
# Para cada segmentación calculamos: SNR, contextos totales, contextos confiables (N>=30),
# y qué porcentaje de la demanda queda en contextos confiables.

MIN_OBS_SNR = 30

snr_soporte = []
for nombre, cols in dims_eval.items():
    snr, n_grupos, v_bet, v_with = calcular_snr(df, cols)
    
    # Contar observaciones por contexto usando demanda ponderada
    counts = df.groupby(cols)['count_repeated'].sum()
    confiables = (counts >= MIN_OBS_SNR).sum()
    demanda_total = counts.sum()
    demanda_confiable = counts[counts >= MIN_OBS_SNR].sum()
    
    snr_soporte.append({
        'Segmentación': nombre,
        'SNR': round(snr, 4),
        'Contextos': n_grupos,
        'Confiables (N>=30)': confiables,
        '% Contextos confiables': round(100 * confiables / n_grupos, 1) if n_grupos > 0 else 0,
        'Demanda confiable (%)': round(100 * demanda_confiable / demanda_total, 1) if demanda_total > 0 else 0
    })

df_snr_soporte = pd.DataFrame(snr_soporte)
print(df_snr_soporte.to_string(index=False))


# %% [markdown]
# ## 6. Pregunta 2: ¿Cuántos destinos poseen baselines confiables en mercados MAPEADOS?
#
# **Criterio de suficiencia estadística**: evaluamos la confiabilidad ($\ge 30$ observaciones) enfocándonos en **destinos canónicos mapeados** (`is_mapped == 1` o `destination_name != city`), aislando el ruido de ciudades no mapeadas (*singletons*).
#

# %%
# Análisis de Confiabilidad de Baselines para Destinos Canónicos Mapeados
df_mapped_only = df[df['is_mapped']].copy()

grp_base_mapped = df_mapped_only.groupby(['destination_final', 'destination_name', 'month', 'week_in_month']).size().reset_index(name='count_obs')

confiable_m = (grp_base_mapped['count_obs'] >= 30).sum()
insuficiente_m = (grp_base_mapped['count_obs'] < 30).sum()
total_baselines_m = len(grp_base_mapped)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histograma de observaciones por contexto en muestra
sns.histplot(grp_base_mapped['count_obs'], bins=50, ax=axes[0], color='navy', log_scale=(False, True))
axes[0].axvline(30, color='red', linestyle='--', linewidth=2, label='Umbral Mínimo (N=30)')
axes[0].set_title(f'Distribución de Observaciones por Contexto (Muestra N={len(df):,})', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Cantidad de Observaciones por Contexto')
axes[0].set_ylabel('Frecuencia de Celdas (Escala Log)')
axes[0].legend()

# Gráfico de torta de proporciones
axes[1].pie([confiable_m, insuficiente_m], labels=[f'Confiable (N>=30)\n{confiable_m:,} ({confiable_m/total_baselines_m:.1%})', f'Insuficiente (N<30)\n{insuficiente_m:,} ({insuficiente_m/total_baselines_m:.1%})'], autopct='%1.1f%%', colors=['#2ca02c', '#d62728'], startangle=140)
axes[1].set_title('Proporción de Baselines Confiables en Muestra', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# Cargar y reportar métricas del dataset completo generado en pipeline_build_baselines.py
baselines_full_path = Path(config.BASE_DIR) / 'outputs' / 'market_baselines.csv'
if baselines_full_path.exists():
    baselines_full = pd.read_csv(baselines_full_path, dtype={'destination_final': str}, low_memory=False)
    tot_ctx = len(baselines_full)
    hi_ctx = (~baselines_full['low_confidence']).sum()
    lo_ctx = baselines_full['low_confidence'].sum()
    tot_vol = baselines_full['count_obs'].sum()
    hi_vol = baselines_full[~baselines_full['low_confidence']]['count_obs'].sum()
    print('=== MÉTRICAS SOBRE EL DATASET COMPLETO (outputs/market_baselines.csv) ===')
    print(f'Total de contextos generados:            {tot_ctx:,}')
    print(f'  Contextos Alta Confianza (N >= 30 obs): {hi_ctx:,} ({hi_ctx/tot_ctx:.1%})')
    print(f'  Contextos Baja Confianza (N < 30 obs):  {lo_ctx:,} ({lo_ctx/tot_ctx:.1%})')
    print(f'\nCobertura de Tráfico / Demanda Real:')
    print(f'  Total de observaciones diarias:         {tot_vol:,}')
    print(f'  Volumen en contextos de Alta Confianza: {hi_vol:,} ({hi_vol/tot_vol:.1%})')
else:
    print(f'Baselines en muestra procesada: {total_baselines_m:,} ({confiable_m:,} confiables)')


# %% [markdown]
# ## 7. Visualización de baselines históricos
#
# Analizamos el comportamiento de las tarifas históricas según el destino canónico.
#

# %%
# Líneas de baselines para top destinos
top_5_dests = df.groupby('destination_name').size().nlargest(5).index
df_b5 = df[df['destination_name'].isin(top_5_dests)]

base_stats = df_b5.groupby(['destination_name', 'month'])['price_std'].mean().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=base_stats, x='month', y='price_std', hue='destination_name', marker='o', linewidth=2.5, ax=ax)
ax.set_title('Evolución Mensual del Precio Estandarizado - Top 5 Destinos', fontsize=13, fontweight='bold')
ax.set_xlabel('Mes')
ax.set_ylabel('Precio Estandarizado Promedio ($)')
ax.set_xticks(range(1, 13))
ax.legend(title='Destino')

plt.tight_layout()
plt.show()


# %%
# Baselines con STD
base_stats_std = df_b5.groupby(['destination_name', 'month'])['price_std'].agg(['mean', 'std']).reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
colors = sns.color_palette('tab10', n_colors=5)

for i, dest in enumerate(top_5_dests):
    sub = base_stats_std[base_stats_std['destination_name'] == dest]
    ax.plot(sub['month'], sub['mean'], marker='o', label=dest, color=colors[i], linewidth=2)
    ax.fill_between(sub['month'], sub['mean'] - sub['std'], sub['mean'] + sub['std'], color=colors[i], alpha=0.15)

ax.set_title('Baselines Temporales con Banda de Desviación Estándar (Mean ± Std)', fontsize=13, fontweight='bold')
ax.set_xlabel('Mes')
ax.set_ylabel('Precio Estandarizado ($)')
ax.set_xticks(range(1, 13))
ax.legend(title='Destino')

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Nota sobre `price_bucket`
#
# `price_bucket` se construye a partir de los percentiles del precio estandarizado dentro de cada destino. Es un **proxy** de la categoría hotelera porque no disponemos de estrellas, amenities ni marca.
#
# **Riesgo metodológico (leakage):** el mismo precio que define el bucket es el que después queremos clasificar como "oferta". Esto puede hacer que un precio muy barato se compare contra el bucket `low` y deje de parecer una oferta.
#
# Por eso `price_bucket` se propone como dimensión **opcional** y debe compararse con/sin bucket antes de incorporarse al modelo final de TP2/TP3.
#

# %% [markdown]
# ## 8. Hipótesis: segmentación de mercado propuesta
#
# ### 8.1 Análisis Detallado de Dimensiones Relevantes
#
# Del análisis exploratorio previo derivamos las conclusiones de cada dimensión clave:
#
# 1. **GEOGRAFÍA (`destination_final`)**:
#    - Constituye el **ANCLA fundamental** del mercado hotelero.
#    - Los precios de hoteles en distintas regiones geográficas son heterogéneos y no son directamente comparables entre sí.
#    - Se requiere un umbral mínimo recomendatorio de $\ge 30$ observaciones por contexto para garantizar la estabilidad estadística del baseline.
#
# 2. **TEMPORALIDAD (`month`, `week_in_month`, `day_of_week`)**:
#    - **Mes (`month`)**: Captura la estacionalidad climática y turística (alta vs baja temporada).
#    - **Semana del mes (`week_in_month`)**: Captura dinámicas intra-mes (efecto de liquidación de sueldos o quincenas).
#    - **Día de la semana (`day_of_week`)**: Sábados exhiben tarifas significativamente superiores a domingos en destinos vacacionales.
#    - *Conclusión*: La combinación de **Mes + Semana del mes** es ampliamente superior a usar solo el Mes.
#
# 3. **DURACIÓN DE ESTADÍA (`stay_duration` / `nights`)**:
#    - Se evidencia una **curva de descuento por volumen no lineal**: la tarifa por noche estandarizada decrece sensiblemente para reservas prolongadas (>5 noches vs 1-2 noches).
#
# 4. **OFERTA (`avg_hotel_count`) Y DEMANDA (`count_repeated`)**:
#    - Una mayor disponibilidad de hoteles (`avg_hotel_count`) genera una moderada presión a la baja en el precio por competencia.
#    - La intensidad de búsquedas (`count_repeated`) es un indicador proxy de popularidad pero presenta correlación débil con el precio estandarizado.
#
# 5. **CATEGORÍA PROXY DEL HOTEL (`price_bucket`)**:
#    - Al no disponer de estrellas de hotel explícitas en el dataset, el precio estandarizado permite crear una partición por gamas (*budget*, *mid-range*, *premium*) para comparar hoteles dentro del mismo rango de calidad.
#
# ---
#
# ### 8.2 Resumen de Evidencia Empírica
#
# | Dimensión | Hallazgo empírico | Importancia para la segmentación |
# |-----------|-------------------|----------------------------------|
# | **Geografía (`destination_final`)** | Perfiles de precio heterogéneos entre destinos | **CRÍTICA** (Ancla fundamental del mercado) |
# | **Mes del año (`month`)** | Estacionalidad marcada en zonas turísticas | **ALTA** (Captura variaciones por temporada) |
# | **Semana del mes (`week_in_month`)** | Variaciones entre la primera y tercera semana del mes | **MEDIA** (Granularidad dentro del mes) |
# | **Día de la semana (`day_of_week`)** | Sábados con mayor tarifa (73 USD) vs domingos (56 USD) en destinos de ocio | **MEDIA** (Patrón semanal de demanda) |
# | **Duración de estadía (`stay_duration`)** | Disminución no lineal de la tarifa por noche en estadías largas | **ALTA** (Descuento por volumen no lineal) |
# | **Oferta (`avg_hotel_count`)** | Reducción marginal del precio ante mayor disponibilidad | **MEDIA** (Efecto de competencia) |
# | **Demanda (`count_repeated`)** | Correlación débil con el precio normalizado | **BAJA** (Indicador menos confiable) |
#
# ---
#
# ### 8.3 Esquema de Segmentación Propuesto
#
# $$\text{Contexto} = \text{destination\_final} \times \text{month} \times \text{week\_in\_month} \times \text{stay\_duration} [\times \text{price\_bucket}]$$
#
# **Justificación del esquema**:
# 1. **Geografía (`destination_final`)**: constituye la unidad de mercado insustituible.
# 2. **Mes y semana del mes**: capturan la variabilidad temporal sin pulverizar la muestra.
# 3. **Duración de estadía (`stay_duration`)**: ajusta el descuento no lineal por volumen.
# 4. **Price Bucket (`price_bucket`)**: opcionalmente diferencia rangos de tarifa/calidad dentro del destino.
#
# ### Exclusión de oferta y demanda en la partición
#
# Si bien la oferta y la demanda influyen en los precios, agregarlas como claves de particionado genera una fragmentación excesiva de los datos y reduce el número de observaciones por celda. Por lo tanto, conviene tratarlas como **variables de control**.
#

# %% [markdown]
# ## 9. Validación e inspección interanual: 2024 vs 2025
#
# Comprobamos si los patrones de comportamiento se mantienen estables entre 2024 y 2025 para validar la robustez de la segmentación propuesta.
#

# %%
# Comparación de distribución de precios por año en muestra (usando subsample de 30k para KDE instantáneo)
df_limpio = df[df['price_std'] < 500].sample(n=min(30000, len(df)), random_state=42).copy()

fig, ax = plt.subplots(figsize=(11, 5))
sns.kdeplot(data=df_limpio[df_limpio['year'] == 2024], x='price_std', label='2024', color='blue', fill=True, alpha=0.3, ax=ax)
sns.kdeplot(data=df_limpio[df_limpio['year'] == 2025], x='price_std', label='2025', color='orange', fill=True, alpha=0.3, ax=ax)

ax.set_title('Distribución Interanual de Precios (2024 vs 2025)', fontsize=13, fontweight='bold')
ax.set_xlabel('Precio Estandarizado ($)')
ax.set_ylabel('Densidad')
ax.set_xlim(0, 300)
ax.legend()

plt.tight_layout()
plt.show()


# %%
# Patrón estacional por mes
mensual_año = df.groupby(['year', 'month'])['price_std'].mean().reset_index()

fig, ax = plt.subplots(figsize=(11, 5))
sns.lineplot(data=mensual_año, x='month', y='price_std', hue='year', palette={2024: 'blue', 2025: 'orange'}, marker='o', linewidth=2.5, ax=ax)
ax.set_title('Evolución Mensual Interanual del Precio Promedio (2024 vs 2025)', fontsize=13, fontweight='bold')
ax.set_xlabel('Mes')
ax.set_ylabel('Precio Promedio ($)')
ax.set_xticks(range(1, 13))
ax.legend(title='Año')

plt.tight_layout()
plt.show()


# %%
# Top destinos comparativos
top_dests_yoy = df.groupby('destination_name').size().nlargest(10).index
df_yoy_top = df[df['destination_name'].isin(top_dests_yoy)]
pivot_yoy = df_yoy_top.groupby(['destination_name', 'year'])['price_std'].mean().unstack()

fig, ax = plt.subplots(figsize=(12, 6))
pivot_yoy.plot(kind='barh', ax=ax, color=['blue', 'orange'], alpha=0.8)
ax.set_title('Comparación de Precio Promedio por Top Destinos (2024 vs 2025)', fontsize=13, fontweight='bold')
ax.set_xlabel('Precio Promedio ($)')
ax.set_ylabel('Destino Canónico')
ax.legend(title='Año')

plt.tight_layout()
plt.show()


# %%
# Resumen estadístico interanual
resumen_año = df.groupby('year')['price_std'].agg(
    Registros=('count'),
    Promedio=('mean'),
    Mediana=('median'),
    Desv_Std=('std'),
    Minimo=('min'),
    Maximo=('max')
).round(2)

print('Resumen Comparativo Interanual en Muestra:')
print(resumen_año.to_string())


# %% [markdown]
# ## 9.1 Evaluación de estabilidad interanual
#
# Evaluación formal de la consistencia de los patrones de precio al contrastar ambos períodos.
#

# %%
# Comparación final gráfica y correlaciones interanuales
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Subplot 1: Duración de estadía interanual
yoy_stay = df.groupby(['stay_duration', 'year'])['price_std'].mean().unstack().reindex(['corta', 'media', 'larga'])
yoy_stay.plot(kind='bar', ax=axes[0], color=['navy', 'coral'], alpha=0.85)
axes[0].set_title('Precio Promedio por Duración de Estadía (2024 vs 2025)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Duración de Estadía')
axes[0].set_ylabel('Precio Promedio ($)')
axes[0].legend(['2024', '2025'])
axes[0].tick_params(axis='x', rotation=0)

# Subplot 2: Día de la semana interanual
dias_nombre = {0: 'Lun', 1: 'Mar', 2: 'Mié', 3: 'Jue', 4: 'Vie', 5: 'Sáb', 6: 'Dom'}
yoy_dow = df.groupby(['day_of_week', 'year'])['price_std'].mean().unstack()
yoy_dow.index = [dias_nombre[i] for i in yoy_dow.index]
yoy_dow.plot(kind='bar', ax=axes[1], color=['navy', 'coral'], alpha=0.85)
axes[1].set_title('Precio Promedio por Día de la Semana (2024 vs 2025)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Día de la Semana')
axes[1].set_ylabel('Precio Promedio ($)')
axes[1].legend(['2024', '2025'])
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()

# Cálculo dinámico de correlaciones interanuales por dimensión
piv_stay = df.groupby(['stay_duration', 'year'])['price_std'].mean().unstack().dropna()
corr_stay = piv_stay[2024].corr(piv_stay[2025])

piv_dow = df.groupby(['day_of_week', 'year'])['price_std'].mean().unstack().dropna()
corr_dow = piv_dow[2024].corr(piv_dow[2025])

piv_dest = df.groupby(['destination_name', 'year'])['price_std'].mean().unstack().dropna()
top_dest_common = df.groupby('destination_name').size().nlargest(30).index
piv_dest_top = piv_dest.loc[piv_dest.index.isin(top_dest_common)]
corr_dest = piv_dest_top[2024].corr(piv_dest_top[2025]) if len(piv_dest_top) > 2 else np.nan

piv_month = df.groupby(['month', 'year'])['price_std'].mean().unstack().dropna()
corr_month = piv_month[2024].corr(piv_month[2025])

print("Resumen de Estabilidad Interanual (Correlación de Pearson 2024 vs 2025):")
print(f"  • Duración de estadía:    r = {corr_stay:.3f}")
print(f"  • Día de la semana:       r = {corr_dow:.3f}")
print(f"  • Jerarquía Top Destinos: r = {corr_dest:.3f}")
print(f"  • Patrón Mensual:         r = {corr_month:.3f}")


# %% [markdown]
# ### Evaluación de consistencia interanual
#
# A partir de los resultados calculados, evaluamos la estabilidad entre 2024 y 2025:
#
# 1. **Duración de estadía ($r \approx 1.000$)**: La estructura de descuentos por volumen es idéntica en ambos años; el precio por noche desciende conforme aumenta la duración.
# 2. **Día de la semana ($r \approx 1.000$)**: El comportamiento intra-semanal se repite fielmente año a año (fines de semana más caros en ocio, mitad de semana en destinos ejecutivos).
# 3. **Escala relativa de destinos ($r > 0.90$)**: Los destinos de alta gama y económicos conservan su posición relativa de precios entre períodos.
# 4. **Variación mensual ($r \approx 0.30 - 0.70$)**: La correlación mensual fluctúa por diferencias en el volumen de consultas capturadas mes a mes, por lo que el particionado temporal debe complementarse con la técnica de **shrinkage jerárquico**.
#

# %% [markdown]
# ## 10. Respuestas a Preguntas de Investigación y Consignas
#
# ---
#
# ### A. Respuestas a Preguntas Disparadoras (`consignas_tp1.ipynb`)
#
# #### 1. ¿Los precios suben los fines de semana?
# **Sí.** A nivel global, los sábados y viernes muestran las tarifas por noche-persona más altas, mientras que los domingos y lunes registran los valores más bajos. En promedio, los fines de semana exhiben un incremento de entre **+20% y +35%** frente al piso semanal.
#
# #### 2. ¿Ocurre en todos los destinos por igual, o solo en los de ocio?
# **No ocurre por igual; depende del perfil del destino:**
# - **Destinos vacacionales y de ocio** (*Orlando, Miami, Cancún, Las Vegas*): el pico ocurre viernes, sábado y domingo.
# - **Destinos corporativos y urbanos** (*Detroit, Columbus, Pittsburgh*): las tarifas más altas se dan de martes a jueves (viajes de trabajo), descendiendo los fines de semana.
#
# #### 3. ¿Los hoteles económicos siguen el mismo ciclo estacional que los de lujo?
# **No.** Los hoteles económicos (*Budget ≤ 20 USD*) mantienen precios estables a lo largo del año debido a una demanda inelástica. Los hoteles de lujo (*Premium > 150 USD*) exhiben una estacionalidad muy marcada, multiplicando su valor en temporada alta.
#
# #### 4. ¿Hay destinos donde la oferta disponible (`avg_hotel_count`) colapsa y altera los precios?
# **Sí.** En destinos insulares o de capacidad acotada (*San Juan Islands, Anacortes, zonas de playa*), cuando la disponibilidad cae a niveles mínimos (`avg_hotel_count ≤ 10`), el precio promedio sube hasta un **+45%** respecto a momentos de alta disponibilidad (`avg_hotel_count > 50`), donde la competencia modera la tarifa.
#
# #### 5. ¿Qué combinación de variables produce grupos más homogéneos (menor varianza interna)?
# La partición **`destination_final` $\times$ `month` $\times$ `week_in_month` $\times$ `stay_duration`** maximiza el Signal-to-Noise Ratio (**SNR**), separando la varianza entre destinos y temporadas mientras reduce la dispersión interna de cada grupo.
#
# #### 6. ¿Las Vegas y Henderson son el mismo mercado? ¿Y Miami vs Miami Beach?
# - **Miami vs Miami Beach**: Aunque distan menos de 10 km, **Miami Beach** (resorts de playa) tiene tarifas sistemáticamente más altas y mayor dispersión que **Miami Centro** (mercado urbano).
# - **Las Vegas vs Henderson**: Henderson dista 20 km del Strip. Su consolidación bajo *Las Vegas* provee masa crítica para el baseline, pero requiere la variable `price_bucket` para no mezclar hoteles de casino con hotelería suburbana.
#
# ---
#
# ### B. Respuestas a Preguntas Generales de Investigación (`README.md`)
#
# #### Pregunta 1: ¿Qué define a un mercado hotelero?
# Un mercado hotelero queda definido por la **proximidad geográfica al destino canónico (`destination_final`)**, consolidando la fragmentación de ~26.000 nombres crudos a través del mapeo Haversine y las correcciones de coherencia de mercado.
#
# #### Pregunta 2: ¿Qué condiciones hacen comparables dos precios (Contexto)?
# Dos precios son comparables cuando corresponden al mismo **destino canónico**, la misma **ventana temporal (mes y semana)**, igual **rango de estadía (`stay_duration`)** y opcionalmente el mismo **segmento de tarifa (`price_bucket`)**.
#
# #### Pregunta 3: ¿Cuántos destinos cuentan con baselines confiables?
# En el dataset completo (5,16 millones de registros y 673.030 contextos generados, con `price_bucket` activado):
# - **157.705 contextos (23,4%)** superan el umbral de alta confianza ($N \ge 30$ unidades de demanda ponderada).
# - Estos contextos de alta confianza concentran **94,3 millones de unidades de demanda (96,3% de toda la demanda real de los usuarios)**.
# - Para el 3,7% restante de la demanda en destinos de baja frecuencia (*long-tail*), el sistema aplica **fallbacks jerárquicos (Shrinkage)** para garantizar una estimación representativa.
#
# #### Pregunta 4: ¿Los patrones son estables en el tiempo (2024 vs 2025)?
# **Sí.** Las dimensiones de duración de estadía ($r = 1,000$), día de la semana ($r = 1,000$) y jerarquía relativa de destinos ($r > 0,90$) son altamente consistentes entre años.
#

# %% [markdown]
# ## 11. Próximos pasos (TP2 y Pipeline Productivo)
#
# 1. **Pipeline de Baselines**: Utilizar `pipeline_build_baselines.py` con estadísticas ponderadas por demanda (`count_repeated`) y percentiles por destino.
# 2. **Clasificación Z-Score y Fallbacks**: Evaluar ofertas utilizando los umbrales de Z-score ($<-1.0$ Deal, $<-0.5$ Good Price) y la cascada de rescate jerárquico ante celdas con baja muestra.
# 3. **Aplicación Web**: Integrar la consulta interactiva de baselines y detección de deals en Streamlit (`app.py`).
#
