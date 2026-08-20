import json
import re
from pathlib import Path

# Paths
base_dir = Path("/home/francisco/Documents/GITHUB/mentoria")
nb_path = base_dir / "TP1" / "TP1_exploracion_mercado.ipynb"

# We will construct a clean list of notebook cells
cells = []

def add_md(source_text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_text.strip().split("\n")]
    })

def add_code(source_code):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_code.strip().split("\n")]
    })

# ----------------------------------------------------------------------
# 1. TITLE & OVERVIEW
# ----------------------------------------------------------------------
add_md("""# TP1 — Definición de Mercado y Análisis Exploratorio

**Diplomatura en Ciencia de Datos · Mentoría ID90Travel**

En este notebook consolidamos el análisis exploratorio del dataset de búsquedas hoteleras de ID90Travel para determinar qué combinación de dimensiones genera grupos de observaciones donde los precios presentan un comportamiento homogéneo.

Incluye todo el código ejecutable celda por celda (standalone), incorporando la carga de datos SQLite, la estandarización de precios por *room-night-person*, el proceso de **mapping de destinos, visualización del mapa de remapeo (antes vs después)** y el análisis exploratorio completo (geográfico, temporal, oferta/demanda, SNR de homogeneidad, confiabilidad de baselines y comparación interanual 2024 vs 2025).

---""")

# ----------------------------------------------------------------------
# 2. CONFIGURATION AND DATA LOADING
# ----------------------------------------------------------------------
add_md("""## 1. Configuración y Carga de Datos

Inicializamos las librerías necesarias, configuramos los parámetros globales y verificamos la conexión a la base de datos SQLite y a los archivos de configuración.""")

add_code("""import sys
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
from math import radians, sin, cos, sqrt, atan2
import folium

# Configuración gráfica de Seaborn/Matplotlib
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

DB_PATH = Path(config.BASE_DIR) / 'data' / 'hotel_data.db'
OUTPUT_DIR = Path(config.BASE_DIR) / 'TP1' / 'outputs'

print(f'✓ Base de datos: {DB_PATH}')
print(f'✓ Directorio de Outputs: {OUTPUT_DIR}')""")

# ----------------------------------------------------------------------
# 3. SQL QUERY WITH STANDARDIZATION & CANONICAL MAPPING
# ----------------------------------------------------------------------
add_md("""## 2. Estructura y Carga del Dataset desde SQLite con Mapping Estandarizado

Cargamos el dataset completo desde SQLite, aplicando la fórmula de estandarización de precio por habitación-noche-persona:
$$\\text{price\\_std} = \\frac{\\text{avg\\_price\\_average}}{\\text{nights} \\times \\text{number\\_of\\_rooms} \\times \\max(\\text{adults} + \\text{kids}, 1)}$$

Asimismo, vinculamos la tabla de **mapping de destinos canónicos** (`destination_with_nearest.csv`) en SQLite para consolidar las ~26.000 ciudades crudas en destinos homogéneos.""")

add_code("""conn = sqlite3.connect(str(DB_PATH))

# Cargar tabla de mapping en SQLite
mapping_df = pd.read_csv(config.DESTINATION_MAPPING_FILE)
mapping_df.to_sql('mapping', conn, if_exists='replace', index=False)

query = \"\"\"
WITH valid_data AS (
    SELECT *
    FROM raw_data
    WHERE nights > 0 
      AND number_of_rooms > 0 
      AND (number_of_adults + number_of_kids) > 0
),
standardized AS (
    SELECT *,
        avg_price_average / (nights * number_of_rooms * MAX(number_of_adults + number_of_kids, 1)) as price_std,
        CAST(strftime('%w', date_start) AS INTEGER) as day_of_week,
        CAST(strftime('%m', date_start) AS INTEGER) as month,
        CAST(strftime('%Y', date_start) AS INTEGER) as year,
        CASE 
            WHEN CAST(strftime('%d', date_start) AS INTEGER) <= 7 THEN 1
            WHEN CAST(strftime('%d', date_start) AS INTEGER) <= 15 THEN 2
            WHEN CAST(strftime('%d', date_start) AS INTEGER) <= 22 THEN 3
            ELSE 4
        END as week_in_month,
        CASE WHEN nights <= 2 THEN 'corta' WHEN nights <= 5 THEN 'media' ELSE 'larga' END as stay_duration
    FROM valid_data
),
with_dest AS (
    SELECT s.*, 
        COALESCE(m3.nearest_destination_id, m2.nearest_destination_id, m1.nearest_destination_id, s.city) as destination_final,
        COALESCE(m3.nearest_destination_name, m2.nearest_destination_name, m1.nearest_destination_name, s.city) as destination_name
    FROM standardized s
    LEFT JOIN mapping m3 ON UPPER(s.country_code) || ' - ' || UPPER(s.state) || ' - ' || s.city = m3.reference
    LEFT JOIN mapping m2 ON UPPER(s.country_code) || ' - ' || s.city = m2.reference
    LEFT JOIN mapping m1 ON UPPER(s.country_code) || ' - ' || UPPER(s.state) = m1.reference
)
SELECT * FROM with_dest WHERE destination_final IS NOT NULL
\"\"\"

df = pd.read_sql_query(query, conn)
conn.close()

print(f' Registros válidos procesados: {len(df):,}')
print(f' Destinos canónicos únicos:    {df["destination_final"].nunique():,}')
print(f' Rango de fechas:             {df["date_start"].min()} a {df["date_start"].max()}')
print('\\nEstadísticas de precio normalizado (price_std):')
print(df['price_std'].describe().round(2))""")

# ----------------------------------------------------------------------
# 4. DESTINATION MAPPING & MANUAL CORRECTIONS SECTION
# ----------------------------------------------------------------------
add_md("""## 3. Mapeo de Destinos y Validación de Correcciones Manuales

El mapping geográfico agrupa ~26.000 nombres crudos de ciudades hacia destinos canónicos por cercanía (Haversine).

Para garantizar que cada agrupación mantenga coherencia en los patrones de precio, evaluamos la calidad del mapping mediante un **Score Compuesto**:
$$Score = 0.5 \\times \\text{CV\\_score} + 0.5 \\times \\text{Corr\\_score}$$

Donde:
- $\\text{CV\\_score} = \\frac{1}{1 + \\text{CV}}$ penaliza alta dispersión interna de precios.
- $\\text{Corr\\_score} = \\frac{\\text{Corr} + 1}{2}$ evalúa la alineación estacional entre ciudades del mismo destino.

### Correcciones Manuales Aplicadas (`07e_aplicar_cambios_mapping.py`):
Se identificaron y reasignaron ciudades mal mapeadas por distancia o comportamiento disímil:
- **Azusa** (Anaheim & Buena Park) $\\rightarrow$ Redondo Beach (alineación con mercado costero urbano de Los Ángeles).
- **Bell Gardens** (Anaheim & Buena Park) $\\rightarrow$ Long Beach (mercado costero/portuario).
- **Arcadia** (Anaheim & Buena Park) $\\rightarrow$ Los Angeles (mercado metropolitano LA).
- **Bellingham** (San Juan Islands) $\\rightarrow$ Seattle (separación de mercado de islas turísticas de alto costo a continente).
- **Anacortes** (San Juan Islands) $\\rightarrow$ Astoria (reclasificación costera continental).
- **Ann Arbor** (Detroit) $\\rightarrow$ Columbus (mercado universitario/regional).
- **Arlington** (Seattle) $\\rightarrow$ Cambridge (alineación regional).""")

add_code("""# Inspección y validación del mapping de destinos
print(f'Total de registros de referencias en el mapping: {len(mapping_df):,}')
print(f'Destinos canónicos en mapping:                    {mapping_df["nearest_destination_id"].nunique():,}')

# Cobertura en el dataset actual
con_mapeo = df['destination_name'].notna().sum()
sin_mapeo = len(df) - con_mapeo

print(f'\\nCobertura del mapping en el dataset:')
print(f'  Con mapeo canónico:             {con_mapeo:,} ({con_mapeo/len(df):.1%})')
print(f'  Sin mapeo (fallback a ciudad):  {sin_mapeo:,} ({sin_mapeo/len(df):.1%})')

# Resumen de cambios de mapping documentados en 07e
cambios_aplicados = [
    {'Ciudad': 'Azusa', 'Destino Original': 'Anaheim & Buena Park', 'Destino Reasignado': 'Redondo Beach'},
    {'Ciudad': 'Bell Gardens', 'Destino Original': 'Anaheim & Buena Park', 'Destino Reasignado': 'Long Beach'},
    {'Ciudad': 'Arcadia', 'Destino Original': 'Anaheim & Buena Park', 'Destino Reasignado': 'Los Angeles'},
    {'Ciudad': 'Bellingham', 'Destino Original': 'San Juan Islands', 'Destino Reasignado': 'Seattle'},
    {'Ciudad': 'Anacortes', 'Destino Original': 'San Juan Islands', 'Destino Reasignado': 'Astoria'},
    {'Ciudad': 'Ann Arbor', 'Destino Original': 'Detroit', 'Destino Reasignado': 'Columbus'},
    {'Ciudad': 'Arlington', 'Destino Original': 'Seattle', 'Destino Reasignado': 'Cambridge'}
]

df_cambios = pd.DataFrame(cambios_aplicados)
print('\\nResumen de Correcciones Manuales de Mapping Aplicadas:')
print(df_cambios.to_string(index=False))""")

# ----------------------------------------------------------------------
# MAPA COMPARATIVO ANTES Y DESPUÉS DEL REMAPEO
# ----------------------------------------------------------------------
add_md("""### 3.1 Visualización Geográfica: Mapa de Remapeo Antes vs Después

Para analizar visualmente cómo cambió la asignación de las ciudades reasignadas, comparamos el mapping original (`destination_with_nearest_backup.csv`) con el mapping optimizado (`destination_with_nearest.csv`).

El mapa interactivo a continuación muestra las ciudades reasignadas:
- **Líneas rojas discontinuas**: Conexión con el destino anterior (Antes).
- **Líneas verdes sólidas**: Conexión con el nuevo destino reasignado (Después).""")

add_code("""# Cargar ambos mappings para comparación espacial
map_new = pd.read_csv(Path(config.BASE_DIR) / 'data' / 'destination_with_nearest.csv')
map_old = pd.read_csv(Path(config.BASE_DIR) / 'data' / 'destination_with_nearest_backup.csv')

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon_deg = (lon2 - lon1 + 180) % 360 - 180
    dlon = radians(dlon_deg)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

# Centroides por destino canónico antes y después
c_old = map_old.groupby('nearest_destination_name').agg(lat=('latitude','mean'), lon=('longitude','mean')).reset_index()
c_new = map_new.groupby('nearest_destination_name').agg(lat=('latitude','mean'), lon=('longitude','mean')).reset_index()

cambios_info = [
    ('Azusa', 'Anaheim & Buena Park', 'Redondo Beach', 'Alineación con mercado urbano costero LA'),
    ('Bell Gardens', 'Anaheim & Buena Park', 'Long Beach', 'Alineación con mercado portuario/costero'),
    ('Arcadia', 'Anaheim & Buena Park', 'Los Angeles', 'Alineación con mercado metropolitano LA'),
    ('Bellingham', 'San Juan Islands', 'Seattle', 'Separación de mercado de islas turísticas a continente'),
    ('Anacortes', 'San Juan Islands', 'Astoria', 'Reclasificación costera continental'),
    ('Ann Arbor', 'Detroit', 'Columbus', 'Alineación con mercado universitario/regional')
]

# Mapa interactivo Folium
m_remap = folium.Map(location=[39.8283, -98.5795], zoom_start=4, tiles='CartoDB positron')

fg_old = folium.FeatureGroup(name='Mapeo Anterior (Línea Roja Discontinua)', show=True)
fg_new = folium.FeatureGroup(name='Mapeo Nuevo (Línea Verde Sólida)', show=True)

res_map = []
for city, d_old, d_new, motivo in cambios_info:
    row_city = map_old[map_old['city'] == city]
    if row_city.empty:
        continue
    lat_c, lon_c = row_city['latitude'].values[0], row_city['longitude'].values[0]
    
    old_row = c_old[c_old['nearest_destination_name'] == d_old]
    new_row = c_new[c_new['nearest_destination_name'] == d_new]
    
    if not old_row.empty and not new_row.empty:
        lat_o, lon_o = old_row['lat'].values[0], old_row['lon'].values[0]
        lat_n, lon_n = new_row['lat'].values[0], new_row['lon'].values[0]
        
        dist_o = haversine(lat_c, lon_c, lat_o, lon_o)
        dist_n = haversine(lat_c, lon_c, lat_n, lon_n)
        
        res_map.append({
            'Ciudad': city,
            'Destino Anterior': d_old,
            'Destino Nuevo': d_new,
            'Dist. Anterior (km)': round(dist_o, 1),
            'Dist. Nueva (km)': round(dist_n, 1),
            'Motivo Reasignación': motivo
        })
        
        # Marcador ciudad
        folium.CircleMarker(
            location=[lat_c, lon_c],
            radius=6, color='blue', fill=True, fill_color='blue', fill_opacity=0.9,
            popup=f"<b>{city}</b><br>Antes: {d_old} ({dist_o:.1f}km)<br>Después: {d_new} ({dist_n:.1f}km)<br><i>{motivo}</i>"
        ).add_to(m_remap)
        
        # Línea anterior (roja discontinua)
        folium.PolyLine(
            locations=[[lat_c, lon_c], [lat_o, lon_o]],
            color='red', weight=2, opacity=0.7, dash_array='5, 5',
            tooltip=f"{city} → {d_old} (Antes)"
        ).add_to(fg_old)
        
        # Línea nueva (verde sólida)
        folium.PolyLine(
            locations=[[lat_c, lon_c], [lat_n, lon_n]],
            color='green', weight=3, opacity=0.9,
            tooltip=f"{city} → {d_new} (Después)"
        ).add_to(fg_new)

fg_old.add_to(m_remap)
fg_new.add_to(m_remap)
folium.LayerControl(collapsed=False).add_to(m_remap)

# Guardar mapa HTML
maps_out = OUTPUT_DIR / 'maps'
maps_out.mkdir(parents=True, exist_ok=True)
map_file = maps_out / '06_mapa_remapeo_antes_despues.html'
m_remap.save(map_file)
print(f'✓ Mapa de remapeo guardado en: {map_file}')

df_res_map = pd.DataFrame(res_map)
print('\\nTabla Comparativa de Ciudades Remapeadas:')
print(df_res_map.to_string(index=False))

m_remap""")

add_code("""# Gráfico Matplotlib comparativo de vectores de remapeo por Zonas (Costa Oeste y Midwest)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: California / Costa Oeste
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

# Subplot 2: Pacific Northwest (Bellingham / Anacortes)
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
plt.show()""")

# ----------------------------------------------------------------------
# 5. MARKET DEFINITION: GEOGRAPHIC DIMENSION
# ----------------------------------------------------------------------
add_md("""## 4. Pregunta 1: ¿Qué define a un mercado hotelero?

Analizamos la combinación de dimensiones que genera grupos con variabilidad de precio homogénea.

### 4.1 Dimensión Geográfica

Comparamos la distribución de precios entre los destinos con mayor volumen de búsquedas.""")

add_code("""# Boxplot de distribución de precios por top 20 destinos
top_20_dests = df.groupby('destination_name').size().nlargest(20).index
df_top20 = df[df['destination_name'].isin(top_20_dests)].copy()

median_order = df_top20.groupby('destination_name')['price_std'].median().sort_values(ascending=False).index
df_top20['destination_name'] = pd.Categorical(df_top20['destination_name'], categories=median_order, ordered=True)

fig, ax = plt.subplots(figsize=(14, 8))
sns.boxplot(data=df_top20, x='price_std', y='destination_name', ax=ax, showfliers=False, palette='viridis')
ax.set_title('Distribución de Precios por Destino (Top 20)', fontsize=14, fontweight='bold')
ax.set_xlabel('Precio Estandarizado ($/habitación-noche-persona)')
ax.set_ylabel('Destino Canónico')
ax.set_xlim(0, 200)

plt.tight_layout()
plt.show()""")

add_code("""# Heatmap de precio promedio por destino y mes (Top 15 destinos)
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
plt.show()""")

# ----------------------------------------------------------------------
# 6. TEMPORAL DIMENSION
# ----------------------------------------------------------------------
add_md("""### 4.2 Dimensión Temporal (Estacionalidad y Día de la Semana)

Evaluamos las variaciones de precio asociadas al día de la semana y a las semanas dentro del mes.""")

add_code("""# Precio por día de la semana
dias_nombre = {0: 'Dom', 1: 'Lun', 2: 'Mar', 3: 'Mié', 4: 'Jue', 5: 'Vie', 6: 'Sáb'}
df['day_name'] = df['day_of_week'].map(dias_nombre)
order_dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']

precio_dia = df.groupby('day_name')['price_std'].agg(['mean', 'median', 'std', 'count']).reindex(order_dias)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=precio_dia.index, y=precio_dia['mean'], palette='Blues_d', ax=ax)
ax.set_title('Precio Estandarizado Promedio por Día de la Semana', fontsize=13, fontweight='bold')
ax.set_xlabel('Día de Inicio de Estadía')
ax.set_ylabel('Precio Promedio ($)')
for i, v in enumerate(precio_dia['mean']):
    ax.text(i, v + 0.5, f'${v:.1f}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()""")

add_code("""# Patrones semanales (Semana del Mes 1 a 4)
precio_semana = df.groupby('week_in_month')['price_std'].mean().reset_index()
precio_semana['week_label'] = ['Semana 1 (1-7)', 'Semana 2 (8-15)', 'Semana 3 (16-22)', 'Semana 4 (23+)']

fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=precio_semana, x='week_label', y='price_std', marker='o', color='crimson', linewidth=2.5, markersize=8, ax=ax)
ax.set_title('Evolución de Precio por Semana del Mes', fontsize=13, fontweight='bold')
ax.set_xlabel('Semana del Mes')
ax.set_ylabel('Precio Promedio ($)')
ax.set_ylim(precio_semana['price_std'].min() * 0.95, precio_semana['price_std'].max() * 1.05)

plt.tight_layout()
plt.show()""")

# ----------------------------------------------------------------------
# 7. PRODUCT & SUPPLY/DEMAND DIMENSIONS
# ----------------------------------------------------------------------
add_md("""### 4.3 Dimensión de Producto y Oferta/Demanda

Analizamos la relación entre volumen de oferta (`avg_hotel_count`), demanda (`count_repeated`) y duración de estadía con el precio estandarizado.""")

add_code("""# Matriz de Correlación entre Variables Cuantitativas
vars_cuant = ['price_std', 'nights', 'number_of_rooms', 'number_of_adults', 'number_of_kids', 'avg_hotel_count', 'count_repeated']
corr_matrix = df[vars_cuant].corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax, linewidths=0.5)
ax.set_title('Matriz de Correlación entre Variables', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()""")

add_code("""# Descuentos por Volumen / Duración de Estadía
precio_estadia = df.groupby('stay_duration')['price_std'].agg(['mean', 'median', 'count']).reindex(['corta', 'media', 'larga'])

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=precio_estadia.index, y=precio_estadia['mean'], palette='Greens_d', ax=ax)
ax.set_title('Precio Estandarizado por Duración de Estadía', fontsize=13, fontweight='bold')
ax.set_xlabel('Categoría de Estadía (corta: <=2n, media: <=5n, larga: >5n)')
ax.set_ylabel('Precio Promedio ($/room-night-person)')
for i, v in enumerate(precio_estadia['mean']):
    ax.text(i, v + 0.5, f'${v:.1f}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()""")

# ----------------------------------------------------------------------
# 8. HOMOGENEITY ANALYSIS (SNR)
# ----------------------------------------------------------------------
add_md("""## 5. Análisis de Homogeneidad (Signal-to-Noise Ratio)

Calculamos el **Signal-to-Noise Ratio (SNR)** para evaluar la variabilidad explicada entre grupos respecto a la varianza intra-grupo:
$$SNR = \\frac{\\text{Varianza Entre Grupos}}{\\text{Varianza Intra Grupo}}$$

Un SNR más elevado indica una mayor homogeneidad interna dentro de los mercados definidos.""")

add_code("""# Cálculo de SNR para distintas combinaciones de dimensiones

# Varianza Total
var_total = df['price_std'].var()

def calcular_snr(df_data, group_cols):
    grouped = df_data.groupby(group_cols)['price_std']
    counts = grouped.count()
    means = grouped.mean()
    vars_intra = grouped.var().fillna(0)
    
    # Filtrar grupos con al menos 2 observaciones
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

df_snr_resumen = pd.DataFrame(res_snr)
print('Tabla Resumen de Signal-to-Noise Ratio (SNR) por Combinación:')
print(df_snr_resumen.to_string(index=False))""")

add_code("""# Gráfico de barras de SNR por combinación de dimensiones
fig, ax = plt.subplots(figsize=(10, 5))
snr_vals = [float(r['SNR']) for r in res_snr]
comb_names = [r['Combinación de Dimensiones'] for r in res_snr]

sns.barplot(x=comb_names, y=snr_vals, palette='Purples_d', ax=ax)
ax.set_title('Signal-to-Noise Ratio (SNR) por Combinación de Dimensiones', fontsize=13, fontweight='bold')
ax.set_ylabel('SNR (Varianza Entre / Varianza Intra)')
plt.xticks(rotation=15, ha='right')
for i, v in enumerate(snr_vals):
    ax.text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()""")

# ----------------------------------------------------------------------
# 9. BASELINE RELIABILITY (QUESTION 2)
# ----------------------------------------------------------------------
add_md("""## 6. Confiabilidad del Baseline (Pregunta 2)

Evaluamos la suficiencia muestral de los baselines construidos por `destination_final × month × week_in_month`.

Establecemos que un baseline es confiable cuando cuenta con al menos **30 observaciones**.""")

add_code("""# Evaluación de observaciones por combinación (Destino x Mes x Semana)
grp_base = df.groupby(['destination_final', 'month', 'week_in_month']).size().reset_index(name='count_obs')

confiable = (grp_base['count_obs'] >= 30).sum()
insuficiente = (grp_base['count_obs'] < 30).sum()
total_baselines = len(grp_base)

print(f'Total de celdas de baseline (Destino x Mes x Semana): {total_baselines:,}')
print(f'  Baselines Confiables (>= 30 obs):   {confiable:,} ({confiable/total_baselines:.1%})')
print(f'  Baselines Insuficientes (< 30 obs): {insuficiente:,} ({insuficiente/total_baselines:.1%})')

# Visualización de la distribución de observaciones
fig, ax = plt.subplots(figsize=(10, 5))
grp_base['count_obs'].clip(upper=150).hist(bins=30, color='teal', edgecolor='white', ax=ax)
ax.axvline(30, color='red', linestyle='--', linewidth=2, label='Umbral Mínimo (30 obs)')
ax.set_title('Distribución de Cantidad de Observaciones por Baseline', fontsize=13, fontweight='bold')
ax.set_xlabel('Observaciones por Grupo (Clipped a 150)')
ax.set_ylabel('Cantidad de Baselines')
ax.legend()

plt.tight_layout()
plt.show()""")

add_code("""# Visualización de Líneas de Baseline para Top Destinos con banda std
top_5_dests = df.groupby('destination_name').size().nlargest(5).index
df_b5 = df[df['destination_name'].isin(top_5_dests)]

base_stats = df_b5.groupby(['destination_name', 'month'])['price_std'].agg(['mean', 'std']).reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
colors = sns.color_palette('tab10', n_colors=5)

for i, dest in enumerate(top_5_dests):
    sub = base_stats[base_stats['destination_name'] == dest]
    ax.plot(sub['month'], sub['mean'], marker='o', label=dest, color=colors[i], linewidth=2)
    ax.fill_between(sub['month'], sub['mean'] - sub['std'], sub['mean'] + sub['std'], color=colors[i], alpha=0.15)

ax.set_title('Evolución Mensual del Precio Estandarizado (Mean ± Std) - Top 5 Destinos', fontsize=13, fontweight='bold')
ax.set_xlabel('Mes')
ax.set_ylabel('Precio Estandarizado ($)')
ax.set_xticks(range(1, 13))
ax.legend(title='Destino')

plt.tight_layout()
plt.show()""")

# ----------------------------------------------------------------------
# 10. YEAR-OVER-YEAR COMPARISON (2024 vs 2025)
# ----------------------------------------------------------------------
add_md("""## 7. Comparación Interanual (2024 vs 2025)

Analizamos la estabilidad de los precios y patrones entre los períodos anuales 2024 y 2025.""")

add_code("""# Comparación de distribución de precios por año
df_limpio = df[df['price_std'] < 500].copy()

fig, ax = plt.subplots(figsize=(11, 5))
sns.kdeplot(data=df_limpio[df_limpio['year'] == 2024], x='price_std', label='2024', color='blue', fill=True, alpha=0.3, ax=ax)
sns.kdeplot(data=df_limpio[df_limpio['year'] == 2025], x='price_std', label='2025', color='orange', fill=True, alpha=0.3, ax=ax)

ax.set_title('Distribución Interanual de Precios (2024 vs 2025)', fontsize=13, fontweight='bold')
ax.set_xlabel('Precio Estandarizado ($)')
ax.set_ylabel('Densidad')
ax.set_xlim(0, 300)
ax.legend()

plt.tight_layout()
plt.show()""")

add_code("""# Evolución mensual interanual
mensual_año = df.groupby(['year', 'month'])['price_std'].mean().reset_index()

fig, ax = plt.subplots(figsize=(11, 5))
sns.lineplot(data=mensual_año, x='month', y='price_std', hue='year', palette={2024: 'blue', 2025: 'orange'}, marker='o', linewidth=2.5, ax=ax)
ax.set_title('Evolución Mensual Interanual del Precio Promedio (2024 vs 2025)', fontsize=13, fontweight='bold')
ax.set_xlabel('Mes')
ax.set_ylabel('Precio Promedio ($)')
ax.set_xticks(range(1, 13))
ax.legend(title='Año')

plt.tight_layout()
plt.show()""")

add_code("""# Resumen estadístico interanual
resumen_año = df.groupby('year')['price_std'].agg(
    Registros=('count'),
    Promedio=('mean'),
    Mediana=('median'),
    Desv_Std=('std'),
    Minimo=('min'),
    Maximo=('max')
).round(2)

print('Resumen Comparativo Interanual:')
print(resumen_año.to_string())""")

# ----------------------------------------------------------------------
# SAVE NOTEBOOK
# ----------------------------------------------------------------------
nb_json = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb_json, f, indent=1, ensure_ascii=False)

print(f"✓ Notebook {nb_path} generado exitosamente con {len(cells)} celdas.")
