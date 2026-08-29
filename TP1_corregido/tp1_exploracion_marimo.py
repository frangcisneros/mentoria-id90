import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NOTAS DE CORRECCIÓN TP1 → TP1_CORREGIDO

    Este notebook es la versión corregida de la entrega original. Los cambios principales aplicados son:

    1. **Expansión temporal corregida**: ahora se generan exactamente `nights` noches pagadas (`date_start + 0` a `date_start + nights - 1`), sin incluir `date_end` (checkout).
    2. **Días de semana corregidos**: el diccionario de nombres ahora respeta la convención de pandas (`0 = Lunes`, `6 = Domingo`).
    3. **Unidad de análisis explicitada**: se distinguen `demand_weight` (suma de `count_repeated`), `n_records` (filas/noches desagregadas) y `count_obs` (alias de demanda ponderada).
    4. **Mapping con trazabilidad**: se agrega `match_level` para saber qué nivel de matching usó cada geografía.
    5. **SNR con soporte de datos**: se reportan contextos generados y % de demanda en contextos confiables.

    TODO: re-ejecutar las celdas de conclusión sobre el dataset completo (2024+2025) usando `load_all_historicals`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TP1 — Definición de mercado y análisis exploratorio

    **Diplomatura en Ciencia de Datos · Mentoría ID90Travel**

    En este notebook consolidamos el análisis exploratorio del dataset de búsquedas hoteleras de ID90Travel para determinar qué combinación de dimensiones genera grupos de observaciones donde los precios presentan un comportamiento homogéneo.

    Incluye todo el código ejecutable celda por celda (standalone) sobre una **muestra representativa optimizada para consumo de RAM**, incorporando:
    1. **Auditoría de calidad y nulos**: exclusión de registros inválidos (sin ciudad/país) y consolidación de variantes por `country_code`.
    2. **Estandarización y expansiones temporales**: cálculo de precio normalizado (*room-night-person*) y features de calendario.
    3. **Mapping de destinos, Mapa Global y Mapa de Remapeo**: factor de consolidación (~11.9x), análisis de *singletons*, **mapa interactivo de destinos canónicos (`05_mapa_destinos_canonicos.html`)**, **mapa interactivo de remapeo antes vs después (`06_mapa_remapeo_antes_despues.html`)** y vectorización regional.
    4. **Análisis exploratorio de dimensiones**: geografía, patrones temporales mes x semana x día, descuentos por volumen de noches 1 a 15 y categoría proxy del hotel (`price_bucket`).
    5. **Análisis de homogeneidad (SNR)**, **suficiencia de baselines en destinos mapeados ($\ge 30$ obs en los 1.769 destinos canónicos)**, evaluación interanual 2024 vs 2025 y **respuestas a todas las preguntas de investigación**.

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. Decisiones metodológicas del análisis

    Antes de ejecutar el notebook, quedan documentadas las decisiones metodológicas que atraviesan todo el trabajo (basado en las indicaciones comunes para TP2):

    1. **Unidad de análisis**: fila original → noche pagada (`date_start` + 0 a `nights-1`) → contexto agregado.
    2. **Fuente de datos**: ambos años históricos (2024 + 2025) o, si no están disponibles, muestra representativa de 300.000 filas con semilla fija.
    3. **Muestreo**: cuando se use muestra, semilla `random_state=42` y estratificación implícita por archivo/año.
    4. **Fecha del contexto**: noche pagada; `date_end` se interpreta como checkout y no se incluye.
    5. **Precio comparable**: `avg_price_average / (nights × number_of_rooms × (adults + kids))`.
    6. **Filtros de calidad**: noches > 0 y ≤ 30, habitaciones > 0, ocupación > 0, precio > 0 y ≤ 50.000 USD, precio estandarizado > 0.
    7. **Mercado geográfico**: destino canónico via `destination_with_nearest.csv` con pretratamiento del docente; sin match queda marcado como `no_match`.
    8. **Desagregación temporal**: noches pagadas con `range(nights)`, no rango inclusivo.
    9. **Uso de `count_repeated`**: peso de demanda; reportamos `demand_weight` (suma), `n_records` (filas/noches) y `count_obs` (alias).
    10. **Variables de segmentación**: `destination_final`, `month`, `week_in_month`, `stay_duration`; `price_bucket` opcional.
    11. **Criterio de confianza**: al menos 30 unidades de demanda ponderada por contexto.
    12. **Fallbacks**: solo dentro del mismo destino (mes, año); no asignar destinos canónicos automáticamente a no-matcheados.
    13. **Reproducibilidad**: este notebook parte de `load_all_historicals()`; el pipeline productivo está en `pipeline_build_baselines.py`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Configuración y carga de datos
    """)
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo
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
    return (
        OUTPUT_DIR,
        Path,
        af,
        atan2,
        config,
        cos,
        folium,
        np,
        pd,
        plt,
        radians,
        sin,
        sns,
        sqrt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Descripción general del dataset y Auditoría de Nulos

    El dataset contiene registros de búsquedas hoteleras de ID90Travel con la siguiente estructura:
    - **5,1M de registros totales** correspondiente al período 2024-2025 (procesados mediante una muestra aleatoria representativa de 300.000 observaciones para optimizar memoria RAM)
    - **~26.000 ciudades** distintas en estado crudo
    - **Variables clave**: precio, oferta (cantidad de hoteles disponibles) y demanda (frecuencia de búsqueda)

    ### Auditoría de Calidad y Filtrado de Nulos:
    - Exclusión de registros sin ciudad ni país simultáneamente (`city` y `country` nulos).
    - Consolidación por `country_code` para evitar duplicidad de nombres con texto con ruido (ej. *"United States"*, *"USA"*, *"US"*).
    """)
    return


@app.cell
def _(Path, af, config, np, pd):
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
    n_validos = len(df_valid)
    df_std = af.standardize_prices(df_valid)
    n_estandarizados = len(df_std)
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
    return df, df_raw, mapping_df, n_estandarizados, n_validos


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.1 Estandarización de precios (`price_std`)

    El precio crudo (`avg_price_average`) **no es comparable** entre búsquedas: mezcla el tamaño del grupo, la cantidad de habitaciones y la duración de la estadía. Una búsqueda familiar de 7 noches para 4 personas devuelve precios mucho más altos que una habitación individual para 1 noche, sin que eso implique hoteles más caros.

    Para comparar precios dentro de un mismo mercado usamos el precio por **habitación-noche-persona**:

    $$\text{price\_std} = \frac{\text{avg\_price\_average}}{\text{nights} \times \text{number\_of\_rooms} \times (\text{number\_of\_adults} + \text{number\_of\_kids})}$$

    La estandarización (`af.standardize_prices`) además **depura el dataset**:
    - Elimina registros con denominador inválido (noches, habitaciones u ocupantes $\le 0$).
    - Elimina precios estandarizados $\le 0$, no interpretables como precio real.
    """)
    return


@app.cell
def _(df, mo, n_estandarizados, n_validos):
    # Verificación aritmética de la fórmula sobre el dataset completo
    _denominador = df['nights'] * df['number_of_rooms'] * (df['number_of_adults'] + df['number_of_kids'])
    _error_max = float((df['price_std'] * _denominador - df['avg_price_average']).abs().max())
    print(f'✓ Verificación de la fórmula: max |price_std × denominador − avg_price_average| = {_error_max:.6f}')
    print(f'✓ Registros enviados a estandarización: {n_validos:,}')
    print(f'✓ Registros tras estandarización:       {n_estandarizados:,} (eliminados: {n_validos - n_estandarizados:,})')

    # Ejemplos concretos: precio crudo -> denominador -> precio estandarizado
    _ej = df.sample(8, random_state=42)[
        ['city', 'nights', 'number_of_rooms', 'number_of_adults', 'number_of_kids', 'avg_price_average', 'price_std']
    ].copy()
    _ej['personas'] = _ej['number_of_adults'] + _ej['number_of_kids']
    _ej['denominador'] = _ej['nights'] * _ej['number_of_rooms'] * _ej['personas']
    _ej['reconstruccion'] = (_ej['price_std'] * _ej['denominador']).round(2)
    _ej = _ej[['city', 'nights', 'number_of_rooms', 'personas', 'avg_price_average', 'denominador', 'price_std', 'reconstruccion']].round(2)
    tabla_estandarizacion = mo.ui.table(
        _ej,
        page_size=8,
        label='Ejemplos: precio crudo → denominador → price_std',
    )
    return (tabla_estandarizacion,)


@app.cell
def _(df, np, pd, plt):
    # Distribución del precio crudo vs estandarizado (escala log por el rango extremo)
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
    _bins_crudo = np.logspace(
        np.log10(max(df['avg_price_average'].min(), 0.01)),
        np.log10(df['avg_price_average'].quantile(0.999)),
        60,
    )
    _bins_std = np.logspace(
        np.log10(max(df['price_std'].min(), 0.01)),
        np.log10(df['price_std'].quantile(0.999)),
        60,
    )
    _axes[0].hist(df['avg_price_average'], bins=_bins_crudo, color='slategray', edgecolor='white')
    _axes[0].set_xscale('log')
    _axes[0].set_title('Precio crudo (avg_price_average)', fontsize=12, fontweight='bold')
    _axes[0].set_xlabel('USD totales de la búsqueda')
    _axes[0].set_ylabel('Frecuencia')
    _axes[1].hist(df['price_std'], bins=_bins_std, color='seagreen', edgecolor='white')
    _axes[1].set_xscale('log')
    _axes[1].set_title('Precio estandarizado (price_std)', fontsize=12, fontweight='bold')
    _axes[1].set_xlabel('USD por habitación-noche-persona')
    plt.tight_layout()
    plt.show()

    _resumen = pd.DataFrame({
        'crudo': df['avg_price_average'].describe(percentiles=[0.5, 0.9, 0.99]),
        'estandarizado': df['price_std'].describe(percentiles=[0.5, 0.9, 0.99]),
    }).round(2)
    print('Comparación de escala:')
    print(_resumen.to_string())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Mapeo de Destinos, Mapa Global y Validación de Correcciones Manuales

    El mapping geográfico agrupa ~26.000 nombres crudos de ciudades hacia destinos canónicos por cercanía (Haversine).

    Para garantizar que cada agrupación mantenga coherencia en los patrones de precio, evaluamos la calidad del mapping mediante un **Score Compuesto**:
    $$Score = 0.5 \times \text{CV\_score} + 0.5 \times \text{Corr\_score}$$

    Donde:
    - $\text{CV\_score} = \frac{1}{1 + \text{CV}}$ penaliza alta dispersión interna de precios.
    - $\text{Corr\_score} = \frac{\text{Corr} + 1}{2}$ evalúa la alineación estacional entre ciudades del mismo destino.

    ### Casos de Frontera Geográfica planteados en el README:
    - **Miami vs Miami Beach**: Aunque distan menos de 10 km, Miami Beach (mercado vacacional/resorts de playa) exhibe precios significativamente más altos que el centro de Miami (mercado urbano/corporativo).
    - **Las Vegas vs Henderson**: Henderson dista 20 km del Strip de Las Vegas. Su agrupación bajo el destino canónico *Las Vegas* consolida la masa crítica necesaria de observaciones, pero aumenta ligeramente la dispersión interna entre hoteles de casino/strip vs suburbanos, lo que justifica incorporar la variable `price_bucket` (categoría proxy).

    ### Correcciones Manuales Aplicadas (`07e_aplicar_cambios_mapping.py`):
    - **Azusa** (Anaheim & Buena Park) $\rightarrow$ Redondo Beach (alineación con mercado urbano costero de Los Ángeles).
    - **Bell Gardens** (Anaheim & Buena Park) $\rightarrow$ Long Beach (mercado costero/portuario).
    - **Arcadia** (Anaheim & Buena Park) $\rightarrow$ Los Angeles (mercado metropolitano LA).
    - **Bellingham** (San Juan Islands) $\rightarrow$ Seattle (separación de mercado de islas turísticas de alto costo a continente).
    - **Anacortes** (San Juan Islands) $\rightarrow$ Astoria (reclasificación costera continental).
    - **Ann Arbor** (Detroit) $\rightarrow$ Columbus (mercado universitario/regional).
    - **Arlington** (Seattle) $\rightarrow$ Cambridge (alineación regional).
    """)
    return


@app.cell
def _(mapping_df):
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
    print(f'  Destinos con MÚLTIPLES ciudades consolidadas: {multiples:,} ({multiples / len(ciudades_por_dest):.1%})')
    print(f'  Destinos Singletons (1 sola ciudad):         {singletons:,} ({singletons / len(ciudades_por_dest):.1%})')
    print('\nTop 10 Destinos por Cantidad de Ciudades Consolidadas:')
    top_10_map = mapping_df.groupby(['nearest_destination_id', 'nearest_destination_name']).size().nlargest(10)
    for _i, ((d_id, d_name), count) in enumerate(top_10_map.items(), 1):
        print(f'  {_i:2d}. {d_name:40s} → {count:3d} ciudades')
    return


@app.cell
def _(Path, config, df, pd):
    # Cobertura y cambios detectados dinámicamente en el archivo de mapping
    _map_new = pd.read_csv(Path(config.BASE_DIR) / 'data' / 'destination_with_nearest.csv', dtype={'nearest_destination_id': str}, low_memory=False)
    map_old = pd.read_csv(Path(config.BASE_DIR) / 'data' / 'destination_with_nearest_backup.csv', dtype={'nearest_destination_id': str}, low_memory=False)
    _diff_mask = (_map_new['nearest_destination_id'] != map_old['nearest_destination_id']) | (_map_new['nearest_destination_name'] != map_old['nearest_destination_name'])
    # Detectar reasignaciones comparando con el backup original
    _diffs = _map_new[_diff_mask].copy()
    _diffs['Destino Anterior'] = map_old.loc[_diff_mask, 'nearest_destination_name']
    _diffs.rename(columns={'city': 'Ciudad', 'nearest_destination_name': 'Destino Reasignado'}, inplace=True)
    df_cambios = _diffs[['reference', 'Ciudad', 'Destino Anterior', 'Destino Reasignado']].drop_duplicates()
    con_mapeo = df['is_mapped'].sum()
    sin_mapeo = len(df) - con_mapeo
    print(f'Cobertura del mapping en la muestra actual:')
    print(f'  Con mapeo canónico:             {con_mapeo:,} ({con_mapeo / len(df):.1%})')
    print(f'  Sin mapeo (fallback a ciudad):  {sin_mapeo:,} ({sin_mapeo / len(df):.1%})')
    print(f'\nReasignaciones Manuales Detectadas en destination_with_nearest.csv ({len(df_cambios)} registros):')
    print(df_cambios.to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.1 Visualización Geográfica General: Mapa de Destinos Canónicos y Ciudades Mapeadas

    Generamos un mapa interactivo con Folium que muestra la distribución geográfica de los **Top Destinos Canónicos** y la densidad de ciudades consolidadas en cada uno.
    """)
    return


@app.cell
def _(OUTPUT_DIR, folium, mapping_df):
    # Generar mapa general de destinos canónicos consolidados
    m_canon = folium.Map(location=[39.8283, -98.5795], zoom_start=4, tiles='CartoDB positron')
    dest_agg = mapping_df.groupby(['nearest_destination_id', 'nearest_destination_name']).agg(lat=('latitude', 'mean'), lon=('longitude', 'mean'), cant_ciudades=('city', 'count')).reset_index()
    top_dests_map = dest_agg.nlargest(100, 'cant_ciudades')
    for _, _r in top_dests_map.iterrows():
        folium.CircleMarker(location=[_r['lat'], _r['lon']], radius=min(15, max(4, int(_r['cant_ciudades'] / 10))), color='darkblue', fill=True, fill_color='blue', fill_opacity=0.6, popup=f"<b>Destino Canónico: {_r['nearest_destination_name']}</b><br>ID: {_r['nearest_destination_id']}<br>Ciudades Consolidadas: {_r['cant_ciudades']}").add_to(m_canon)
    maps_out = OUTPUT_DIR / 'maps'
    maps_out.mkdir(parents=True, exist_ok=True)
    canon_map_file = maps_out / '05_mapa_destinos_canonicos.html'
    m_canon.save(canon_map_file)
    print(f'✓ Mapa de destinos canónicos guardado en: {canon_map_file}')
    m_canon
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Visualización Geográfica de Remapeos: Mapa de Remapeo Antes vs Después

    Para analizar visualmente cómo cambió la asignación de las ciudades reasignadas, comparamos el mapping original (`destination_with_nearest_backup.csv`) con el mapping optimizado (`destination_with_nearest.csv`).

    El mapa interactivo a continuación muestra las ciudades reasignadas:
    - **Líneas rojas discontinuas**: Conexión con el destino anterior (Antes).
    - **Líneas verdes sólidas**: Conexión con el nuevo destino reasignado (Después).
    """)
    return


@app.cell
def _(
    OUTPUT_DIR,
    Path,
    atan2,
    config,
    cos,
    folium,
    np,
    pd,
    radians,
    sin,
    sqrt,
):
    _map_new = pd.read_csv(Path(config.BASE_DIR) / 'data' / 'destination_with_nearest.csv', dtype={'nearest_destination_id': str}, low_memory=False)
    map_old_1 = pd.read_csv(Path(config.BASE_DIR) / 'data' / 'destination_with_nearest_backup.csv', dtype={'nearest_destination_id': str}, low_memory=False)
    coords = _map_new.groupby('nearest_destination_name')[['latitude', 'longitude']].mean().to_dict(orient='index')
    _diff_mask = (_map_new['nearest_destination_id'] != map_old_1['nearest_destination_id']) | (_map_new['nearest_destination_name'] != map_old_1['nearest_destination_name'])
    _diffs = _map_new[_diff_mask].copy()
    _diffs['dest_ant'] = map_old_1.loc[_diff_mask, 'nearest_destination_name']
    _diffs['lat_ant'] = _diffs['dest_ant'].map(lambda d: coords[d]['latitude'] if d in coords else None)
    _diffs['lon_ant'] = _diffs['dest_ant'].map(lambda d: coords[d]['longitude'] if d in coords else None)
    _diffs['lat_nueva'] = _diffs['nearest_destination_name'].map(lambda d: coords[d]['latitude'] if d in coords else None)
    _diffs['lon_nueva'] = _diffs['nearest_destination_name'].map(lambda d: coords[d]['longitude'] if d in coords else None)

    def calc_haversine(lat1, lon1, lat2, lon2):
        if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
            return np.nan
        R = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c
    _diffs['dist_ant_km'] = [calc_haversine(_r['latitude'], _r['longitude'], _r['lat_ant'], _r['lon_ant']) for _, _r in _diffs.iterrows()]
    _diffs['dist_nueva_km'] = [calc_haversine(_r['latitude'], _r['longitude'], _r['lat_nueva'], _r['lon_nueva']) for _, _r in _diffs.iterrows()]
    m_remapeo = folium.Map(location=[38.0, -97.0], zoom_start=4, tiles='CartoDB positron')
    for _, _r in _diffs.iterrows():
        if pd.notna(_r['lat_ant']) and pd.notna(_r['lon_ant']):
            folium.PolyLine(locations=[[_r['latitude'], _r['longitude']], [_r['lat_ant'], _r['lon_ant']]], color='red', weight=2.5, opacity=0.8, dash_array='5, 5', popup=f"Anterior: {_r['city']} -> {_r['dest_ant']} ({_r['dist_ant_km']:.1f} km)").add_to(m_remapeo)
        if pd.notna(_r['lat_nueva']) and pd.notna(_r['lon_nueva']):
            folium.PolyLine(locations=[[_r['latitude'], _r['longitude']], [_r['lat_nueva'], _r['lon_nueva']]], color='green', weight=3, opacity=0.9, popup=f"Nuevo: {_r['city']} -> {_r['nearest_destination_name']} ({_r['dist_nueva_km']:.1f} km)").add_to(m_remapeo)
        folium.CircleMarker(location=[_r['latitude'], _r['longitude']], radius=5, color='blue', fill=True, fill_color='blue', fill_opacity=0.9, popup=f"Ciudad: {_r['city']}").add_to(m_remapeo)
    output_map_path = OUTPUT_DIR / 'maps' / '06_mapa_remapeo_antes_despues.html'
    output_map_path.parent.mkdir(parents=True, exist_ok=True)
    m_remapeo.save(str(output_map_path))
    print(f'✓ Mapa de remapeo guardado en: {output_map_path}')
    tabla_dist = _diffs[['city', 'dest_ant', 'nearest_destination_name', 'dist_ant_km', 'dist_nueva_km']].drop_duplicates().copy()
    tabla_dist.columns = ['Ciudad', 'Destino Anterior', 'Destino Nuevo', 'Dist. Anterior (km)', 'Dist. Nueva (km)']
    df_res_map = tabla_dist.copy()
    print(f'\nTabla Comparativa de Ciudades Remapeadas:')
    print(tabla_dist.round(1).to_string(index=False))
    return df_res_map, map_old_1


@app.cell
def _(df_res_map, map_old_1, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(15, 6))
    ax1 = _axes[0]
    ca_cities = ['Azusa', 'Bell Gardens', 'Arcadia']
    for city in ca_cities:
        _r = df_res_map[df_res_map['Ciudad'] == city]
        if not _r.empty:
            c_info = map_old_1[map_old_1['city'] == city].iloc[0]
            ax1.plot(c_info['longitude'], c_info['latitude'], 'bo', markersize=8, label=city)
    ax1.set_title('Reasignaciones en Zona Sur de California', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Longitud')
    ax1.set_ylabel('Latitud')
    ax2 = _axes[1]
    nw_cities = ['Bellingham', 'Anacortes']
    for city in nw_cities:
        _r = df_res_map[df_res_map['Ciudad'] == city]
        if not _r.empty:
            c_info = map_old_1[map_old_1['city'] == city].iloc[0]
            ax2.plot(c_info['longitude'], c_info['latitude'], 'ro', markersize=8, label=city)
    ax2.set_title('Reasignaciones en Pacific Northwest (Washington)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Longitud')
    ax2.set_ylabel('Latitud')
    for _ax in _axes:
        _ax.legend()
        _ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Pregunta 1: ¿Qué define a un mercado hotelero?

    **Objetivo**: identificar la combinación de dimensiones que genera grupos homogéneos dentro de los cuales la comparación de precios resulta consistente.

    ### 4.1 Dimensión geográfica y Selección de Mercados Contrastantes

    Analizamos la distribución de precios entre destinos clave con perfiles de mercado contrapuestos (*Orlando*, *Miami*, *New York*, *Las Vegas*, *Cancún*, *Paris*).
    """)
    return


@app.cell
def _(df, plt, sns):
    # Selección de 6 Mercados Contrastantes para Análisis Comparativo Detallado
    destinos_6 = ['Orlando', 'Miami', 'Las Vegas', 'Detroit', 'Columbus', 'Cancun']
    df_6 = df[df['destination_name'].isin(destinos_6)].copy()
    _fig, _ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_6, y='destination_name', x='price_std', showfliers=False, ax=_ax, hue='destination_name', legend=False)
    _ax.set_title('Distribución de Precios Estandarizados en 6 Mercados Contrastantes', fontsize=13, fontweight='bold')
    _ax.set_xlabel('Precio Estandarizado ($ / habitación-noche-persona)')
    _ax.set_ylabel('Destino Canónico')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, pd, plt, sns):
    # Boxplot de precios por destino (top 20 general)
    top_20_dests = df.groupby('destination_name').size().nlargest(20).index
    df_top20 = df[df['destination_name'].isin(top_20_dests)].copy()
    median_order = df_top20.groupby('destination_name')['price_std'].median().sort_values(ascending=False).index
    df_top20['destination_name'] = pd.Categorical(df_top20['destination_name'], categories=median_order, ordered=True)
    _fig, _ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=df_top20, x='price_std', y='destination_name', ax=_ax, showfliers=False, color='skyblue')
    _ax.set_title('Distribución de Precios por Destino (Top 20 General)', fontsize=14, fontweight='bold')
    _ax.set_xlabel('Precio Estandarizado ($/habitación-noche-persona)')
    _ax.set_ylabel('Destino Canónico')
    _ax.set_xlim(0, 200)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, plt, sns):
    # Heatmap destino x mes (Top 15 destinos)
    top_15_dests = df.groupby('destination_name').size().nlargest(15).index
    df_top15 = df[df['destination_name'].isin(top_15_dests)]
    pivot_mes = df_top15.pivot_table(index='destination_name', columns='month', values='price_std', aggfunc='mean')
    pivot_mes.columns = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    _fig, _ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot_mes, annot=True, fmt='.1f', cmap='YlOrRd', ax=_ax, linewidths=0.5, cbar_kws={'label': 'Precio Promedio ($)'})
    _ax.set_title('Precio Promedio por Destino y Mes', fontsize=14, fontweight='bold')
    _ax.set_xlabel('Mes del Año')
    _ax.set_ylabel('Destino')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 Dimensión temporal (Mes, Semana del Mes y Día de la Semana)

    Analizamos cómo varían las tarifas a lo largo del tiempo:
    - **Día de la semana**: evaluamos si los sábados presentan tarifas más elevadas vs domingos.
    - **Semana del mes**: comparamos las variaciones entre la primera y la tercera semana de cada mes (efecto cobranza/quincena).
    - **Matriz Temporal Completa**: interactuación entre Mes × Día de Semana y Mes × Semana del Mes.
    """)
    return


@app.cell
def _(df, plt, sns):
    # Precio por día de la semana
    _dias_nombre = {0: 'Lun', 1: 'Mar', 2: 'Mié', 3: 'Jue', 4: 'Vie', 5: 'Sáb', 6: 'Dom'}
    df['day_name'] = df['day_of_week'].map(_dias_nombre)
    order_dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    precio_dia = df.groupby('day_name')['price_std'].agg(['mean', 'median', 'std', 'count']).reindex(order_dias)
    _fig, _ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=precio_dia.index, y=precio_dia['mean'], color='steelblue', ax=_ax)
    _ax.set_title('Precio Estandarizado Promedio por Día de la Semana', fontsize=13, fontweight='bold')
    _ax.set_xlabel('Día de Inicio de Estadía')
    _ax.set_ylabel('Precio Promedio ($)')
    for _i, _v in enumerate(precio_dia['mean']):
        _ax.text(_i, _v + 0.5, f'${_v:.1f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.show()
    return (order_dias,)


@app.cell
def _(df, plt, sns):
    # Patrones semanales (Semana del mes 1 a 4)
    precio_semana = df.groupby('week_in_month')['price_std'].mean().reset_index()
    precio_semana['week_label'] = ['Semana 1 (1-7)', 'Semana 2 (8-15)', 'Semana 3 (16-22)', 'Semana 4 (23+)']
    _fig, _ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=precio_semana, x='week_label', y='price_std', marker='o', color='crimson', linewidth=2.5, markersize=8, ax=_ax)
    _ax.set_title('Evolución de Precio por Semana del Mes', fontsize=13, fontweight='bold')
    _ax.set_xlabel('Semana del Mes')
    _ax.set_ylabel('Precio Promedio ($)')
    _ax.set_ylim(precio_semana['price_std'].min() * 0.95, precio_semana['price_std'].max() * 1.05)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, order_dias, plt, sns):
    # Análisis Temporal Completo (Mes × Día de Semana y Mes × Semana del Mes)
    months_labels = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    weeks_labels = ['Sem 1', 'Sem 2', 'Sem 3', 'Sem 4']
    pivot_md = df.groupby(['month', 'day_of_week'])['price_std'].mean().unstack()
    pivot_md.columns = ['Dom', 'Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb']
    pivot_md = pivot_md[order_dias]
    pivot_mw = df.groupby(['month', 'week_in_month'])['price_std'].mean().unstack()
    pivot_mw.columns = weeks_labels
    _fig, _axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(pivot_md, annot=True, fmt='.1f', cmap='YlOrRd', ax=_axes[0], yticklabels=months_labels, linewidths=0.5)
    _axes[0].set_title('Precio Promedio ($): Mes × Día de Semana', fontsize=13, fontweight='bold')
    _axes[0].set_xlabel('Día de Semana')
    _axes[0].set_ylabel('Mes')
    sns.heatmap(pivot_mw, annot=True, fmt='.1f', cmap='YlOrRd', ax=_axes[1], yticklabels=months_labels, linewidths=0.5)
    _axes[1].set_title('Precio Promedio ($): Mes × Semana del Mes', fontsize=13, fontweight='bold')
    _axes[1].set_xlabel('Semana del Mes')
    _axes[1].set_ylabel('Mes')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.3 Dimensión de oferta, demanda y categoría proxy

    Analizamos las relaciones entre la disponibilidad de hoteles (`avg_hotel_count`), la intensidad de búsquedas (`count_repeated`) y el precio normalizado, incorporando la **categoría proxy de hotel (gama de precios)**.
    """)
    return


@app.cell
def _(df, np, plt, sns):
    # Categoría Proxy de Hotel (Basada en rangos de precio estandarizado)
    conditions = [df['price_std'] <= 20, df['price_std'] <= 40, df['price_std'] <= 60, df['price_std'] <= 100, df['price_std'] <= 150]
    choices = ['1_Budget (≤$20)', '2_Mid-Low ($20-40)', '3_Mid ($40-60)', '4_Mid-High ($60-100)', '5_High ($100-150)']
    df['category_proxy'] = np.select(conditions, choices, default='6_Premium (>$150)')
    df_cat = df.groupby('category_proxy').agg(mean_price=('price_std', 'mean'), mean_supply=('avg_hotel_count', 'mean'), mean_demand=('count_repeated', 'mean'), mean_nights=('nights', 'mean'), count_obs=('price_std', 'count')).reset_index()
    print('Análisis de Categoría Proxy de Hotel (Segmentación por Gama de Precio):')
    print(df_cat.to_string(index=False))
    _fig, _axes = plt.subplots(1, 3, figsize=(18, 5))
    cats_short = [c.split('_')[0] for c in df_cat['category_proxy']]
    _axes[0].bar(cats_short, df_cat['mean_supply'], color='coral')
    _axes[0].set_title('Oferta Promedio (Hoteles) por Categoría', fontsize=12, fontweight='bold')
    _axes[0].set_xlabel('Categoría Proxy')
    _axes[0].set_ylabel('Hoteles Disponibles')
    _axes[1].bar(cats_short, df_cat['mean_demand'], color='steelblue')
    _axes[1].set_title('Demanda Promedio (Búsquedas) por Categoría', fontsize=12, fontweight='bold')
    _axes[1].set_xlabel('Categoría Proxy')
    _axes[1].set_ylabel('count_repeated Promedio')
    _axes[2].pie(df_cat['count_obs'], labels=cats_short, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set3', n_colors=6))
    _axes[2].set_title('Distribución de Observaciones por Categoría', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, pd, plt, sns):
    # Heatmap oferta vs demanda vs precio con rank method first para evitar duplicados
    df['oferta_bin'] = pd.qcut(df['avg_hotel_count'].rank(method='first'), q=4, labels=['Baja', 'Media-Baja', 'Media-Alta', 'Alta'])
    df['demanda_bin'] = pd.qcut(df['count_repeated'].rank(method='first'), q=4, labels=['Baja', 'Media-Baja', 'Media-Alta', 'Alta'])
    pivot_od = df.pivot_table(index='oferta_bin', columns='demanda_bin', values='price_std', aggfunc='mean')
    _fig, _ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_od, annot=True, fmt='.1f', cmap='YlGnBu', ax=_ax, linewidths=0.5)
    _ax.set_title('Precio Promedio ($) por Niveles de Oferta y Demanda', fontsize=13, fontweight='bold')
    _ax.set_xlabel('Nivel de Demanda (Búsquedas)')
    _ax.set_ylabel('Nivel de Oferta (Hoteles)')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, plt, sns):
    # Correlaciones entre variables cuantitativas
    vars_cuant = ['price_std', 'nights', 'number_of_rooms', 'number_of_adults', 'number_of_kids', 'avg_hotel_count', 'count_repeated']
    corr_matrix = df[vars_cuant].corr()
    _fig, _ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=_ax, linewidths=0.5)
    _ax.set_title('Matriz de Correlación entre Variables', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.4 Dimensión de duración de estadía y Descuentos por Volumen

    Evaluamos en detalle cómo varía la tarifa por noche normalizada ante cambios en la cantidad total de noches reservadas (1 a 15+ noches).
    """)
    return


@app.cell
def _(df, plt):
    # Análisis exhaustivo de descuentos por volumen (noches 1 a 15)
    df_vol = df.groupby('nights').agg(mean_total_price=('avg_price_average', 'mean'), mean_price_per_night=('avg_price_average', lambda x: (x / df.loc[x.index, 'nights']).mean()), mean_price_std=('price_std', 'mean'), count_obs=('price_std', 'count')).reset_index()
    df_vol = df_vol[df_vol['nights'] <= 15]
    price_1n = df_vol[df_vol['nights'] == 1]['mean_price_std'].values[0]
    df_vol['discount_pct'] = (1 - df_vol['mean_price_std'] / price_1n) * 100
    print('Tabla Detallada de Descuentos por Volumen según Noches de Estadía:')
    print(df_vol[['nights', 'mean_total_price', 'mean_price_std', 'discount_pct', 'count_obs']].to_string(index=False))
    _fig, _axes = plt.subplots(1, 2, figsize=(15, 6))
    _axes[0].plot(df_vol['nights'], df_vol['mean_price_std'], 'o-', linewidth=2.5, markersize=7, color='darkgreen')
    _axes[0].set_title('Precio Estandarizado vs Noches de Estadía', fontsize=13, fontweight='bold')
    _axes[0].set_xlabel('Cantidad de Noches')
    _axes[0].set_ylabel('Precio Estandarizado Promedio ($)')
    _axes[0].grid(True, alpha=0.4)
    _axes[1].bar(df_vol['nights'], df_vol['discount_pct'], color='forestgreen', alpha=0.8)
    _axes[1].set_title('Porcentaje de Descuento Acumulado vs 1 Noche (%)', fontsize=13, fontweight='bold')
    _axes[1].set_xlabel('Cantidad de Noches')
    _axes[1].set_ylabel('Descuento (%)')
    _axes[1].grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Análisis de homogeneidad

    Para determinar qué combinación de dimensiones genera grupos donde los precios resultan comparables, calculamos la relación señal-ruido o **Signal-to-Noise Ratio (SNR)**. Un valor más alto de SNR indica mayor varianza explicada entre grupos y menor dispersión interna.
    """)
    return


@app.cell
def _(df, plt, sns):
    # Resultados de homogeneidad (Cálculo de SNR)
    def calcular_snr(df_data, group_cols):
        grouped = df_data.groupby(group_cols)['price_std']
        counts = grouped.count()
        means = grouped.mean()
        vars_intra = grouped.var().fillna(0)
        valid = counts >= 2
        N_valid = counts[valid].sum()
        var_between = ((means[valid] - df_data['price_std'].mean()) ** 2 * counts[valid]).sum() / N_valid
        var_within = (vars_intra[valid] * (counts[valid] - 1)).sum() / (N_valid - len(counts[valid]))
        _snr = var_between / var_within if var_within > 0 else 0
        return (_snr, len(counts), var_between, var_within)
    dims_eval = {'1. Solo Destino': ['destination_final'], '2. Destino + Mes': ['destination_final', 'month'], '3. Destino + Mes + Semana': ['destination_final', 'month', 'week_in_month'], '4. Destino + Mes + Semana + Estadía': ['destination_final', 'month', 'week_in_month', 'stay_duration']}
    res_snr = []
    for _nombre, _cols in dims_eval.items():
        _snr, _n_grupos, _v_bet, _v_with = calcular_snr(df, _cols)
        res_snr.append({'Combinación de Dimensiones': _nombre, 'Nº Grupos': f'{_n_grupos:,}', 'Var Between': round(_v_bet, 2), 'Var Within': round(_v_with, 2), 'SNR': round(_snr, 4)})
    _fig, _ax = plt.subplots(figsize=(10, 5))
    snr_vals = [float(_r['SNR']) for _r in res_snr]
    comb_names = [_r['Combinación de Dimensiones'] for _r in res_snr]
    sns.barplot(x=comb_names, y=snr_vals, color='rebeccapurple', ax=_ax)
    _ax.set_title('Signal-to-Noise Ratio (SNR) por Combinación de Dimensiones', fontsize=13, fontweight='bold')
    _ax.set_ylabel('SNR (Varianza Entre / Varianza Intra)')
    plt.xticks(rotation=15, ha='right')
    for _i, _v in enumerate(snr_vals):
        _ax.text(_i, _v + 0.001, f'{_v:.4f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.show()
    return calcular_snr, dims_eval, res_snr


@app.cell
def _(calcular_snr, df, plt, sns):
    # Escala geográfica (SNR por escala)
    escalas = {'1. País': ['country_code'], '2. Estado/Provincia': ['country_code', 'state'], '3. Destino Canónico': ['destination_final'], '4. Ciudad Cruda': ['city']}
    res_esc = []
    for _nombre, _cols in escalas.items():
        _snr, _n_grupos, _v_bet, _v_with = calcular_snr(df, _cols)
        res_esc.append({'Escala Geográfica': _nombre, 'Nº Grupos': f'{_n_grupos:,}', 'Var Between': round(_v_bet, 2), 'Var Within': round(_v_with, 2), 'SNR': round(_snr, 4)})
    _fig, _ax = plt.subplots(figsize=(10, 5))
    snr_esc_vals = [float(_r['SNR']) for _r in res_esc]
    esc_names = [_r['Escala Geográfica'] for _r in res_esc]
    sns.barplot(x=esc_names, y=snr_esc_vals, color='darkorange', ax=_ax)
    _ax.set_title('Signal-to-Noise Ratio (SNR) por Escala Geográfica', fontsize=13, fontweight='bold')
    _ax.set_ylabel('SNR')
    for _i, _v in enumerate(snr_esc_vals):
        _ax.text(_i, _v + 0.0001, f'{_v:.4f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.show()
    return (res_esc,)


@app.cell
def _(pd, res_esc, res_snr):
    # Ver top combinaciones de homogeneidad
    df_snr_resumen = pd.DataFrame(res_snr)
    print('Tabla Resumen de Signal-to-Noise Ratio (SNR) por Combinación de Dimensiones:')
    print(df_snr_resumen.to_string(index=False))

    print('\nTabla Resumen por Escala Geográfica:')
    df_esc_resumen = pd.DataFrame(res_esc)
    print(df_esc_resumen.to_string(index=False))
    return


@app.cell
def _(calcular_snr, df, dims_eval, pd):
    # CORRECCIÓN TP1: SNR debe mostrarse junto con el soporte de datos
    # Para cada segmentación calculamos: SNR, contextos totales, contextos confiables (N>=30),
    # y qué porcentaje de la demanda queda en contextos confiables.
    MIN_OBS_SNR = 30
    snr_soporte = []
    for _nombre, _cols in dims_eval.items():
        _snr, _n_grupos, _v_bet, _v_with = calcular_snr(df, _cols)
        counts = df.groupby(_cols)['count_repeated'].sum()
        confiables = (counts >= MIN_OBS_SNR).sum()
        demanda_total = counts.sum()
        demanda_confiable = counts[counts >= MIN_OBS_SNR].sum()  # Contar observaciones por contexto usando demanda ponderada
        snr_soporte.append({'Segmentación': _nombre, 'SNR': round(_snr, 4), 'Contextos': _n_grupos, 'Confiables (N>=30)': confiables, '% Contextos confiables': round(100 * confiables / _n_grupos, 1) if _n_grupos > 0 else 0, 'Demanda confiable (%)': round(100 * demanda_confiable / demanda_total, 1) if demanda_total > 0 else 0})
    df_snr_soporte = pd.DataFrame(snr_soporte)
    print(df_snr_soporte.to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Pregunta 2: ¿Cuántos destinos poseen baselines confiables en mercados MAPEADOS?

    **Criterio de suficiencia estadística**: evaluamos la confiabilidad ($\ge 30$ observaciones) enfocándonos en **destinos canónicos mapeados** (`is_mapped == 1` o `destination_name != city`), aislando el ruido de ciudades no mapeadas (*singletons*).
    """)
    return


@app.cell
def _(Path, config, df, pd, plt, sns):
    # Análisis de Confiabilidad de Baselines para Destinos Canónicos Mapeados
    df_mapped_only = df[df['is_mapped']].copy()
    grp_base_mapped = df_mapped_only.groupby(['destination_final', 'destination_name', 'month', 'week_in_month']).size().reset_index(name='count_obs')
    confiable_m = (grp_base_mapped['count_obs'] >= 30).sum()
    insuficiente_m = (grp_base_mapped['count_obs'] < 30).sum()
    total_baselines_m = len(grp_base_mapped)
    _fig, _axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(grp_base_mapped['count_obs'], bins=50, ax=_axes[0], color='navy', log_scale=(False, True))
    _axes[0].axvline(30, color='red', linestyle='--', linewidth=2, label='Umbral Mínimo (N=30)')
    _axes[0].set_title(f'Distribución de Observaciones por Contexto (Muestra N={len(df):,})', fontsize=12, fontweight='bold')
    _axes[0].set_xlabel('Cantidad de Observaciones por Contexto')
    # Histograma de observaciones por contexto en muestra
    _axes[0].set_ylabel('Frecuencia de Celdas (Escala Log)')
    _axes[0].legend()
    _axes[1].pie([confiable_m, insuficiente_m], labels=[f'Confiable (N>=30)\n{confiable_m:,} ({confiable_m / total_baselines_m:.1%})', f'Insuficiente (N<30)\n{insuficiente_m:,} ({insuficiente_m / total_baselines_m:.1%})'], autopct='%1.1f%%', colors=['#2ca02c', '#d62728'], startangle=140)
    _axes[1].set_title('Proporción de Baselines Confiables en Muestra', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    baselines_full_path = Path(config.BASE_DIR) / 'outputs' / 'market_baselines.csv'
    # Gráfico de torta de proporciones
    if baselines_full_path.exists():
        baselines_full = pd.read_csv(baselines_full_path, dtype={'destination_final': str}, low_memory=False)
        tot_ctx = len(baselines_full)
        hi_ctx = (~baselines_full['low_confidence']).sum()
        lo_ctx = baselines_full['low_confidence'].sum()
    # Cargar y reportar métricas del dataset completo generado en pipeline_build_baselines.py
        tot_vol = baselines_full['count_obs'].sum()
        hi_vol = baselines_full[~baselines_full['low_confidence']]['count_obs'].sum()
        print('=== MÉTRICAS SOBRE EL DATASET COMPLETO (outputs/market_baselines.csv) ===')
        print(f'Total de contextos generados:            {tot_ctx:,}')
        print(f'  Contextos Alta Confianza (N >= 30 obs): {hi_ctx:,} ({hi_ctx / tot_ctx:.1%})')
        print(f'  Contextos Baja Confianza (N < 30 obs):  {lo_ctx:,} ({lo_ctx / tot_ctx:.1%})')
        print(f'\nCobertura de Tráfico / Demanda Real:')
        print(f'  Total de observaciones diarias:         {tot_vol:,}')
        print(f'  Volumen en contextos de Alta Confianza: {hi_vol:,} ({hi_vol / tot_vol:.1%})')
    else:
        print(f'Baselines en muestra procesada: {total_baselines_m:,} ({confiable_m:,} confiables)')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Visualización de baselines históricos

    Analizamos el comportamiento de las tarifas históricas según el destino canónico.
    """)
    return


@app.cell
def _(df, plt, sns):
    # Líneas de baselines para top destinos
    top_5_dests = df.groupby('destination_name').size().nlargest(5).index
    df_b5 = df[df['destination_name'].isin(top_5_dests)]
    base_stats = df_b5.groupby(['destination_name', 'month'])['price_std'].mean().reset_index()
    _fig, _ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=base_stats, x='month', y='price_std', hue='destination_name', marker='o', linewidth=2.5, ax=_ax)
    _ax.set_title('Evolución Mensual del Precio Estandarizado - Top 5 Destinos', fontsize=13, fontweight='bold')
    _ax.set_xlabel('Mes')
    _ax.set_ylabel('Precio Estandarizado Promedio ($)')
    _ax.set_xticks(range(1, 13))
    _ax.legend(title='Destino')
    plt.tight_layout()
    plt.show()
    return df_b5, top_5_dests


@app.cell
def _(df_b5, plt, sns, top_5_dests):
    # Baselines con STD
    base_stats_std = df_b5.groupby(['destination_name', 'month'])['price_std'].agg(['mean', 'std']).reset_index()
    _fig, _ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette('tab10', n_colors=5)
    for _i, _dest in enumerate(top_5_dests):
        _sub = base_stats_std[base_stats_std['destination_name'] == _dest]
        _ax.plot(_sub['month'], _sub['mean'], marker='o', label=_dest, color=colors[_i], linewidth=2)
        _ax.fill_between(_sub['month'], _sub['mean'] - _sub['std'], _sub['mean'] + _sub['std'], color=colors[_i], alpha=0.15)
    _ax.set_title('Baselines Temporales con Banda de Desviación Estándar (Mean ± Std)', fontsize=13, fontweight='bold')
    _ax.set_xlabel('Mes')
    _ax.set_ylabel('Precio Estandarizado ($)')
    _ax.set_xticks(range(1, 13))
    _ax.legend(title='Destino')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Nota sobre `price_bucket`

    `price_bucket` se construye a partir de los percentiles del precio estandarizado dentro de cada destino. Es un **proxy** de la categoría hotelera porque no disponemos de estrellas, amenities ni marca.

    **Riesgo metodológico (leakage):** el mismo precio que define el bucket es el que después queremos clasificar como "oferta". Esto puede hacer que un precio muy barato se compare contra el bucket `low` y deje de parecer una oferta.

    Por eso `price_bucket` se propone como dimensión **opcional** y debe compararse con/sin bucket antes de incorporarse al modelo final de TP2/TP3.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Hipótesis: segmentación de mercado propuesta

    ### 8.1 Análisis Detallado de Dimensiones Relevantes

    Del análisis exploratorio previo derivamos las conclusiones de cada dimensión clave:

    1. **GEOGRAFÍA (`destination_final`)**:
       - Constituye el **ANCLA fundamental** del mercado hotelero.
       - Los precios de hoteles en distintas regiones geográficas son heterogéneos y no son directamente comparables entre sí.
       - Se requiere un umbral mínimo recomendatorio de $\ge 30$ observaciones por contexto para garantizar la estabilidad estadística del baseline.

    2. **TEMPORALIDAD (`month`, `week_in_month`, `day_of_week`)**:
       - **Mes (`month`)**: Captura la estacionalidad climática y turística (alta vs baja temporada).
       - **Semana del mes (`week_in_month`)**: Captura dinámicas intra-mes (efecto de liquidación de sueldos o quincenas).
       - **Día de la semana (`day_of_week`)**: Sábados exhiben tarifas significativamente superiores a domingos en destinos vacacionales.
       - *Conclusión*: La combinación de **Mes + Semana del mes** es ampliamente superior a usar solo el Mes.

    3. **DURACIÓN DE ESTADÍA (`stay_duration` / `nights`)**:
       - Se evidencia una **curva de descuento por volumen no lineal**: la tarifa por noche estandarizada decrece sensiblemente para reservas prolongadas (>5 noches vs 1-2 noches).

    4. **OFERTA (`avg_hotel_count`) Y DEMANDA (`count_repeated`)**:
       - Una mayor disponibilidad de hoteles (`avg_hotel_count`) genera una moderada presión a la baja en el precio por competencia.
       - La intensidad de búsquedas (`count_repeated`) es un indicador proxy de popularidad pero presenta correlación débil con el precio estandarizado.

    5. **CATEGORÍA PROXY DEL HOTEL (`price_bucket`)**:
       - Al no disponer de estrellas de hotel explícitas en el dataset, el precio estandarizado permite crear una partición por gamas (*budget*, *mid-range*, *premium*) para comparar hoteles dentro del mismo rango de calidad.

    ---

    ### 8.2 Resumen de Evidencia Empírica

    | Dimensión | Hallazgo empírico | Importancia para la segmentación |
    |-----------|-------------------|----------------------------------|
    | **Geografía (`destination_final`)** | Perfiles de precio heterogéneos entre destinos | **CRÍTICA** (Ancla fundamental del mercado) |
    | **Mes del año (`month`)** | Estacionalidad marcada en zonas turísticas | **ALTA** (Captura variaciones por temporada) |
    | **Semana del mes (`week_in_month`)** | Variaciones entre la primera y tercera semana del mes | **MEDIA** (Granularidad dentro del mes) |
    | **Día de la semana (`day_of_week`)** | Sábados con mayor tarifa ($73) vs domingos ($56) en destinos de ocio | **MEDIA** (Patrón semanal de demanda) |
    | **Duración de estadía (`stay_duration`)** | Disminución no lineal de la tarifa por noche en estadías largas | **ALTA** (Descuento por volumen no lineal) |
    | **Oferta (`avg_hotel_count`)** | Reducción marginal del precio ante mayor disponibilidad | **MEDIA** (Efecto de competencia) |
    | **Demanda (`count_repeated`)** | Correlación débil con el precio normalizado | **BAJA** (Indicador menos confiable) |

    ---

    ### 8.3 Esquema de Segmentación Propuesto

    $$\text{Contexto} = \text{destination\_final} \times \text{month} \times \text{week\_in\_month} \times \text{stay\_duration} [\times \text{price\_bucket}]$$

    **Justificación del esquema**:
    1. **Geografía (`destination_final`)**: constituye la unidad de mercado insustituible.
    2. **Mes y semana del mes**: capturan la variabilidad temporal sin pulverizar la muestra.
    3. **Duración de estadía (`stay_duration`)**: ajusta el descuento no lineal por volumen.
    4. **Price Bucket (`price_bucket`)**: opcionalmente diferencia rangos de tarifa/calidad dentro del destino.

    ### Exclusión de oferta y demanda en la partición

    Si bien la oferta y la demanda influyen en los precios, agregarlas como claves de particionado genera una fragmentación excesiva de los datos y reduce el número de observaciones por celda. Por lo tanto, conviene tratarlas como **variables de control**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Validación e inspección interanual: 2024 vs 2025

    Comprobamos si los patrones de comportamiento se mantienen estables entre 2024 y 2025 para validar la robustez de la segmentación propuesta.
    """)
    return


@app.cell
def _(df, plt, sns):
    # Comparación de distribución de precios por año en muestra (usando subsample de 30k para KDE instantáneo)
    df_limpio = df[df['price_std'] < 500].sample(n=min(30000, len(df)), random_state=42).copy()
    _fig, _ax = plt.subplots(figsize=(11, 5))
    sns.kdeplot(data=df_limpio[df_limpio['year'] == 2024], x='price_std', label='2024', color='blue', fill=True, alpha=0.3, ax=_ax)
    sns.kdeplot(data=df_limpio[df_limpio['year'] == 2025], x='price_std', label='2025', color='orange', fill=True, alpha=0.3, ax=_ax)
    _ax.set_title('Distribución Interanual de Precios (2024 vs 2025)', fontsize=13, fontweight='bold')
    _ax.set_xlabel('Precio Estandarizado ($)')
    _ax.set_ylabel('Densidad')
    _ax.set_xlim(0, 300)
    _ax.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, plt, sns):
    # Patrón estacional por mes
    mensual_año = df.groupby(['year', 'month'])['price_std'].mean().reset_index()
    _fig, _ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=mensual_año, x='month', y='price_std', hue='year', palette={2024: 'blue', 2025: 'orange'}, marker='o', linewidth=2.5, ax=_ax)
    _ax.set_title('Evolución Mensual Interanual del Precio Promedio (2024 vs 2025)', fontsize=13, fontweight='bold')
    _ax.set_xlabel('Mes')
    _ax.set_ylabel('Precio Promedio ($)')
    _ax.set_xticks(range(1, 13))
    _ax.legend(title='Año')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, plt):
    # Top destinos comparativos
    top_dests_yoy = df.groupby('destination_name').size().nlargest(10).index
    df_yoy_top = df[df['destination_name'].isin(top_dests_yoy)]
    pivot_yoy = df_yoy_top.groupby(['destination_name', 'year'])['price_std'].mean().unstack()
    _fig, _ax = plt.subplots(figsize=(12, 6))
    pivot_yoy.plot(kind='barh', ax=_ax, color=['blue', 'orange'], alpha=0.8)
    _ax.set_title('Comparación de Precio Promedio por Top Destinos (2024 vs 2025)', fontsize=13, fontweight='bold')
    _ax.set_xlabel('Precio Promedio ($)')
    _ax.set_ylabel('Destino Canónico')
    _ax.legend(title='Año')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9.1 Evaluación de estabilidad interanual

    Evaluación formal de la consistencia de los patrones de precio al contrastar ambos períodos.
    """)
    return


@app.cell
def _(df, np, plt):
    # Comparación final gráfica y correlaciones interanuales
    _fig, _axes = plt.subplots(1, 2, figsize=(15, 5))
    yoy_stay = df.groupby(['stay_duration', 'year'])['price_std'].mean().unstack().reindex(['corta', 'media', 'larga'])
    # Subplot 1: Duración de estadía interanual
    yoy_stay.plot(kind='bar', ax=_axes[0], color=['navy', 'coral'], alpha=0.85)
    _axes[0].set_title('Precio Promedio por Duración de Estadía (2024 vs 2025)', fontsize=12, fontweight='bold')
    _axes[0].set_xlabel('Duración de Estadía')
    _axes[0].set_ylabel('Precio Promedio ($)')
    _axes[0].legend(['2024', '2025'])
    _axes[0].tick_params(axis='x', rotation=0)
    _dias_nombre = {0: 'Lun', 1: 'Mar', 2: 'Mié', 3: 'Jue', 4: 'Vie', 5: 'Sáb', 6: 'Dom'}
    yoy_dow = df.groupby(['day_of_week', 'year'])['price_std'].mean().unstack()
    # Subplot 2: Día de la semana interanual
    yoy_dow.index = [_dias_nombre[_i] for _i in yoy_dow.index]
    yoy_dow.plot(kind='bar', ax=_axes[1], color=['navy', 'coral'], alpha=0.85)
    _axes[1].set_title('Precio Promedio por Día de la Semana (2024 vs 2025)', fontsize=12, fontweight='bold')
    _axes[1].set_xlabel('Día de la Semana')
    _axes[1].set_ylabel('Precio Promedio ($)')
    _axes[1].legend(['2024', '2025'])
    _axes[1].tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.show()
    piv_stay = df.groupby(['stay_duration', 'year'])['price_std'].mean().unstack().dropna()
    corr_stay = piv_stay[2024].corr(piv_stay[2025])
    piv_dow = df.groupby(['day_of_week', 'year'])['price_std'].mean().unstack().dropna()
    corr_dow = piv_dow[2024].corr(piv_dow[2025])
    # Cálculo dinámico de correlaciones interanuales por dimensión
    piv_dest = df.groupby(['destination_name', 'year'])['price_std'].mean().unstack().dropna()
    top_dest_common = df.groupby('destination_name').size().nlargest(30).index
    piv_dest_top = piv_dest.loc[piv_dest.index.isin(top_dest_common)]
    corr_dest = piv_dest_top[2024].corr(piv_dest_top[2025]) if len(piv_dest_top) > 2 else np.nan
    piv_month = df.groupby(['month', 'year'])['price_std'].mean().unstack().dropna()
    corr_month = piv_month[2024].corr(piv_month[2025])
    print('Resumen de Estabilidad Interanual (Correlación de Pearson 2024 vs 2025):')
    print(f'  • Duración de estadía:    r = {corr_stay:.3f}')
    print(f'  • Día de la semana:       r = {corr_dow:.3f}')
    print(f'  • Jerarquía Top Destinos: r = {corr_dest:.3f}')
    print(f'  • Patrón Mensual:         r = {corr_month:.3f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluación de consistencia interanual

    A partir de los resultados calculados, evaluamos la estabilidad entre 2024 y 2025:

    1. **Duración de estadía ($r \approx 1.000$)**: La estructura de descuentos por volumen es idéntica en ambos años; el precio por noche desciende conforme aumenta la duración.
    2. **Día de la semana ($r \approx 1.000$)**: El comportamiento intra-semanal se repite fielmente año a año (fines de semana más caros en ocio, mitad de semana en destinos ejecutivos).
    3. **Escala relativa de destinos ($r > 0.90$)**: Los destinos de alta gama y económicos conservan su posición relativa de precios entre períodos.
    4. **Variación mensual ($r \approx 0.30 - 0.70$)**: La correlación mensual fluctúa por diferencias en el volumen de consultas capturadas mes a mes, por lo que el particionado temporal debe complementarse con la técnica de **shrinkage jerárquico**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Respuestas a Preguntas de Investigación y Consignas

    ---

    ### A. Respuestas a Preguntas Disparadoras (`consignas_tp1.ipynb`)

    #### 1. ¿Los precios suben los fines de semana?
    **Sí.** A nivel global, los sábados y viernes muestran las tarifas por noche-persona más altas, mientras que los domingos y lunes registran los valores más bajos. En promedio, los fines de semana exhiben un incremento de entre **+20% y +35%** frente al piso semanal.

    #### 2. ¿Ocurre en todos los destinos por igual, o solo en los de ocio?
    **No ocurre por igual; depende del perfil del destino:**
    - **Destinos vacacionales y de ocio** (*Orlando, Miami, Cancún, Las Vegas*): el pico ocurre viernes, sábado y domingo.
    - **Destinos corporativos y urbanos** (*Detroit, Columbus, Pittsburgh*): las tarifas más altas se dan de martes a jueves (viajes de trabajo), descendiendo los fines de semana.

    #### 3. ¿Los hoteles económicos siguen el mismo ciclo estacional que los de lujo?
    **No.** Los hoteles económicos (*Budget ≤$20*) mantienen precios estables a lo largo del año debido a una demanda inelástica. Los hoteles de lujo (*Premium >$150*) exhiben una estacionalidad muy marcada, multiplicando su valor en temporada alta.

    #### 4. ¿Hay destinos donde la oferta disponible (`avg_hotel_count`) colapsa y altera los precios?
    **Sí.** En destinos insulares o de capacidad acotada (*San Juan Islands, Anacortes, zonas de playa*), cuando la disponibilidad cae a niveles mínimos (`avg_hotel_count ≤ 10`), el precio promedio sube hasta un **+45%** respecto a momentos de alta disponibilidad (`avg_hotel_count > 50`), donde la competencia modera la tarifa.

    #### 5. ¿Qué combinación de variables produce grupos más homogéneos (menor varianza interna)?
    La partición **`destination_final` $\times$ `month` $\times$ `week_in_month` $\times$ `stay_duration`** maximiza el Signal-to-Noise Ratio (**SNR**), separando la varianza entre destinos y temporadas mientras reduce la dispersión interna de cada grupo.

    #### 6. ¿Las Vegas y Henderson son el mismo mercado? ¿Y Miami vs Miami Beach?
    - **Miami vs Miami Beach**: Aunque distan menos de 10 km, **Miami Beach** (resorts de playa) tiene tarifas sistemáticamente más altas y mayor dispersión que **Miami Centro** (mercado urbano).
    - **Las Vegas vs Henderson**: Henderson dista 20 km del Strip. Su consolidación bajo *Las Vegas* provee masa crítica para el baseline, pero requiere la variable `price_bucket` para no mezclar hoteles de casino con hotelería suburbana.

    ---

    ### B. Respuestas a Preguntas Generales de Investigación (`README.md`)

    #### Pregunta 1: ¿Qué define a un mercado hotelero?
    Un mercado hotelero queda definido por la **proximidad geográfica al destino canónico (`destination_final`)**, consolidando la fragmentación de ~26.000 nombres crudos a través del mapeo Haversine y las correcciones de coherencia de mercado.

    #### Pregunta 2: ¿Qué condiciones hacen comparables dos precios (Contexto)?
    Dos precios son comparables cuando corresponden al mismo **destino canónico**, la misma **ventana temporal (mes y semana)**, igual **rango de estadía (`stay_duration`)** y opcionalmente el mismo **segmento de tarifa (`price_bucket`)**.

    #### Pregunta 3: ¿Cuántos destinos cuentan con baselines confiables?
    En el dataset completo (5,16 millones de registros y 673.030 contextos generados, con `price_bucket` activado):
    - **157.705 contextos (23,4%)** superan el umbral de alta confianza ($N \ge 30$ unidades de demanda ponderada).
    - Estos contextos de alta confianza concentran **94,3 millones de unidades de demanda (96,3% de toda la demanda real de los usuarios)**.
    - Para el 3,7% restante de la demanda en destinos de baja frecuencia (*long-tail*), el sistema aplica **fallbacks jerárquicos (Shrinkage)** para garantizar una estimación representativa.

    #### Pregunta 4: ¿Los patrones son estables en el tiempo (2024 vs 2025)?
    **Sí.** Las dimensiones de duración de estadía ($r = 1,000$), día de la semana ($r = 1,000$) y jerarquía relativa de destinos ($r > 0,90$) son altamente consistentes entre años.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # TP2 — Preparar los datos y dar forma al mercado

    **Mentoría DiploDatos 2026 · ID90Travel**

    ## Índice de consignas

    1. [Consigna oficial](#consigna-oficial-tp2)
    2. [Dataset curado: limpieza y estandarización](#111-dataset-curado-limpieza-y-estandarización)
    3. [Features de contexto construidas](#112-features-de-contexto-construidas)
    4. [Mapping de destinos aplicado](#113-mapping-de-destinos-aplicado)
    5. [Definición de mercado implementada](#114-definición-de-mercado-implementada)
    6. [Distribución de precios dentro de cada segmento](#115-distribución-de-precios-dentro-de-cada-segmento)
    7. [Estadísticos para detectar precios inusualmente bajos](#116-estadísticos-para-detectar-precios-inusualmente-bajos)
    8. [Conclusiones del TP2](#117-conclusiones-del-tp2)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Consigna oficial TP2

    > #### TP2 — Preparar los datos y dar forma al mercado · Entrega 28/08
    >
    > El TP2 tiene dos partes. La primera es operativa: limpiar y estructurar los datos para que sean comparables. Esto implica normalizar el precio a una unidad común (`precio_total / (noches × habitaciones × personas)`), construir las features de contexto que definen el mercado según lo que mostró el TP1 (día de la semana, mes, categoría de hotel, etc.), y aplicar el mapping de `destination_with_nearest.csv` para consolidar los ~26.000 nombres de ciudad en identificadores únicos — o proponer una agrupación alternativa con evidencia que la soporte. Es también el momento de auditar la calidad de los datos: precios inválidos, noches inconsistentes, búsquedas con ocupación cero.
    >
    > La segunda parte es analítica: con los datos ya segmentados en mercados, explorar cómo se distribuyen los precios dentro de cada uno. ¿Qué forma tiene esa distribución? ¿Hay precios que claramente se alejan del resto? ¿Qué estadístico distingue mejor los precios inusualmente bajos — percentiles, distancia a la media, otra cosa? Este análisis no tiene que llegar a un algoritmo final, pero sí a primeras observaciones concretas que orienten el diseño del TP3.
    >
    > La decisión más importante del TP2 es la definición de mercado que el grupo va a usar. Vale documentarla y justificarla con evidencia — porque condiciona todo lo que viene después.
    >
    > *Entrega: dataset curado con features construidas, definición de mercado implementada, y análisis exploratorio de cómo se distribuyen los precios dentro de cada segmento.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.1 Dataset curado: limpieza y estandarización

    Revisamos el pipeline de calidad aplicado en la sección 1 para verificar que el dataset quedó curado según lo requerido por la consigna.
    """)
    return


@app.cell
def _(df, df_raw, n_estandarizados, n_validos, pd):
    # Resumen del dataset curado
    tp2_quality_summary = pd.DataFrame({
        'etapa': ['raw', 'validados', 'estandarizados', 'mapeados'],
        'n_registros': [len(df_raw), n_validos, n_estandarizados, df['is_mapped'].sum()],
        'pct_del_anterior': [100.0,
                             100.0 * n_validos / len(df_raw),
                             100.0 * n_estandarizados / n_validos if n_validos > 0 else 0,
                             100.0 * df['is_mapped'].sum() / len(df) if len(df) > 0 else 0],
    })
    print('Resumen de curación del dataset')
    print(tp2_quality_summary.to_string(index=False))

    # Verificación de filtros de calidad en el dataframe final
    tp2_invalid = {
        'precio_std <= 0': (df['price_std'] <= 0).sum(),
        'noches <= 0': (df['nights'] <= 0).sum(),
        'habitaciones <= 0': (df['number_of_rooms'] <= 0).sum(),
        'ocupacion <= 0': ((df['number_of_adults'] + df['number_of_kids']) <= 0).sum(),
        'noches > 30': (df['nights'] > 30).sum(),
    }
    print('\nRegistros con problemas de calidad en df final:')
    for k, v in tp2_invalid.items():
        print(f'  {k}: {v:,}')

    print(f"\nPolítica de outliers: precios estandarizados mayores a 0, noches <= 30, y winsorización al percentil 99.9 de `avg_price_average_std`.")
    print(f"Rango final de price_std: [{df['price_std'].min():.2f}, {df['price_std'].max():.2f}]")
    print(f"Percentil 99.9 de price_std: {df['price_std'].quantile(0.999):.2f}")
    return tp2_invalid, tp2_quality_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.2 Features de contexto construidas

    El TP2 pide construir las variables que definen el mercado. Verificamos que están disponibles y documentamos su significado.
    """)
    return


@app.cell
def _(df, pd):
    tp2_features = pd.DataFrame({
        'feature': ['price_std', 'day_of_week', 'month', 'week_in_month', 'year', 'stay_duration', 'destination_final', 'destination_name'],
        'tipo': ['continua', 'ordinal', 'ordinal', 'ordinal', 'ordinal', 'categórica', 'categórica', 'categórica'],
        'descripcion': [
            'Precio por habitación-noche-persona (USD)',
            'Día de la semana del check-in (0=Lunes, 6=Domingo)',
            'Mes del check-in (1-12)',
            'Semana del mes del check-in (1-4)',
            'Año del check-in (2024/2025)',
            'Duración de la estadía: corta (1-2 noches), media (3-5), larga (>5)',
            'ID del destino canónico (post-mapping)',
            'Nombre del destino canónico (post-mapping)',
        ],
    })
    print('Features de contexto disponibles en df:')
    print(tp2_features.to_string(index=False))

    print('\nMuestra de registros con features:')
    print(df[['city', 'country_code', 'date_start', 'nights', 'number_of_rooms',
              'number_of_adults', 'number_of_kids', 'price_std', 'day_of_week',
              'month', 'week_in_month', 'year', 'stay_duration',
              'destination_final', 'destination_name']].head(5).to_string(index=False))
    return tp2_features


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.3 Mapping de destinos aplicado

    Aplicamos `destination_with_nearest.csv` para consolidar los ~26.000 nombres crudos en destinos canónicos. Reportamos cobertura por filas, por demanda ponderada y por geografías únicas, junto con la trazabilidad de cada match.
    """)
    return


@app.cell
def _(df, mapping_df, pd):
    # Cobertura general
    total_rows = len(df)
    mapped_rows = df['is_mapped'].sum()
    total_demand = df['count_repeated'].sum()
    mapped_demand = df.loc[df['is_mapped'], 'count_repeated'].sum()

    # Geografías únicas crudas vs mapeadas
    geo_cols = ['country_code', 'country', 'state', 'city']
    raw_geo = df[geo_cols].drop_duplicates()
    mapped_geo = df.loc[df['is_mapped'], geo_cols].drop_duplicates()

    tp2_mapping_summary = pd.DataFrame({
        'metrica': ['filas', 'demanda_ponderada', 'geografias_unicas'],
        'total': [total_rows, total_demand, len(raw_geo)],
        'mapeadas': [mapped_rows, mapped_demand, len(mapped_geo)],
        'pct_mapeado': [100.0 * mapped_rows / total_rows,
                        100.0 * mapped_demand / total_demand,
                        100.0 * len(mapped_geo) / len(raw_geo)],
    })
    print('Cobertura del mapping de destinos')
    print(tp2_mapping_summary.to_string(index=False))

    # Niveles de match
    if 'match_level' in df.columns:
        tp2_match_levels = df['match_level'].value_counts().reset_index()
        tp2_match_levels.columns = ['match_level', 'n_registros']
        tp2_match_levels['pct'] = 100.0 * tp2_match_levels['n_registros'] / total_rows
        print('\nDistribución por nivel de match:')
        print(tp2_match_levels.to_string(index=False))
    else:
        tp2_match_levels = None
        print('\nColumna match_level no disponible en este df.')

    # Top no mapeados por demanda
    if 'match_level' in df.columns:
        tp2_unmatched = (
            df.loc[df['match_level'] == 'no_match']
            .groupby(geo_cols, dropna=False)
            .agg(demand=('count_repeated', 'sum'), rows=('count_repeated', 'size'))
            .sort_values('demand', ascending=False)
            .head(15)
            .reset_index()
        )
        print('\nTop 15 geografías no mapeadas por demanda:')
        print(tp2_unmatched.to_string(index=False))
    else:
        tp2_unmatched = None
    return tp2_mapping_summary, tp2_match_levels, tp2_unmatched


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.4 Definición de mercado implementada

    Basándonos en la evidencia del TP1, definimos el mercado como:

    $$
    \text{Contexto} = \text{destination\_final} \times \text{month} \times \text{week\_in\_month} \times \text{stay\_duration}
    $$

    Opcionalmente se puede agregar `price_bucket` como proxy de categoría hotelera. A continuación medimos la densidad de cada versión.
    """)
    return


@app.cell
def _(df, pd):
    # Contextos sin price_bucket
    tp2_context_cols = ['destination_final', 'month', 'week_in_month', 'stay_duration']
    tp2_contexts = df.groupby(tp2_context_cols).agg(
        n_records=('price_std', 'count'),
        demand_weight=('count_repeated', 'sum'),
        mean_price=('price_std', 'mean'),
        std_price=('price_std', 'std'),
        p10=('price_std', lambda x: x.quantile(0.10)),
        p25=('price_std', lambda x: x.quantile(0.25)),
        p50=('price_std', lambda x: x.quantile(0.50)),
        p75=('price_std', lambda x: x.quantile(0.75)),
    ).reset_index()
    tp2_contexts['q_lo'] = tp2_contexts['p25'] - 1.5 * (tp2_contexts['p75'] - tp2_contexts['p25'])

    # Contextos con price_bucket (si existe)
    if 'price_bucket' in df.columns:
        tp2_contexts_bucket = df.groupby(tp2_context_cols + ['price_bucket']).agg(
            n_records=('price_std', 'count'),
            demand_weight=('count_repeated', 'sum'),
            mean_price=('price_std', 'mean'),
            std_price=('price_std', 'std'),
        ).reset_index()
    else:
        tp2_contexts_bucket = None

    # Resumen de cobertura
    def _summarize(contexts, label):
        total = len(contexts)
        high_conf = (contexts['demand_weight'] >= 30).sum()
        total_demand = contexts['demand_weight'].sum()
        high_conf_demand = contexts.loc[contexts['demand_weight'] >= 30, 'demand_weight'].sum()
        return {
            'segmentacion': label,
            'contextos_totales': total,
            'contextos_n_ge_30': high_conf,
            'pct_contextos_confiables': 100.0 * high_conf / total if total > 0 else 0,
            'demanda_total': total_demand,
            'demanda_en_confiables': high_conf_demand,
            'pct_demanda_confiable': 100.0 * high_conf_demand / total_demand if total_demand > 0 else 0,
        }

    tp2_context_summary = pd.DataFrame([
        _summarize(tp2_contexts, 'destino + mes + semana + estadia'),
    ])
    if tp2_contexts_bucket is not None:
        tp2_context_summary = pd.concat([
            tp2_context_summary,
            pd.DataFrame([_summarize(tp2_contexts_bucket, 'destino + mes + semana + estadia + price_bucket')]),
        ], ignore_index=True)

    print('Densidad de la definición de mercado propuesta')
    print(tp2_context_summary.to_string(index=False))

    # Ejemplo de contextos para un destino popular
    tp2_top_dest = df.groupby('destination_name')['count_repeated'].sum().idxmax()
    tp2_example_contexts = tp2_contexts[
        tp2_contexts['destination_final'] == df.loc[df['destination_name'] == tp2_top_dest, 'destination_final'].iloc[0]
    ].sort_values('demand_weight', ascending=False).head(10)
    print(f'\nTop 10 contextos más densos para {tp2_top_dest}:')
    print(tp2_example_contexts.to_string(index=False))
    return tp2_context_summary, tp2_contexts, tp2_contexts_bucket, tp2_example_contexts, tp2_top_dest


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.5 Distribución de precios dentro de cada segmento

    Exploramos la forma de la distribución de `price_std` dentro de los principales segmentos de mercado. Esto orienta la elección del estadístico de detección para el TP3.
    """)
    return


@app.cell
def _(df, plt, sns):
    # Destinos top por demanda
    tp2_top_dests = (
        df.groupby('destination_name')['count_repeated']
        .sum()
        .sort_values(ascending=False)
        .head(8)
        .index
    )
    tp2_df_top = df[df['destination_name'].isin(tp2_top_dests)].copy()

    # Histograma global vs destinos top
    _fig, _axes = plt.subplots(3, 3, figsize=(15, 12))
    _axes = _axes.flatten()
    sns.histplot(tp2_df_top['price_std'].clip(upper=tp2_df_top['price_std'].quantile(0.99)),
                 bins=60, kde=True, ax=_axes[0], color='steelblue')
    _axes[0].set_title('Global (recortado al p99)')
    _axes[0].set_xlabel('price_std')

    for _i, _dest in enumerate(tp2_top_dests[:8], start=1):
        _sub = tp2_df_top[tp2_df_top['destination_name'] == _dest]
        _upper = _sub['price_std'].quantile(0.99)
        sns.histplot(_sub['price_std'].clip(upper=_upper), bins=40, kde=True, ax=_axes[_i], color='teal')
        _axes[_i].set_title(_dest)
        _axes[_i].set_xlabel('price_std')

    plt.suptitle('Distribución de price_std por destino canónico (top por demanda)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    # Boxplot por stay_duration y month (global)
    _fig2, _axes2 = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df[df['price_std'] < df['price_std'].quantile(0.99)],
                x='stay_duration', y='price_std', hue='stay_duration',
                order=['corta', 'media', 'larga'], ax=_axes2[0],
                palette='Set2', legend=False)
    _axes2[0].set_title('price_std por duración de estadía')

    sns.boxplot(data=df[df['price_std'] < df['price_std'].quantile(0.99)],
                x='month', y='price_std', hue='month', ax=_axes2[1],
                palette='coolwarm', legend=False)
    _axes2[1].set_title('price_std por mes')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, pd, tp2_contexts):
    # Resumen de distribución por segmento de mercado
    tp2_dist_by_segment = tp2_contexts.copy()
    tp2_dist_by_segment['cv'] = tp2_dist_by_segment['std_price'] / tp2_dist_by_segment['mean_price']

    print('Resumen de distribución de precios por contexto de mercado')
    print(tp2_dist_by_segment[['mean_price', 'std_price', 'cv']].describe().round(3).to_string())

    # Contextos con mayor y menor coeficiente de variación
    print('\nContextos con mayor variabilidad relativa (CV):')
    print(tp2_dist_by_segment.nlargest(10, 'cv')[['destination_final', 'month', 'week_in_month', 'stay_duration',
                                                   'n_records', 'demand_weight', 'mean_price', 'std_price', 'cv']]
          .to_string(index=False))

    print('\nContextos con menor variabilidad relativa (CV):')
    print(tp2_dist_by_segment.nsmallest(10, 'cv')[['destination_final', 'month', 'week_in_month', 'stay_duration',
                                                    'n_records', 'demand_weight', 'mean_price', 'std_price', 'cv']]
          .to_string(index=False))
    return tp2_dist_by_segment


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.6 Estadísticos para detectar precios inusualmente bajos

    La consigna del TP2 pide explorar qué estadístico distingue mejor los precios inusualmente bajos. Comparamos cuatro alternativas sobre los datos curados:

    - **Percentil 10 por contexto**: precio por debajo del 10% histórico.
    - **Percentil 25 por contexto**: precio por debajo del cuartil inferior.
    - **Z-score < -1**: precio a más de 1 desvío estándar por debajo de la media del contexto.
    - **IQR (Q1 - 1.5·IQR)**: límite inferior del boxplot.

    No elegimos aún el definitivo; solo comparamos cobertura y correlación entre métodos para orientar el TP3.
    """)
    return


@app.cell
def _(df, np, pd, tp2_contexts):
    # Reutilizamos los estadísticos por contexto ya calculados en 11.4
    tp2_stats = tp2_contexts[['destination_final', 'month', 'week_in_month', 'stay_duration',
                              'mean_price', 'std_price', 'p10', 'p25', 'p75', 'q_lo']].copy()

    # Merge con una muestra representativa para evaluar los métodos
    tp2_sample = df.sample(n=min(100_000, len(df)), random_state=42).copy()
    tp2_eval = tp2_sample.merge(tp2_stats, on=['destination_final', 'month', 'week_in_month', 'stay_duration'], how='left')

    # Máscaras de detección
    tp2_eval['flag_p10'] = tp2_eval['price_std'] <= tp2_eval['p10']
    tp2_eval['flag_p25'] = tp2_eval['price_std'] <= tp2_eval['p25']
    tp2_eval['flag_zscore_lt_minus1'] = (tp2_eval['price_std'] - tp2_eval['mean_price']) / tp2_eval['std_price'] < -1
    tp2_eval['flag_iqr'] = tp2_eval['price_std'] <= tp2_eval['q_lo']

    # Resumen de cobertura
    tp2_method_summary = pd.DataFrame({
        'metodo': ['percentil_10', 'percentil_25', 'z_score_lt_-1', 'iqr_q1_1.5iqr'],
        'n_detectados': [
            tp2_eval['flag_p10'].sum(),
            tp2_eval['flag_p25'].sum(),
            tp2_eval['flag_zscore_lt_minus1'].sum(),
            tp2_eval['flag_iqr'].sum(),
        ],
        'pct_detectados': [
            100.0 * tp2_eval['flag_p10'].mean(),
            100.0 * tp2_eval['flag_p25'].mean(),
            100.0 * tp2_eval['flag_zscore_lt_minus1'].mean(),
            100.0 * tp2_eval['flag_iqr'].mean(),
        ],
    })
    print('Comparación de métodos para detectar precios inusualmente bajos (sobre muestra de 100k)')
    print(tp2_method_summary.to_string(index=False))

    # Correlación entre flags
    tp2_flag_cols = ['flag_p10', 'flag_p25', 'flag_zscore_lt_minus1', 'flag_iqr']
    tp2_corr = tp2_eval[tp2_flag_cols].astype(float).corr()
    print('\nCorrelación entre métodos (sobre registros con contexto estadístico):')
    print(tp2_corr.round(3).to_string())

    # Precio medio detectado por cada método
    tp2_mean_detected = pd.DataFrame({
        'metodo': ['percentil_10', 'percentil_25', 'z_score_lt_-1', 'iqr_q1_1.5iqr'],
        'precio_medio_detectado': [
            tp2_eval.loc[tp2_eval['flag_p10'], 'price_std'].mean(),
            tp2_eval.loc[tp2_eval['flag_p25'], 'price_std'].mean(),
            tp2_eval.loc[tp2_eval['flag_zscore_lt_minus1'], 'price_std'].mean(),
            tp2_eval.loc[tp2_eval['flag_iqr'], 'price_std'].mean(),
        ],
    })
    print('\nPrecio medio de los registros detectados por cada método:')
    print(tp2_mean_detected.to_string(index=False))
    return tp2_corr, tp2_eval, tp2_mean_detected, tp2_method_summary, tp2_stats


@app.cell
def _(plt, sns, tp2_eval):
    # Visualización de la relación entre z-score y percentil dentro de la muestra
    tp2_eval['z_score'] = (tp2_eval['price_std'] - tp2_eval['mean_price']) / tp2_eval['std_price']

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=tp2_eval.sample(n=min(20_000, len(tp2_eval)), random_state=42),
                    x='z_score', y='price_std', alpha=0.3, ax=_axes[0], color='darkblue')
    _axes[0].axvline(-1, color='red', linestyle='--', label='z = -1')
    _axes[0].set_title('Relación entre z-score y price_std')
    _axes[0].legend()

    sns.histplot(tp2_eval['z_score'].dropna().clip(-5, 5), bins=80, kde=True, ax=_axes[1], color='purple')
    _axes[1].axvline(-1, color='red', linestyle='--', label='z = -1')
    _axes[1].set_title('Distribución de z-scores por contexto (clip [-5, 5])')
    _axes[1].legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.7 Conclusiones del TP2

    1. **Dataset curado**: el pipeline elimina registros con denominador inválido, precios no positivos, noches inconsistentes y aplica winsorización al percentil 99.9. El resultado es un dataset comparable en unidades de precio por habitación-noche-persona.

    2. **Features de contexto**: están construidas las variables que definen el mercado — geografía canónica, temporalidad (mes, semana del mes, día de la semana) y duración de estadía.

    3. **Mapping de destinos**: se aplica `destination_with_nearest.csv` con trazabilidad por `match_level`. La mayor parte de la demanda queda mapeada, aunque una cola larga de geografías administrativas menores permanece como `no_match`.

    4. **Definición de mercado**: usamos `destination_final × month × week_in_month × stay_duration`, opcionalmente con `price_bucket`. La segmentación concentra la mayor parte de la demanda en contextos con suficiente masa estadística (demanda ponderada ≥ 30).

    5. **Distribución de precios**: las distribuciones por destino son asimétricas positivas con cola derecha larga. La variabilidad relativa (CV) cambia fuertemente entre contextos, lo que refuerza la necesidad de segmentar antes de comparar.

    6. **Estadísticos para detección**: el percentil 10 y el z-score < -1 detectan conjuntos parcialmente solapados pero con perfiles distintos. El percentil 10 es más estable en contextos chicos; el z-score es más sensible a la dispersión interna del contexto. La elección final queda para el TP3, donde se evaluará contra baselines ponderados y fallbacks.

    ---

    ### Decisiones metodológicas para el TP3

    - Mantener la geografía canónica como ancla inseparable del mercado.
    - Usar `month`, `week_in_month` y `stay_duration` como dimensiones de contexto.
    - Evaluar `price_bucket` como dimensión opcional, midiendo su impacto en la sensibilidad del detector.
    - Considerar el z-score ponderado por demanda como método principal, con fallback a percentiles cuando el contexto tenga baja muestra.
    """)
    return


if __name__ == "__main__":
    app.run()
