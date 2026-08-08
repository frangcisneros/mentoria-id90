# 14_baselines_confiables_y_mapeo.py
# ¿Cuántos destinos tienen baselines confiables? % registros sin mapeo

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
import config

DB_PATH = Path(config.BASE_DIR) / 'data' / 'hotel_data.db'
OUTPUT_DIR = Path(config.BASE_DIR) / 'TP1' / 'outputs'


def analyze_baselines_confiables():
    """Analiza qué destinos tienen suficientes datos para baselines confiables."""
    
    conn = sqlite3.connect(str(DB_PATH))
    
    # Cargar mapping
    mapping_df = pd.read_csv(config.DESTINATION_MAPPING_FILE)
    mapping_df.to_sql('mapping', conn, if_exists='replace', index=False)
    
    # Primero calcular baselines
    query_baselines = """
    WITH valid_data AS (
        SELECT *
        FROM raw_data
        WHERE nights > 0 
            AND number_of_rooms > 0 
            AND (number_of_adults + number_of_kids) > 0
    ),
    standardized AS (
        SELECT 
            *,
            avg_price_average / (nights * number_of_rooms * MAX(number_of_adults + number_of_kids, 1)) as price_std,
            CAST(strftime('%m', date_start) AS INTEGER) as month,
            CASE 
                WHEN CAST(strftime('%d', date_start) AS INTEGER) <= 7 THEN 1
                WHEN CAST(strftime('%d', date_start) AS INTEGER) <= 15 THEN 2
                WHEN CAST(strftime('%d', date_start) AS INTEGER) <= 22 THEN 3
                ELSE 4
            END as week_in_month
        FROM valid_data
    ),
    with_destination AS (
        SELECT 
            s.*,
            COALESCE(
                m3.nearest_destination_id,
                m2.nearest_destination_id,
                m1.nearest_destination_id,
                s.city
            ) as destination_final,
            COALESCE(
                m3.nearest_destination_name,
                m2.nearest_destination_name,
                m1.nearest_destination_name,
                s.city
            ) as destination_name
        FROM standardized s
        LEFT JOIN mapping m3 ON UPPER(s.country_code) || ' - ' || UPPER(s.state) || ' - ' || s.city = m3.reference
        LEFT JOIN mapping m2 ON UPPER(s.country_code) || ' - ' || s.city = m2.reference
        LEFT JOIN mapping m1 ON UPPER(s.country_code) || ' - ' || UPPER(s.state) = m1.reference
    )
    SELECT 
        destination_final,
        destination_name,
        month,
        week_in_month,
        COUNT(*) as count_obs,
        AVG(price_std) as mean_price
    FROM with_destination
    WHERE destination_final IS NOT NULL
    GROUP BY destination_final, destination_name, month, week_in_month
    """
    
    print("Calculando baselines por destino...")
    start = time.time()
    
    baselines = pd.read_sql_query(query_baselines, conn)
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    print(f"Total baselines: {len(baselines):,}")
    
    # Definir umbral de confiabilidad
    MIN_OBSERVATIONS = 30  # Mínimo de observaciones por contexto
    
    baselines['confiable'] = baselines['count_obs'] >= MIN_OBSERVATIONS
    
    # Resumen por destino
    resumen = baselines.groupby(['destination_final', 'destination_name']).agg(
        total_contexts=('month', 'count'),
        confiables=('confiable', 'sum'),
        total_obs=('count_obs', 'sum')
    ).reset_index()
    
    resumen['pct_confiables'] = (resumen['confiables'] / resumen['total_contexts'] * 100).round(1)
    
    # Clasificar destinos
    resumen['nivel_confianza'] = pd.cut(
        resumen['pct_confiables'],
        bins=[0, 25, 50, 75, 100],
        labels=['Baja (0-25%)', 'Media (25-50%)', 'Alta (50-75%)', 'Muy Alta (75-100%)']
    )
    
    print(f"\nResumen de confiabilidad (mínimo {MIN_OBSERVATIONS} obs/contexto):")
    print(f"{'Nivel':<20} {'Destinos':>10} {'Porcentaje':>10}")
    print("-" * 45)
    
    nivel_counts = resumen['nivel_confianza'].value_counts().sort_index()
    total_destinos = len(resumen)
    
    for nivel, count in nivel_counts.items():
        pct = count / total_destinos * 100
        print(f"{str(nivel):<20} {count:>10} {pct:>9.1f}%")
    
    # Top destinos más confiables
    print(f"\nTop 20 destinos más confiables (con más obs):")
    top_confiables = resumen.nlargest(20, 'total_obs')[['destination_name', 'total_contexts', 'confiables', 'pct_confiables', 'total_obs']]
    print(top_confiables.to_string(index=False))
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Distribución de niveles de confianza
    colors = ['#ff6b6b', '#feca57', '#48dbfb', '#0abde3']
    nivel_counts.plot(kind='bar', ax=axes[0], color=colors)
    axes[0].set_title('Distribución de Confiabilidad de Baselines', fontsize=14)
    axes[0].set_xlabel('Nivel de Confianza')
    axes[0].set_ylabel('Cantidad de Destinos')
    axes[0].tick_params(axis='x', rotation=45)
    
    for i, (nivel, count) in enumerate(nivel_counts.items()):
        axes[0].text(i, count + 50, f'{count}', ha='center', va='bottom', fontsize=10)
    
    # Histograma de observaciones por destino
    axes[1].hist(resumen['total_obs'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=MIN_OBSERVATIONS * 12, color='red', linestyle='--', 
                    label=f'Umbral confiabilidad ({MIN_OBSERVATIONS * 12} obs totales)')
    axes[1].set_title('Distribución de Observaciones por Destino', fontsize=14)
    axes[1].set_xlabel('Total de Observaciones')
    axes[1].set_ylabel('Cantidad de Destinos')
    axes[1].legend()
    axes[1].set_xlim(0, 5000)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '14_confiabilidad_baselines.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráfico guardado en: {output_path}")
    
    # Guardar datos
    csv_path = OUTPUT_DIR / 'data' / '14_confiabilidad_baselines.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(csv_path, index=False)
    print(f"Datos guardados en: {csv_path}")
    
    conn.close()
    return resumen


def analyze_unmapped_by_country():
    """Analiza el porcentaje de registros sin mapeo por país/región."""
    
    conn = sqlite3.connect(str(DB_PATH))
    
    # Cargar mapping
    mapping_df = pd.read_csv(config.DESTINATION_MAPPING_FILE)
    mapping_df.to_sql('mapping', conn, if_exists='replace', index=False)
    
    query = """
    WITH valid_data AS (
        SELECT *
        FROM raw_data
        WHERE nights > 0 
            AND number_of_rooms > 0 
            AND (number_of_adults + number_of_kids) > 0
    ),
    with_mapping AS (
        SELECT 
            s.*,
            COALESCE(
                m3.nearest_destination_id,
                m2.nearest_destination_id,
                m1.nearest_destination_id
            ) as is_mapped
        FROM valid_data s
        LEFT JOIN mapping m3 ON UPPER(s.country_code) || ' - ' || UPPER(s.state) || ' - ' || s.city = m3.reference
        LEFT JOIN mapping m2 ON UPPER(s.country_code) || ' - ' || s.city = m2.reference
        LEFT JOIN mapping m1 ON UPPER(s.country_code) || ' - ' || UPPER(s.state) = m1.reference
    )
    SELECT 
        UPPER(country_code) as country_code,
        country,
        COUNT(*) as total_records,
        SUM(CASE WHEN is_mapped IS NOT NULL THEN 1 ELSE 0 END) as mapped_records,
        SUM(CASE WHEN is_mapped IS NULL THEN 1 ELSE 0 END) as unmapped_records,
        ROUND(SUM(CASE WHEN is_mapped IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_unmapped
    FROM with_mapping
    GROUP BY country_code, country
    ORDER BY total_records DESC
    """
    
    print("\nAnalizando registros sin mapeo por país...")
    start = time.time()
    
    df = pd.read_sql_query(query, conn)
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    
    print(f"\n{'País':<5} {'Nombre':<25} {'Total':>12} {'Mapeados':>12} {'Sin Mapeo':>12} {'% Sin Mapeo':>12}")
    print("-" * 80)
    
    for _, row in df.head(20).iterrows():
        print(f"{row['country_code']:<5} {str(row['country'])[:24]:<25} {row['total_records']:>12,} {row['mapped_records']:>12,} {row['unmapped_records']:>12,} {row['pct_unmapped']:>11.1f}%")
    
    # Top 10 países con más registros sin mapeo
    top_unmapped = df.nlargest(10, 'unmapped_records')
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top países con más registros sin mapeo
    axes[0].barh(top_unmapped['country_code'], top_unmapped['unmapped_records'], color='coral')
    axes[0].set_title('Top 10 Países con Más Registros Sin Mapeo', fontsize=14)
    axes[0].set_xlabel('Cantidad de Registros Sin Mapeo')
    axes[0].set_ylabel('País')
    
    for i, (code, count) in enumerate(zip(top_unmapped['country_code'], top_unmapped['unmapped_records'])):
        axes[0].text(count + 1000, i, f'{count:,}', ha='left', va='center', fontsize=9)
    
    # % sin mapeo por país (top 10 por volumen)
    top_volume = df.head(10)
    axes[1].bar(top_volume['country_code'], top_volume['pct_unmapped'], color='steelblue')
    axes[1].set_title('% Registros Sin Mapeo por País (Top 10 por Volumen)', fontsize=14)
    axes[1].set_xlabel('País')
    axes[1].set_ylabel('% Sin Mapeo')
    axes[1].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
    axes[1].legend()
    
    for i, (code, pct) in enumerate(zip(top_volume['country_code'], top_volume['pct_unmapped'])):
        axes[1].text(i, pct + 1, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '14_registros_sin_mapeo_por_pais.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráfico guardado en: {output_path}")
    
    # Guardar datos
    csv_path = OUTPUT_DIR / 'data' / '14_registros_sin_mapeo_por_pais.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Datos guardados en: {csv_path}")
    
    conn.close()
    return df


if __name__ == "__main__":
    print("="*60)
    print("BASELINES CONFIABLES Y ANÁLISIS DE MAPEO")
    print("="*60)
    
    # Baselines confiables
    df_confianza = analyze_baselines_confiables()
    
    # Registros sin mapeo
    df_unmapped = analyze_unmapped_by_country()
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
