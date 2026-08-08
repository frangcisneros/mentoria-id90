# 12_distribucion_precios_por_destino.py
# Distribución de precios por destino: boxplots y heatmap destino x mes

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


def precio_por_destino_boxplot(top_n=20):
    """Boxplot de precios para los top N destinos por volumen."""
    
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
    standardized AS (
        SELECT 
            *,
            avg_price_average / (nights * number_of_rooms * MAX(number_of_adults + number_of_kids, 1)) as price_std
        FROM valid_data
    ),
    with_destination AS (
        SELECT 
            s.*,
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
        destination_name,
        price_std
    FROM with_destination
    WHERE destination_name IS NOT NULL
    """
    
    print(f"Generando boxplot de precios por destino (top {top_n})...")
    start = time.time()
    
    df = pd.read_sql_query(query, conn)
    
    # Top destinos por volumen
    top_dests = df.groupby('destination_name').size().nlargest(top_n).index
    df_top = df[df['destination_name'].isin(top_dests)]
    
    # Calcular mediana para ordenar
    median_order = df_top.groupby('destination_name')['price_std'].median().sort_values(ascending=False).index
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    print(f"Registros: {len(df_top):,}")
    
    # Boxplot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    df_top['destination_name'] = pd.Categorical(df_top['destination_name'], categories=median_order, ordered=True)
    
    sns.boxplot(data=df_top, x='price_std', y='destination_name', ax=ax, 
                showfliers=False, palette='viridis')
    
    ax.set_title(f'Distribución de Precios por Destino (Top {top_n})', fontsize=14)
    ax.set_xlabel('Precio Estandarizado ($/habitación-noche-persona)')
    ax.set_ylabel('Destino')
    ax.set_xlim(0, 200)  # Limitar para mejor visualización
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '12_boxplot_precios_por_destino.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Boxplot guardado en: {output_path}")
    
    # Estadísticas
    print(f"\nEstadísticas por destino:")
    stats = df_top.groupby('destination_name')['price_std'].agg(['median', 'mean', 'std', 'count'])
    stats = stats.loc[median_order]
    print(stats.round(2).to_string())
    
    conn.close()
    return df_top


def heatmap_destino_mes():
    """Heatmap de precio promedio por destino y mes."""
    
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
    standardized AS (
        SELECT 
            *,
            avg_price_average / (nights * number_of_rooms * MAX(number_of_adults + number_of_kids, 1)) as price_std,
            CAST(strftime('%m', date_start) AS INTEGER) as month
        FROM valid_data
    ),
    with_destination AS (
        SELECT 
            s.*,
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
        destination_name,
        month,
        AVG(price_std) as mean_price,
        COUNT(*) as count_obs
    FROM with_destination
    WHERE destination_name IS NOT NULL
    GROUP BY destination_name, month
    """
    
    print("\nGenerando heatmap destino x mes...")
    start = time.time()
    
    df = pd.read_sql_query(query, conn)
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    
    # Top 15 destinos por volumen
    top_dests = df.groupby('destination_name')['count_obs'].sum().nlargest(15).index
    df_top = df[df['destination_name'].isin(top_dests)]
    
    # Pivot
    pivot = df_top.pivot_table(index='destination_name', columns='month', values='mean_price')
    pivot.columns = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                     'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, 
                linewidths=0.5, cbar_kws={'label': 'Precio Promedio ($)'})
    
    ax.set_title('Precio Promedio por Destino y Mes', fontsize=14)
    ax.set_xlabel('Mes')
    ax.set_ylabel('Destino')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '12_heatmap_destino_mes.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap guardado en: {output_path}")
    
    conn.close()
    return df_top


if __name__ == "__main__":
    print("="*60)
    print("DISTRIBUCIÓN DE PRECIOS POR DESTINO")
    print("="*60)
    
    # Boxplot
    df_boxplot = precio_por_destino_boxplot(top_n=20)
    
    # Heatmap
    df_heatmap = heatmap_destino_mes()
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
