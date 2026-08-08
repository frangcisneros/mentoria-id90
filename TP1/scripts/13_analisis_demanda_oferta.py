# 13_analisis_demanda_oferta.py
# Análisis de count_repeated y avg_hotel_count como proxies de demanda y oferta

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


def analyze_count_repeated():
    """Analiza count_repeated como proxy de demanda."""
    
    conn = sqlite3.connect(str(DB_PATH))
    
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
    )
    SELECT 
        CASE 
            WHEN count_repeated = 1 THEN '1'
            WHEN count_repeated <= 3 THEN '2-3'
            WHEN count_repeated <= 5 THEN '4-5'
            WHEN count_repeated <= 10 THEN '6-10'
            WHEN count_repeated <= 20 THEN '11-20'
            ELSE '20+'
        END as repeated_bucket,
        AVG(count_repeated) as avg_repeated,
        AVG(price_std) as mean_price,
        AVG(avg_price_average) as mean_total_price,
        COUNT(*) as count_obs
    FROM standardized
    GROUP BY repeated_bucket
    ORDER BY MIN(count_repeated)
    """
    
    print("Analizando count_repeated (proxy de demanda)...")
    start = time.time()
    
    df = pd.read_sql_query(query, conn)
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    
    print(f"\n{'count_repeated':<15} {'Promedio':>10} {'Precio Std':>12} {'Precio Total':>15} {'Observaciones':>15}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        print(f"{row['repeated_bucket']:<15} {row['avg_repeated']:>10.1f} ${row['mean_price']:>10.2f} ${row['mean_total_price']:>13.2f} {row['count_obs']:>15,}")
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = sns.color_palette("viridis", len(df))
    
    # Precio vs count_repeated
    axes[0].bar(df['repeated_bucket'], df['mean_price'], color=colors)
    axes[0].set_title('Precio Promedio vs Veces Repetido', fontsize=14)
    axes[0].set_xlabel('Veces que aparece la búsqueda (count_repeated)')
    axes[0].set_ylabel('Precio Estandarizado ($/habitación-noche-persona)')
    
    for i, (bucket, price) in enumerate(zip(df['repeated_bucket'], df['mean_price'])):
        axes[0].text(i, price + 0.5, f'${price:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Precio total vs count_repeated
    axes[1].bar(df['repeated_bucket'], df['mean_total_price'], color=colors)
    axes[1].set_title('Precio Total vs Veces Repetido', fontsize=14)
    axes[1].set_xlabel('Veces que aparece la búsqueda (count_repeated)')
    axes[1].set_ylabel('Precio Total ($)')
    
    for i, (bucket, price) in enumerate(zip(df['repeated_bucket'], df['mean_total_price'])):
        axes[1].text(i, price + 5, f'${price:.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '13_precio_vs_count_repeated.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráfico guardado en: {output_path}")
    
    conn.close()
    return df


def analyze_hotel_count():
    """Analiza avg_hotel_count como proxy de oferta."""
    
    conn = sqlite3.connect(str(DB_PATH))
    
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
    )
    SELECT 
        CASE 
            WHEN avg_hotel_count <= 5 THEN '1-5 hoteles'
            WHEN avg_hotel_count <= 15 THEN '6-15 hoteles'
            WHEN avg_hotel_count <= 30 THEN '16-30 hoteles'
            WHEN avg_hotel_count <= 50 THEN '31-50 hoteles'
            WHEN avg_hotel_count <= 100 THEN '51-100 hoteles'
            ELSE '100+ hoteles'
        END as hotel_bucket,
        AVG(avg_hotel_count) as avg_hotels,
        AVG(price_std) as mean_price,
        AVG(avg_price_average) as mean_total_price,
        COUNT(*) as count_obs
    FROM standardized
    GROUP BY hotel_bucket
    ORDER BY MIN(avg_hotel_count)
    """
    
    print("\nAnalizando avg_hotel_count (proxy de oferta)...")
    start = time.time()
    
    df = pd.read_sql_query(query, conn)
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    
    print(f"\n{'Hoteles':<15} {'Promedio':>10} {'Precio Std':>12} {'Precio Total':>15} {'Observaciones':>15}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        print(f"{row['hotel_bucket']:<15} {row['avg_hotels']:>10.1f} ${row['mean_price']:>10.2f} ${row['mean_total_price']:>13.2f} {row['count_obs']:>15,}")
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = sns.color_palette("magma", len(df))
    
    # Precio vs hoteles
    axes[0].bar(df['hotel_bucket'], df['mean_price'], color=colors)
    axes[0].set_title('Precio Promedio vs Cantidad de Hoteles', fontsize=14)
    axes[0].set_xlabel('Cantidad de Hoteles Disponibles')
    axes[0].set_ylabel('Precio Estandarizado ($/habitación-noche-persona)')
    
    for i, (bucket, price) in enumerate(zip(df['hotel_bucket'], df['mean_price'])):
        axes[0].text(i, price + 0.5, f'${price:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Volumen por bucket
    axes[1].bar(df['hotel_bucket'], df['count_obs'], color=colors)
    axes[1].set_title('Cantidad de Búsquedas por Disponibilidad', fontsize=14)
    axes[1].set_xlabel('Cantidad de Hoteles Disponibles')
    axes[1].set_ylabel('Cantidad de Observaciones')
    
    for i, (bucket, count) in enumerate(zip(df['hotel_bucket'], df['count_obs'])):
        axes[1].text(i, count + 50000, f'{count:,}', ha='center', va='bottom', fontsize=9, rotation=45)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '13_precio_vs_hotel_count.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráfico guardado en: {output_path}")
    
    conn.close()
    return df


def analyze_correlation():
    """Análisis de correlación entre variables de demanda/oferta y precio."""
    
    conn = sqlite3.connect(str(DB_PATH))
    
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
    )
    SELECT 
        price_std,
        count_repeated,
        avg_hotel_count,
        avg_price_average as total_price,
        nights,
        number_of_adults + number_of_kids as total_persons
    FROM standardized
    LIMIT 50000
    """
    
    print("\nAnálisis de correlaciones...")
    start = time.time()
    
    df = pd.read_sql_query(query, conn)
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    
    # Matriz de correlación
    corr_matrix = df[['price_std', 'count_repeated', 'avg_hotel_count', 
                       'total_price', 'nights', 'total_persons']].corr()
    
    print("\nMatriz de correlación:")
    print(corr_matrix.round(3).to_string())
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, ax=ax)
    
    ax.set_title('Correlación entre Variables', fontsize=14)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '13_matriz_correlacion.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nHeatmap guardado en: {output_path}")
    
    conn.close()
    return corr_matrix


if __name__ == "__main__":
    print("="*60)
    print("ANÁLISIS DE DEMANDA Y OFERTA")
    print("="*60)
    
    # Análisis de count_repeated
    df_repeated = analyze_count_repeated()
    
    # Análisis de hotel_count
    df_hotels = analyze_hotel_count()
    
    # Correlaciones
    corr = analyze_correlation()
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
