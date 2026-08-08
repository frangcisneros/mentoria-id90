# 11_analisis_cantidad_hoteles_precio.py
# Análisis: Cantidad de hoteles disponibles vs Precio

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


def analyze_hotel_count():
    """Analiza relación entre cantidad de hoteles disponibles y precio."""
    
    conn = sqlite3.connect(str(DB_PATH))
    
    query = """
    WITH valid_data AS (
        SELECT *
        FROM raw_data
        WHERE nights > 0 
            AND number_of_rooms > 0 
            AND (number_of_adults + number_of_kids) > 0
    ),
    with_price AS (
        SELECT 
            *,
            avg_price_average / (nights * number_of_rooms * MAX(number_of_adults + number_of_kids, 1)) as price_std
        FROM valid_data
    )
    SELECT 
        CASE 
            WHEN avg_hotel_count <= 5 THEN '1-5'
            WHEN avg_hotel_count <= 10 THEN '6-10'
            WHEN avg_hotel_count <= 20 THEN '11-20'
            WHEN avg_hotel_count <= 50 THEN '21-50'
            WHEN avg_hotel_count <= 100 THEN '51-100'
            ELSE '100+'
        END as hotel_count_bucket,
        AVG(avg_hotel_count) as avg_hotels,
        AVG(price_std) as mean_price,
        COUNT(*) as count_obs
    FROM with_price
    GROUP BY hotel_count_bucket
    ORDER BY MIN(avg_hotel_count)
    """
    
    print("Calculando precio por cantidad de hoteles disponibles...")
    start = time.time()
    
    df = pd.read_sql_query(query, conn)
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    
    print(f"\n{'Rango Hoteles':<15} {'Promedio':>10} {'Precio Std':>12} {'Observaciones':>15}")
    print("-" * 55)
    
    for _, row in df.iterrows():
        print(f"{row['hotel_count_bucket']:<15} {row['avg_hotels']:>10.1f} ${row['mean_price']:>10.2f} {row['count_obs']:>15,}")
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico de barras: precio promedio
    colors = sns.color_palette("viridis", len(df))
    axes[0].bar(df['hotel_count_bucket'], df['mean_price'], color=colors)
    axes[0].set_title('Precio Promedio vs Cantidad de Hoteles', fontsize=14)
    axes[0].set_xlabel('Cantidad de Hoteles Disponibles')
    axes[0].set_ylabel('Precio Estandarizado ($/habitación-noche-persona)')
    
    # Agregar valores
    for i, (bucket, price) in enumerate(zip(df['hotel_count_bucket'], df['mean_price'])):
        axes[0].text(i, price + 0.5, f'${price:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Gráfico de barras: cantidad de observaciones
    axes[1].bar(df['hotel_count_bucket'], df['count_obs'], color=colors)
    axes[1].set_title('Cantidad de Búsquedas por Disponibilidad', fontsize=14)
    axes[1].set_xlabel('Cantidad de Hoteles Disponibles')
    axes[1].set_ylabel('Cantidad de Observaciones')
    
    # Agregar valores
    for i, (bucket, count) in enumerate(zip(df['hotel_count_bucket'], df['count_obs'])):
        axes[1].text(i, count + 1000, f'{count:,}', ha='center', va='bottom', fontsize=9, rotation=45)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '11_precio_por_cantidad_hoteles.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráfico guardado en: {output_path}")
    
    # Guardar datos
    csv_path = OUTPUT_DIR / 'data' / '11_precio_por_cantidad_hoteles.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Datos guardados en: {csv_path}")
    
    conn.close()
    return df


def analyze_hotel_count_scatter():
    """Análisis de dispersión: hoteles vs precio."""
    
    conn = sqlite3.connect(str(DB_PATH))
    
    query = """
    WITH valid_data AS (
        SELECT *
        FROM raw_data
        WHERE nights > 0 
            AND number_of_rooms > 0 
            AND (number_of_adults + number_of_kids) > 0
    ),
    with_price AS (
        SELECT 
            *,
            avg_price_average / (nights * number_of_rooms * MAX(number_of_adults + number_of_kids, 1)) as price_std
        FROM valid_data
    )
    SELECT 
        avg_hotel_count,
        price_std
    FROM with_price
    WHERE avg_hotel_count <= 100
    LIMIT 10000
    """
    
    print("\nGenerando gráfico de dispersión...")
    start = time.time()
    
    df = pd.read_sql_query(query, conn)
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    
    # Visualización
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Muestra aleatoria para scatter
    sample = df.sample(min(5000, len(df)), random_state=42)
    
    scatter = ax.scatter(sample['avg_hotel_count'], sample['price_std'], 
                        alpha=0.3, s=10, c=sample['price_std'], cmap='viridis')
    
    ax.set_title('Relación entre Cantidad de Hoteles y Precio', fontsize=14)
    ax.set_xlabel('Cantidad de Hoteles Disponibles')
    ax.set_ylabel('Precio Estandarizado ($/habitación-noche-persona)')
    ax.grid(True, alpha=0.3)
    
    # Línea de tendencia
    z = np.polyfit(sample['avg_hotel_count'], sample['price_std'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(sample['avg_hotel_count'].min(), sample['avg_hotel_count'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Tendencia (pendiente: {z[0]:.2f})')
    ax.legend()
    
    plt.colorbar(scatter, label='Precio Estandarizado')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '11_scatter_hoteles_precio.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico guardado en: {output_path}")
    
    conn.close()
    return df


if __name__ == "__main__":
    print("="*60)
    print("ANÁLISIS: CANTIDAD DE HOTELES VS PRECIO")
    print("="*60)
    
    # Análisis por buckets
    df_buckets = analyze_hotel_count()
    
    # Análisis de dispersión
    df_scatter = analyze_hotel_count_scatter()
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
