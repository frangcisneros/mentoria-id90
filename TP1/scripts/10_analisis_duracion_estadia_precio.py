# 10_analisis_duracion_estadia_precio.py
# Análisis: Duración de estadía vs Precio

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import sqlite3
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
import config

DB_PATH = Path(config.BASE_DIR) / 'data' / 'hotel_data.db'
OUTPUT_DIR = Path(config.BASE_DIR) / 'TP1' / 'outputs'


def analyze_stay_duration():
    """Analiza relación entre duración de estadía y precio."""
    
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
            avg_price_average / (nights * number_of_rooms * MAX(number_of_adults + number_of_kids, 1)) as price_std,
            -- Total price
            avg_price_average as total_price
        FROM valid_data
    )
    SELECT 
        nights,
        AVG(price_std) as mean_price_std,
        AVG(total_price) as mean_total_price,
        COUNT(*) as count_obs
    FROM standardized
    GROUP BY nights
    HAVING COUNT(*) >= 100
    ORDER BY nights
    """
    
    print("Calculando precio por duración de estadía...")
    start = time.time()
    
    df = pd.read_sql_query(query, conn)
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    
    print(f"\n{'Noches':<10} {'Precio Std':>12} {'Precio Total':>15} {'Observaciones':>15}")
    print("-" * 55)
    
    for _, row in df.iterrows():
        print(f"{int(row['nights']):<10} ${row['mean_price_std']:>10.2f} ${row['mean_total_price']:>13.2f} {row['count_obs']:>15,}")
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico de línea: precio estandarizado vs noches
    axes[0].plot(df['nights'], df['mean_price_std'], 'o-', linewidth=2, markersize=8)
    axes[0].set_title('Precio Estandarizado vs Duración de Estadía', fontsize=14)
    axes[0].set_xlabel('Noches')
    axes[0].set_ylabel('Precio Estandarizado ($/habitación-noche-persona)')
    axes[0].grid(True, alpha=0.3)
    
    # Agregar valores
    for x, y in zip(df['nights'], df['mean_price_std']):
        axes[0].annotate(f'${y:.1f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    # Gráfico de línea: precio total vs noches
    axes[1].plot(df['nights'], df['mean_total_price'], 's-', linewidth=2, markersize=8, color='orange')
    axes[1].set_title('Precio Total vs Duración de Estadía', fontsize=14)
    axes[1].set_xlabel('Noches')
    axes[1].set_ylabel('Precio Total ($)')
    axes[1].grid(True, alpha=0.3)
    
    # Agregar valores
    for x, y in zip(df['nights'], df['mean_total_price']):
        axes[1].annotate(f'${y:.0f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '10_precio_por_duracion_estadia.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráfico guardado en: {output_path}")
    
    # Guardar datos
    csv_path = OUTPUT_DIR / 'data' / '10_precio_por_duracion_estadia.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Datos guardados en: {csv_path}")
    
    conn.close()
    return df


def analyze_stay_duration_by_destination(top_n=5):
    """Analiza duración vs precio para los destinos más populares."""
    
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
        nights,
        AVG(price_std) as mean_price,
        COUNT(*) as count_obs
    FROM with_destination
    WHERE destination_name IS NOT NULL
    GROUP BY destination_name, nights
    HAVING COUNT(*) >= 50
    """
    
    print(f"\nCalculando precio por duración para top {top_n} destinos...")
    start = time.time()
    
    df = pd.read_sql_query(query, conn)
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    
    # Top destinos por volumen
    top_dests = df.groupby('destination_name')['count_obs'].sum().nlargest(top_n).index
    
    # Filtrar solo top destinos
    df_top = df[df['destination_name'].isin(top_dests)]
    
    # Visualización
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for dest in top_dests:
        dest_data = df_top[df_top['destination_name'] == dest].sort_values('nights')
        # Filtrar noches razonables (1-14)
        dest_data = dest_data[dest_data['nights'] <= 14]
        ax.plot(dest_data['nights'], dest_data['mean_price'], 'o-', label=dest, linewidth=2, markersize=6)
    
    ax.set_title('Precio Estandarizado vs Duración de Estadía por Destino', fontsize=14)
    ax.set_xlabel('Noches')
    ax.set_ylabel('Precio Estandarizado ($/habitación-noche-persona)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 15)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '10_precio_por_duracion_destino.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráfico guardado en: {output_path}")
    
    conn.close()
    return df_top


if __name__ == "__main__":
    print("="*60)
    print("ANÁLISIS: DURACIÓN DE ESTADÍA VS PRECIO")
    print("="*60)
    
    # Análisis general
    df_general = analyze_stay_duration()
    
    # Análisis por destino
    df_destinos = analyze_stay_duration_by_destination(top_n=5)
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
