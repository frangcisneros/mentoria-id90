# 09_analisis_dia_semana_precio.py
# Análisis: Día de la semana vs Precio

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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_day_of_week():
    """Analiza relación entre día de la semana y precio."""
    
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
            CAST(strftime('%w', date_start) AS INTEGER) as day_of_week,
            CASE CAST(strftime('%w', date_start) AS INTEGER)
                WHEN 0 THEN 'Domingo'
                WHEN 1 THEN 'Lunes'
                WHEN 2 THEN 'Martes'
                WHEN 3 THEN 'Miércoles'
                WHEN 4 THEN 'Jueves'
                WHEN 5 THEN 'Viernes'
                WHEN 6 THEN 'Sábado'
            END as day_name
        FROM valid_data
    )
    SELECT 
        day_of_week,
        day_name,
        AVG(price_std) as mean_price,
        (MAX(price_std) - MIN(price_std)) / 4.0 as std_price,
        COUNT(*) as count_obs
    FROM standardized
    GROUP BY day_of_week, day_name
    ORDER BY day_of_week
    """
    
    print("Calculando precio promedio por día de la semana...")
    start = time.time()
    
    df = pd.read_sql_query(query, conn)
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    print(f"\n{'Día':<12} {'Precio Promedio':>15} {'Desv. Est.':>12} {'Observaciones':>15}")
    print("-" * 55)
    
    for _, row in df.iterrows():
        print(f"{row['day_name']:<12} ${row['mean_price']:>13.2f} ${row['std_price']:>10.2f} {row['count_obs']:>15,}")
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico de barras: precio promedio
    colors = sns.color_palette("viridis", len(df))
    axes[0].bar(df['day_name'], df['mean_price'], color=colors)
    axes[0].set_title('Precio Promedio por Día de la Semana', fontsize=14)
    axes[0].set_xlabel('Día')
    axes[0].set_ylabel('Precio Estandarizado ($/habitación-noche-persona)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Agregar valores
    for i, (name, price) in enumerate(zip(df['day_name'], df['mean_price'])):
        axes[0].text(i, price + 0.5, f'${price:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Gráfico de barras: cantidad de observaciones
    axes[1].bar(df['day_name'], df['count_obs'], color=colors)
    axes[1].set_title('Cantidad de Búsquedas por Día', fontsize=14)
    axes[1].set_xlabel('Día')
    axes[1].set_ylabel('Cantidad de Observaciones')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Agregar valores
    for i, (name, count) in enumerate(zip(df['day_name'], df['count_obs'])):
        axes[1].text(i, count + 500, f'{count:,}', ha='center', va='bottom', fontsize=9, rotation=45)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '09_precio_por_dia_semana.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráfico guardado en: {output_path}")
    
    # Guardar datos
    csv_path = OUTPUT_DIR / 'data' / '09_precio_por_dia_semana.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Datos guardados en: {csv_path}")
    
    conn.close()
    return df


def analyze_day_of_week_by_destination(top_n=10):
    """Analiza día de semana vs precio para los destinos más populares."""
    
    conn = sqlite3.connect(str(DB_PATH))
    
    # Cargar mapping para la query
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
            CAST(strftime('%w', date_start) AS INTEGER) as day_of_week,
            CASE CAST(strftime('%w', date_start) AS INTEGER)
                WHEN 0 THEN 'Dom'
                WHEN 1 THEN 'Lun'
                WHEN 2 THEN 'Mar'
                WHEN 3 THEN 'Mié'
                WHEN 4 THEN 'Jue'
                WHEN 5 THEN 'Vie'
                WHEN 6 THEN 'Sáb'
            END as day_name
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
        day_of_week,
        day_name,
        AVG(price_std) as mean_price,
        COUNT(*) as count_obs
    FROM with_destination
    WHERE destination_name IS NOT NULL
    GROUP BY destination_name, day_of_week, day_name
    """
    
    print(f"\nCalculando precio por día para top {top_n} destinos...")
    start = time.time()
    
    df = pd.read_sql_query(query, conn)
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    
    # Top destinos por volumen
    top_dests = df.groupby('destination_name')['count_obs'].sum().nlargest(top_n).index
    
    # Filtrar solo top destinos
    df_top = df[df['destination_name'].isin(top_dests)]
    
    # Pivot table
    pivot = df_top.pivot_table(
        index='destination_name', 
        columns='day_name', 
        values='mean_price',
        aggfunc='mean'
    )
    
    # Ordenar días
    day_order = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    pivot = pivot.reindex(columns=[d for d in day_order if d in pivot.columns])
    
    print("\nPrecio promedio por destino y día ($/habitación-noche-persona):")
    print(pivot.round(2).to_string())
    
    # Visualización
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
    ax.set_title('Precio Promedio por Destino y Día de la Semana', fontsize=14)
    ax.set_xlabel('Día de la Semana')
    ax.set_ylabel('Destino')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '09_heatmap_precio_dia_destino.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nHeatmap guardado en: {output_path}")
    
    conn.close()
    return df_top


if __name__ == "__main__":
    print("="*60)
    print("ANÁLISIS: DÍA DE LA SEMANA VS PRECIO")
    print("="*60)
    
    # Análisis general
    df_general = analyze_day_of_week()
    
    # Análisis por destino
    df_destinos = analyze_day_of_week_by_destination(top_n=10)
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
