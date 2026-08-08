# 16_patrones_semanales.py
# Análisis de patrones semanales: semana 1 vs semana 3 del mismo mes

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


def analizar_patrones_semanales():
    """Compara precios entre semana 1 y semana 3 del mismo mes."""
    
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
        week_in_month,
        AVG(price_std) as mean_price,
        COUNT(*) as count_obs
    FROM with_destination
    WHERE destination_name IS NOT NULL
    GROUP BY destination_name, month, week_in_month
    """
    
    print("Analizando patrones semanales (Sem 1 vs Sem 3)...")
    start = time.time()
    
    df = pd.read_sql_query(query, conn)
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    
    # Top 10 destinos por volumen
    top_dests = df.groupby('destination_name')['count_obs'].sum().nlargest(10).index
    
    # Filtrar solo semana 1 y 3
    df_sem1 = df[df['week_in_month'] == 1].copy()
    df_sem3 = df[df['week_in_month'] == 3].copy()
    
    # Merge por destino y mes
    comparison = df_sem1.merge(
        df_sem3, 
        on=['destination_name', 'month'], 
        suffixes=('_sem1', '_sem3')
    )
    
    # Calcular diferencia porcentual
    comparison['diff_pct'] = ((comparison['mean_price_sem3'] - comparison['mean_price_sem1']) / 
                              comparison['mean_price_sem1'] * 100)
    
    print(f"\nComparación Semana 1 vs Semana 3 del mismo mes:")
    print(f"{'Destino':<20} {'Mes':>5} {'Sem 1':>10} {'Sem 3':>10} {'Diferencia':>12}")
    print("-" * 65)
    
    for dest in top_dests:
        dest_data = comparison[comparison['destination_name'] == dest].sort_values('month')
        for _, row in dest_data.head(6).iterrows():
            print(f"{row['destination_name'][:19]:<20} {int(row['month']):>5} "
                  f"${row['mean_price_sem1']:>8.2f} ${row['mean_price_sem3']:>8.2f} "
                  f"{row['diff_pct']:>+10.1f}%")
    
    # Resumen por destino
    resumen = comparison.groupby('destination_name').agg(
        avg_sem1=('mean_price_sem1', 'mean'),
        avg_sem3=('mean_price_sem3', 'mean'),
        avg_diff=('diff_pct', 'mean'),
        count=('month', 'count')
    ).reset_index()
    
    resumen = resumen[resumen['destination_name'].isin(top_dests)]
    resumen['avg_diff'] = resumen['avg_diff'].round(1)
    
    print(f"\nPromedio de diferencia Sem 3 vs Sem 1 por destino:")
    print(resumen[['destination_name', 'avg_sem1', 'avg_sem3', 'avg_diff', 'count']].to_string(index=False))
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Gráfico 1: Diferencia porcentual por destino
    colors = ['#ff6b6b' if x > 0 else '#4ecdc4' for x in resumen['avg_diff']]
    axes[0].barh(resumen['destination_name'], resumen['avg_diff'], color=colors)
    axes[0].axvline(x=0, color='black', linewidth=0.5)
    axes[0].set_title('Diferencia Promedio: Semana 3 vs Semana 1', fontsize=14)
    axes[0].set_xlabel('Diferencia Porcentual (%)')
    axes[0].set_ylabel('Destino')
    
    for i, (name, diff) in enumerate(zip(resumen['destination_name'], resumen['avg_diff'])):
        axes[0].text(diff + 0.5 if diff > 0 else diff - 0.5, i, 
                    f'{diff:+.1f}%', ha='left' if diff > 0 else 'right', va='center', fontsize=9)
    
    # Gráfico 2: Líneas comparativas para un destino ejemplo
    dest_ejemplo = 'Orlando' if 'Orlando' in top_dests else top_dests[0]
    dest_comparison = comparison[comparison['destination_name'] == dest_ejemplo].sort_values('month')
    
    months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
              'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    x = range(1, len(dest_comparison) + 1)
    axes[1].plot(x, dest_comparison['mean_price_sem1'].values, 'o-', 
                label='Semana 1', linewidth=2, markersize=8, color='blue')
    axes[1].plot(x, dest_comparison['mean_price_sem3'].values, 's-', 
                label='Semana 3', linewidth=2, markersize=8, color='red')
    
    # Rellenar diferencia
    axes[1].fill_between(x, dest_comparison['mean_price_sem1'].values, 
                         dest_comparison['mean_price_sem3'].values, alpha=0.2)
    
    axes[1].set_title(f'Comparación Semana 1 vs Semana 3: {dest_ejemplo}', fontsize=14)
    axes[1].set_xlabel('Mes')
    axes[1].set_ylabel('Precio Estandarizado ($)')
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(months, rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '16_patrones_semanales.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráfico guardado en: {output_path}")
    
    # Guardar datos
    csv_path = OUTPUT_DIR / 'data' / '16_patrones_semanales.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(csv_path, index=False)
    print(f"Datos guardados en: {csv_path}")
    
    conn.close()
    return comparison


def heatmap_semanal_por_destino():
    """Heatmap de precio promedio por destino y semana del mes."""
    
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
        week_in_month,
        AVG(price_std) as mean_price,
        COUNT(*) as count_obs
    FROM with_destination
    WHERE destination_name IS NOT NULL
    GROUP BY destination_name, week_in_month
    """
    
    print("\nGenerando heatmap de patrones semanales...")
    start = time.time()
    
    df = pd.read_sql_query(query, conn)
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    
    # Top 15 destinos
    top_dests = df.groupby('destination_name')['count_obs'].sum().nlargest(15).index
    df_top = df[df['destination_name'].isin(top_dests)]
    
    # Pivot
    pivot = df_top.pivot_table(index='destination_name', columns='week_in_month', values='mean_price')
    pivot.columns = ['Sem 1', 'Sem 2', 'Sem 3', 'Sem 4']
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Precio Promedio ($)'})
    
    ax.set_title('Precio Promedio por Destino y Semana del Mes', fontsize=14)
    ax.set_xlabel('Semana del Mes', fontsize=12)
    ax.set_ylabel('Destino', fontsize=12)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '16_heatmap_semanal_por_destino.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap guardado en: {output_path}")
    
    conn.close()
    return df_top


if __name__ == "__main__":
    print("="*60)
    print("PATRONES SEMANALES: SEMANA 1 VS SEMANA 3")
    print("="*60)
    
    # Análisis de patrones
    df_comparison = analizar_patrones_semanales()
    
    # Heatmap semanal
    df_heatmap = heatmap_semanal_por_destino()
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
