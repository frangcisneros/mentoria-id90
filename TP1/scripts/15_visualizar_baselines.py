# 15_visualizar_baselines.py
# Visualización gráfica de baselines por destino

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


def visualizar_baselines_top_destinos(top_n=10):
    """Gráfico de líneas: baseline (precio promedio) por mes para top destinos."""
    
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
    
    print(f"Visualizando baselines para top {top_n} destinos...")
    start = time.time()
    
    df = pd.read_sql_query(query, conn)
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    
    # Top destinos por volumen
    top_dests = df.groupby('destination_name')['count_obs'].sum().nlargest(top_n).index
    
    months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
              'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    # === GRÁFICO 1: Líneas por destino ===
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = sns.color_palette("tab10", len(top_dests))
    
    for i, dest in enumerate(top_dests):
        dest_data = df[df['destination_name'] == dest].sort_values('month')
        ax.plot(range(1, 13), dest_data['mean_price'].values, 'o-', 
                label=dest, linewidth=2, markersize=6, color=colors[i])
    
    ax.set_title('Baseline: Precio Promedio por Destino y Mes', fontsize=16)
    ax.set_xlabel('Mes', fontsize=12)
    ax.set_ylabel('Precio Estandarizado ($/habitación-noche-persona)', fontsize=12)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(months)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '15_baselines_lineas_top_destinos.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico de líneas guardado en: {output_path}")
    
    # === GRÁFICO 2: Subplots individuales por destino ===
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, dest in enumerate(top_dests):
        if i >= 10:
            break
        
        dest_data = df[df['destination_name'] == dest].sort_values('month')
        
        axes[i].fill_between(range(1, 13), dest_data['mean_price'].values, 
                            alpha=0.3, color=colors[i])
        axes[i].plot(range(1, 13), dest_data['mean_price'].values, 'o-', 
                    linewidth=2, markersize=5, color=colors[i])
        
        axes[i].set_title(dest, fontsize=12, fontweight='bold')
        axes[i].set_xticks(range(1, 13))
        axes[i].set_xticklabels([m[:3] for m in months], fontsize=8)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylabel('$')
    
    plt.suptitle('Baselines Individuales por Destino', fontsize=16, y=1.02)
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '15_baselines_subplots.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Subplots guardados en: {output_path}")
    
    # === GRÁFICO 3: Heatmap de baselines ===
    pivot = df[df['destination_name'].isin(top_dests)].pivot_table(
        index='destination_name', columns='month', values='mean_price'
    )
    pivot.columns = months
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Precio Promedio ($)'})
    
    ax.set_title('Heatmap de Baselines: Precio por Destino y Mes', fontsize=14)
    ax.set_xlabel('Mes', fontsize=12)
    ax.set_ylabel('Destino', fontsize=12)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '15_baselines_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap guardado en: {output_path}")
    
    conn.close()
    return df


def visualizar_baselines_con_std(top_n=5):
    """Gráfico con banda de ±1 desviación estándar."""
    
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
        SQRT(MAX(0, AVG(price_std * price_std) - AVG(price_std) * AVG(price_std))) as std_price,
        COUNT(*) as count_obs
    FROM with_destination
    WHERE destination_name IS NOT NULL
    GROUP BY destination_name, month
    """
    
    print(f"\nVisualizando baselines con ±1 STD para top {top_n} destinos...")
    start = time.time()
    
    df = pd.read_sql_query(query, conn)
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    
    # Top destinos
    top_dests = df.groupby('destination_name')['count_obs'].sum().nlargest(top_n).index
    
    months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
              'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    fig, axes = plt.subplots(1, top_n, figsize=(20, 6), sharey=False)
    if top_n == 1:
        axes = [axes]
    
    colors = sns.color_palette("tab10", top_n)
    
    for i, dest in enumerate(top_dests):
        dest_data = df[df['destination_name'] == dest].sort_values('month')
        
        x = range(1, 13)
        mean = dest_data['mean_price'].values
        std = dest_data['std_price'].values
        
        # Banda de ±1 STD
        axes[i].fill_between(x, mean - std, mean + std, alpha=0.3, color=colors[i], label='±1 STD')
        axes[i].plot(x, mean, 'o-', linewidth=2, markersize=6, color=colors[i], label='Media')
        
        # Línea de ±1 STD
        axes[i].plot(x, mean - std, '--', linewidth=1, alpha=0.5, color=colors[i])
        axes[i].plot(x, mean + std, '--', linewidth=1, alpha=0.5, color=colors[i])
        
        axes[i].set_title(dest, fontsize=12, fontweight='bold')
        axes[i].set_xticks(range(1, 13))
        axes[i].set_xticklabels([m[:3] for m in months], fontsize=8, rotation=45)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylabel('Precio Std ($)')
        axes[i].legend(fontsize=8)
    
    plt.suptitle('Baselines con Banda de ±1 Desviación Estándar', fontsize=14, y=1.05)
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '15_baselines_con_std.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico con STD guardado en: {output_path}")
    
    conn.close()
    return df


def resumen_estadistico_baselines():
    """Resumen estadístico de todas las baselines."""
    
    conn = sqlite3.connect(str(DB_PATH))
    
    # Cargar baselines calculados
    baselines_path = Path(config.BASE_DIR) / 'outputs' / 'baselines.csv'
    if baselines_path.exists():
        baselines = pd.read_csv(baselines_path)
    else:
        print("Calculando baselines...")
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
            AVG(price_std) as mean_price_std,
            SQRT(MAX(0, AVG(price_std * price_std) - AVG(price_std) * AVG(price_std))) as std_price_std,
            MIN(price_std) as min_price_std,
            MAX(price_std) as max_price_std,
            COUNT(*) as count_obs,
            SUM(count_repeated) as total_repeated
        FROM with_destination
        WHERE destination_final IS NOT NULL
        GROUP BY destination_final, destination_name, month, week_in_month
        """
        
        baselines = pd.read_sql_query(query, conn)
        baselines.to_csv(baselines_path, index=False)
    
    print("\n" + "="*60)
    print("RESUMEN ESTADÍSTICO DE BASELINES")
    print("="*60)
    
    print(f"\nTotal de baselines: {len(baselines):,}")
    print(f"Destinos únicos: {baselines['destination_final'].nunique():,}")
    print(f"Observaciones totales: {baselines['count_obs'].sum():,}")
    
    print(f"\nEstadísticas de precio promedio:")
    print(f"  Mínimo: ${baselines['mean_price_std'].min():.2f}")
    print(f"  Máximo: ${baselines['mean_price_std'].max():.2f}")
    print(f"  Promedio: ${baselines['mean_price_std'].mean():.2f}")
    print(f"  Mediana: ${baselines['mean_price_std'].median():.2f}")
    
    print(f"\nEstadísticas de observaciones por baseline:")
    print(f"  Mínimo: {baselines['count_obs'].min():.0f}")
    print(f"  Máximo: {baselines['count_obs'].max():.0f}")
    print(f"  Promedio: {baselines['count_obs'].mean():.0f}")
    
    # Top 10 baselines con más observaciones
    print(f"\nTop 10 baselines con más observaciones:")
    top10 = baselines.nlargest(10, 'count_obs')[['destination_name', 'month', 'week_in_month', 'mean_price_std', 'count_obs']]
    print(top10.to_string(index=False))
    
    conn.close()
    return baselines


if __name__ == "__main__":
    print("="*60)
    print("VISUALIZACIÓN DE BASELINES")
    print("="*60)
    
    # Resumen estadístico
    baselines = resumen_estadistico_baselines()
    
    # Visualización de top destinos
    df_lineas = visualizar_baselines_top_destinos(top_n=10)
    
    # Visualización con STD
    df_std = visualizar_baselines_con_std(top_n=5)
    
    print("\n" + "="*60)
    print("VISUALIZACIÓN COMPLETADA")
    print("="*60)
