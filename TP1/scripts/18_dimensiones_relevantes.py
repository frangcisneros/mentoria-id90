# 18_dimensiones_relevantes.py
# Análisis de dimensiones relevantes: volumen, categoría proxy, oferta/demanda

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


def analizar_descuentos_volumen():
    """¿Hay descuentos por volumen? Compara precio/noche vs noches."""
    
    conn = sqlite3.connect(str(DB_PATH))
    
    query = """
    WITH valid_data AS (
        SELECT *
        FROM raw_data
        WHERE nights > 0 
            AND number_of_rooms > 0 
            AND (number_of_adults + number_of_kids) > 0
    )
    SELECT 
        nights,
        AVG(avg_price_average) as mean_total_price,
        AVG(avg_price_average / nights) as mean_price_per_night,
        AVG(avg_price_average / (nights * number_of_rooms * MAX(number_of_adults + number_of_kids, 1))) as mean_price_std,
        COUNT(*) as count_obs
    FROM valid_data
    GROUP BY nights
    HAVING COUNT(*) >= 100
    ORDER BY nights
    LIMIT 21
    """
    
    print("Analizando descuentos por volumen (noches vs precio)...")
    start = time.time()
    
    df = pd.read_sql_query(query, conn)
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    
    print(f"\n{'Noches':<8} {'Precio Total':>13} {'Precio/Noche':>14} {'Precio Std':>12} {'Obs':>10}")
    print("-" * 60)
    
    for _, row in df.iterrows():
        print(f"{int(row['nights']):<8} ${row['mean_total_price']:>11.2f} ${row['mean_price_per_night']:>12.2f} "
              f"${row['mean_price_std']:>10.2f} {row['count_obs']:>10,}")
    
    # Calcular descuento relativo a 1 noche
    price_1night = df[df['nights'] == 1]['mean_price_per_night'].values[0]
    df['discount_vs_1night'] = (1 - df['mean_price_per_night'] / price_1night) * 100
    
    print(f"\nDescuento relativo a 1 noche:")
    for _, row in df.iterrows():
        print(f"  {int(row['nights']):2d} noches: {row['discount_vs_1night']:+.1f}%")
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Precio por noche vs noches
    axes[0].plot(df['nights'], df['mean_price_per_night'], 'o-', linewidth=2, markersize=8)
    axes[0].set_title('Precio por Noche vs Duración de Estadía', fontsize=14)
    axes[0].set_xlabel('Noches')
    axes[0].set_ylabel('Precio por Noche ($)')
    axes[0].grid(True, alpha=0.3)
    
    # Línea de tendencia
    z = np.polyfit(df['nights'], df['mean_price_per_night'], 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(1, 20, 100)
    axes[0].plot(x_smooth, p(x_smooth), 'r--', alpha=0.7, label='Tendencia cuadrática')
    axes[0].legend()
    
    # Descuento acumulado
    axes[1].bar(df['nights'], df['discount_vs_1night'], color='steelblue')
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_title('Descuento Acumulado vs 1 Noche', fontsize=14)
    axes[1].set_xlabel('Noches')
    axes[1].set_ylabel('Descuento (%)')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '18_descuentos_volumen.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráfico guardado en: {output_path}")
    
    conn.close()
    return df


def analizar_categoria_proxy():
    """Usa precio como proxy de categoría del hotel."""
    
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
            WHEN price_std <= 20 THEN '1_Budget (≤$20)'
            WHEN price_std <= 40 THEN '2_Mid-Low ($20-40)'
            WHEN price_std <= 60 THEN '3_Mid ($40-60)'
            WHEN price_std <= 100 THEN '4_Mid-High ($60-100)'
            WHEN price_std <= 150 THEN '5_High ($100-150)'
            ELSE '6_Premium (>$150)'
        END as category_proxy,
        AVG(avg_hotel_count) as mean_supply,
        AVG(count_repeated) as mean_demand,
        AVG(price_std) as mean_price,
        AVG(nights) as mean_nights,
        COUNT(*) as count_obs
    FROM standardized
    GROUP BY category_proxy
    ORDER BY category_proxy
    """
    
    print("\nAnálisis de categoría proxy (basado en precio)...")
    start = time.time()
    
    df = pd.read_sql_query(query, conn)
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    
    print(f"\n{'Categoría':<25} {'Precio':>10} {'Oferta':>10} {'Demanda':>10} {'Noches':>10} {'Obs':>12}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        print(f"{row['category_proxy']:<25} ${row['mean_price']:>8.2f} {row['mean_supply']:>10.1f} "
              f"{row['mean_demand']:>10.1f} {row['mean_nights']:>10.1f} {row['count_obs']:>12,}")
    
    # Visualización
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    categories = df['category_proxy'].values
    x = range(len(categories))
    
    # Oferta por categoría
    axes[0].bar(x, df['mean_supply'], color='coral')
    axes[0].set_title('Oferta (Hoteles Disponibles) por Categoría', fontsize=12)
    axes[0].set_xlabel('Categoría de Precio')
    axes[0].set_ylabel('Hoteles Promedio')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c.split('_')[0] for c in categories], rotation=45)
    
    # Demanda por categoría
    axes[1].bar(x, df['mean_demand'], color='steelblue')
    axes[1].set_title('Demanda (Veces Repetido) por Categoría', fontsize=12)
    axes[1].set_xlabel('Categoría de Precio')
    axes[1].set_ylabel('count_repeated Promedio')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c.split('_')[0] for c in categories], rotation=45)
    
    # Distribución de observaciones
    axes[2].pie(df['count_obs'], labels=[c.split('_')[0] for c in categories], 
                autopct='%1.1f%%', startangle=90)
    axes[2].set_title('Distribución de Observaciones por Categoría', fontsize=12)
    
    plt.suptitle('Análisis de Categoría Proxy (Basado en Precio)', fontsize=14, y=1.02)
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '18_categoria_proxy.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráfico guardado en: {output_path}")
    
    conn.close()
    return df


def analizar_oferta_demanda_precio():
    """Relación entre oferta, demanda y precio."""
    
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
            WHEN avg_hotel_count <= 10 THEN '1_Baja Oferta'
            WHEN avg_hotel_count <= 50 THEN '2_Media Oferta'
            ELSE '3_Alta Oferta'
        END as supply_bucket,
        CASE 
            WHEN count_repeated <= 2 THEN '1_Baja Demanda'
            WHEN count_repeated <= 5 THEN '2_Media Demanda'
            ELSE '3_Alta Demanda'
        END as demand_bucket,
        AVG(price_std) as mean_price,
        COUNT(*) as count_obs
    FROM standardized
    GROUP BY supply_bucket, demand_bucket
    ORDER BY supply_bucket, demand_bucket
    """
    
    print("\nAnálisis de oferta vs demanda vs precio...")
    start = time.time()
    
    df = pd.read_sql_query(query, conn)
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    
    print(f"\n{'Oferta':<18} {'Demanda':<18} {'Precio Std':>12} {'Obs':>12}")
    print("-" * 62)
    
    for _, row in df.iterrows():
        print(f"{row['supply_bucket']:<18} {row['demand_bucket']:<18} ${row['mean_price']:>10.2f} {row['count_obs']:>12,}")
    
    # Pivot para heatmap
    pivot = df.pivot_table(index='supply_bucket', columns='demand_bucket', values='mean_price')
    
    # Visualización
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Precio Promedio ($)'})
    
    ax.set_title('Precio Promedio por Oferta y Demanda', fontsize=14)
    ax.set_xlabel('Demanda (count_repeated)', fontsize=12)
    ax.set_ylabel('Oferta (avg_hotel_count)', fontsize=12)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '18_heatmap_oferta_demanda_precio.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráfico guardado en: {output_path}")
    
    conn.close()
    return df


def analisis_temporal_completo():
    """Análisis temporal completo: mes × semana × día."""
    
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
            CAST(strftime('%w', date_start) AS INTEGER) as day_of_week,
            CASE 
                WHEN CAST(strftime('%d', date_start) AS INTEGER) <= 7 THEN 1
                WHEN CAST(strftime('%d', date_start) AS INTEGER) <= 15 THEN 2
                WHEN CAST(strftime('%d', date_start) AS INTEGER) <= 22 THEN 3
                ELSE 4
            END as week_in_month
        FROM valid_data
    )
    SELECT 
        month,
        week_in_month,
        day_of_week,
        AVG(price_std) as mean_price,
        COUNT(*) as count_obs
    FROM standardized
    GROUP BY month, week_in_month, day_of_week
    """
    
    print("\nAnálisis temporal completo (mes × semana × día)...")
    start = time.time()
    
    df = pd.read_sql_query(query, conn)
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    
    months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
              'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    days = ['Dom', 'Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb']
    weeks = ['Sem 1', 'Sem 2', 'Sem 3', 'Sem 4']
    
    # Heatmap: Mes × Día de semana
    pivot_md = df.groupby(['month', 'day_of_week'])['mean_price'].mean().unstack()
    pivot_md.columns = days
    
    # Heatmap: Mes × Semana del mes
    pivot_mw = df.groupby(['month', 'week_in_month'])['mean_price'].mean().unstack()
    pivot_mw.columns = weeks
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    sns.heatmap(pivot_md, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0],
                xticklabels=days, yticklabels=months)
    axes[0].set_title('Precio: Mes × Día de Semana', fontsize=14)
    axes[0].set_xlabel('Día de Semana')
    axes[0].set_ylabel('Mes')
    
    sns.heatmap(pivot_mw, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1],
                xticklabels=weeks, yticklabels=months)
    axes[1].set_title('Precio: Mes × Semana del Mes', fontsize=14)
    axes[1].set_xlabel('Semana del Mes')
    axes[1].set_ylabel('Mes')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '18_analisis_temporal_completo.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráfico guardado en: {output_path}")
    
    conn.close()
    return df


def resumen_dimensiones():
    """Resumen de qué dimensiones son relevantes."""
    
    print("\n" + "="*70)
    print("RESUMEN: DIMENSIONES RELEVANTES PARA EL ANÁLISIS")
    print("="*70)
    
    print("""
1. GEOGRAFÍA (destination_final)
   - Es el ANCLA principal (como dice el profesor)
   - Los precios se miden respecto a ESE lugar
   - Necesario: ~30 obs por contexto para baseline confiable

2. TEMPORALIDAD
   - Mes: Captura estacionalidad (verano vs invierno)
   - Semana del mes: Captura patrones de pago (quincena)
   - Día de semana: Sábado más caro, domingo más barato
   - IMPORTANTE: Combinar mes + semana es mejor que solo mes

3. DURACIÓN DE ESTADÍA (nights)
   - DESCUBRIMIENTO: Hay descuentos por volumen
   - Precio/noche baja significativamente con estancias más largas
   - RECOMENDACIÓN: Incluir como dimensión o normalizar mejor

4. OFERTA Y DEMANDA
   - Oferta (avg_hotel_count): A más hoteles, menor precio
   - Demanda (count_repeated): Proxy de popularidad
   - AMBOS son relevantes para explicar variación de precios

5. CATEGORÍA DEL HOTEL
   - NO tenemos columna directa (estrellas, etc.)
   - Podemos usar precio como proxy de categoría
   - O agrupar por rangos de precio

COMBINACIÓN ÓPTIMA (según análisis de homogeneidad):
   destination_final + month + week_in_month + stay_duration
   
   Esto produce grupos donde:
   - Los precios son mutuamente informativos
   - La oferta/demanda es relativamente homogénea
   - Se pueden calcular baselines confiables
""")
    
    return None


if __name__ == "__main__":
    print("="*70)
    print("ANÁLISIS DE DIMENSIONES RELEVANTES")
    print("="*70)
    
    # 1. Descuentos por volumen
    df_volumen = analizar_descuentos_volumen()
    
    # 2. Categoría proxy
    df_categoria = analizar_categoria_proxy()
    
    # 3. Oferta vs demanda
    df_od = analizar_oferta_demanda_precio()
    
    # 4. Análisis temporal completo
    df_temporal = analisis_temporal_completo()
    
    # 5. Resumen
    resumen_dimensiones()
    
    print("\n" + "="*70)
    print("ANÁLISIS COMPLETADO")
    print("="*70)
