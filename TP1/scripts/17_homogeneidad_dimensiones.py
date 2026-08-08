# 17_homogeneidad_dimensiones.py
# ¿Qué combinación de dimensiones produce grupos homogéneos?

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
from itertools import combinations

DB_PATH = Path(config.BASE_DIR) / 'data' / 'hotel_data.db'
OUTPUT_DIR = Path(config.BASE_DIR) / 'TP1' / 'outputs'


def prepare_base_data():
    """Prepara los datos base con todas las dimensiones calculadas."""
    
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
            CAST(strftime('%w', date_start) AS INTEGER) as day_of_week,
            CAST(strftime('%m', date_start) AS INTEGER) as month,
            CASE 
                WHEN CAST(strftime('%d', date_start) AS INTEGER) <= 7 THEN 1
                WHEN CAST(strftime('%d', date_start) AS INTEGER) <= 15 THEN 2
                WHEN CAST(strftime('%d', date_start) AS INTEGER) <= 22 THEN 3
                ELSE 4
            END as week_in_month,
            CASE 
                WHEN nights <= 2 THEN 'corta'
                WHEN nights <= 5 THEN 'media'
                ELSE 'larga'
            END as stay_duration,
            CASE 
                WHEN avg_hotel_count <= 10 THEN 'baja_oferta'
                WHEN avg_hotel_count <= 50 THEN 'media_oferta'
                ELSE 'alta_oferta'
            END as oferta_bucket,
            CASE 
                WHEN count_repeated <= 2 THEN 'baja_demanda'
                WHEN count_repeated <= 5 THEN 'media_demanda'
                ELSE 'alta_demanda'
            END as demanda_bucket
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
        day_of_week,
        month,
        week_in_month,
        stay_duration,
        oferta_bucket,
        demanda_bucket,
        price_std,
        avg_price_average,
        count_repeated,
        avg_hotel_count,
        nights
    FROM with_destination
    WHERE destination_final IS NOT NULL
    LIMIT 500000
    """
    
    print("Preparando datos base...")
    start = time.time()
    
    df = pd.read_sql_query(query, conn)
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    print(f"Registros: {len(df):,}")
    
    conn.close()
    return df


def calcular_homogeneidad(df, group_cols):
    """
    Calcula métricas de homogeneidad para un grupo de dimensiones.
    
    Returns:
        dict con métricas de homogeneidad
    """
    if len(group_cols) == 0:
        return None
    
    try:
        grouped = df.groupby(group_cols)
        
        # Estadísticas por grupo
        stats = grouped.agg({
            'price_std': ['mean', 'std', 'count'],
            'avg_hotel_count': ['mean', 'std'],
            'count_repeated': ['mean', 'std']
        }).reset_index()
        
        # Aplanar columnas
        stats.columns = group_cols + [
            'price_mean', 'price_std', 'price_count',
            'hotel_mean', 'hotel_std',
            'demand_mean', 'demand_std'
        ]
        
        # Filtrar grupos con suficientes datos
        stats = stats[stats['price_count'] >= 10]
        
        if len(stats) == 0:
            return None
        
        # Métricas de homogeneidad
        cv_price = (stats['price_std'] / stats['price_mean']).mean()  # Coeficiente de variación
        cv_hotel = (stats['hotel_std'] / stats['hotel_mean'].replace(0, 1)).mean()
        cv_demand = (stats['demand_std'] / stats['demand_mean'].replace(0, 1)).mean()
        
        # Variabilidad BETWEEN groups (qué tan diferentes son los grupos entre sí)
        between_var_price = stats['price_mean'].var()
        
        # Variabilidad WITHIN groups (qué tan homogéneos son los grupos)
        within_var_price = (stats['price_std'] ** 2).mean()
        
        # Ratio signal-to-noise
        signal_noise = between_var_price / (within_var_price + 0.001)
        
        return {
            'dimensions': ', '.join(group_cols),
            'n_groups': len(stats),
            'avg_obs_per_group': stats['price_count'].mean(),
            'cv_price': cv_price,
            'cv_hotel': cv_hotel,
            'cv_demand': cv_demand,
            'signal_noise_ratio': signal_noise,
            'between_var': between_var_price,
            'within_var': within_var_price
        }
        
    except Exception as e:
        print(f"Error con {group_cols}: {e}")
        return None


def explorar_combinaciones(df):
    """Explora todas las combinaciones posibles de dimensiones."""
    
    print("\n" + "="*70)
    print("EXPLORACIÓN DE COMBINACIONES DE DIMENSIONES")
    print("="*70)
    
    # Dimensiones candidatas
    dimensions = [
        'destination_final',
        'month',
        'week_in_month',
        'day_of_week',
        'stay_duration',
        'oferta_bucket',
        'demanda_bucket'
    ]
    
    results = []
    
    # Probar combinaciones de 1 a 4 dimensiones
    for r in range(1, 5):
        print(f"\nProbando combinaciones de {r} dimensión(es)...")
        
        for combo in combinations(dimensions, r):
            result = calcular_homogeneidad(df, list(combo))
            if result:
                results.append(result)
    
    # Convertir a DataFrame
    df_results = pd.DataFrame(results)
    
    # Ordenar por signal-to-noise ratio (mejor métrica de homogeneidad)
    df_results = df_results.sort_values('signal_noise_ratio', ascending=False)
    
    print("\n" + "="*70)
    print("TOP 15 COMBINACIONES POR HOMOGENEIDAD")
    print("="*70)
    
    print(f"\n{'#':<4} {'Dimensiones':<55} {'Grupos':>8} {'Obs/Grupo':>10} {'CV Precio':>10} {'S/N Ratio':>10}")
    print("-" * 100)
    
    for i, (_, row) in enumerate(df_results.head(15).iterrows()):
        print(f"{i+1:<4} {row['dimensions'][:54]:<55} {row['n_groups']:>8} "
              f"{row['avg_obs_per_group']:>10.0f} {row['cv_price']:>10.3f} {row['signal_noise_ratio']:>10.3f}")
    
    return df_results


def visualizar_resultados(df_results):
    """Visualiza los resultados de la exploración."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top 10 por Signal-to-Noise Ratio
    top10 = df_results.head(10)
    axes[0, 0].barh(range(len(top10)), top10['signal_noise_ratio'], color='steelblue')
    axes[0, 0].set_yticks(range(len(top10)))
    axes[0, 0].set_yticklabels(top10['dimensions'].str[:40], fontsize=9)
    axes[0, 0].set_xlabel('Signal-to-Noise Ratio')
    axes[0, 0].set_title('Top 10 Combinaciones por S/N Ratio', fontsize=12)
    axes[0, 0].invert_yaxis()
    
    # 2. CV Precio vs Número de Grupos
    axes[0, 1].scatter(df_results['n_groups'], df_results['cv_price'], 
                       c=df_results['signal_noise_ratio'], cmap='viridis', alpha=0.6)
    axes[0, 1].set_xlabel('Número de Grupos')
    axes[0, 1].set_ylabel('CV de Precio (menor = más homogéneo)')
    axes[0, 1].set_title('CV Precio vs Número de Grupos', fontsize=12)
    axes[0, 1].set_xscale('log')
    plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1], label='S/N Ratio')
    
    # 3. CV Oferta vs CV Demanda
    scatter = axes[1, 0].scatter(df_results['cv_hotel'], df_results['cv_demand'],
                                 c=df_results['signal_noise_ratio'], cmap='viridis', alpha=0.6)
    axes[1, 0].set_xlabel('CV de Oferta (avg_hotel_count)')
    axes[1, 0].set_ylabel('CV de Demanda (count_repeated)')
    axes[1, 0].set_title('Homogeneidad de Oferta vs Demanda', fontsize=12)
    plt.colorbar(scatter, ax=axes[1, 0], label='S/N Ratio')
    
    # 4. Distribución de métricas por tipo de dimensión
    df_results['has_geo'] = df_results['dimensions'].str.contains('destination_final')
    df_results['has_time'] = df_results['dimensions'].str.contains('month|week|day')
    df_results['hasStay'] = df_results['dimensions'].str.contains('stay')
    df_results['hasSupplyDemand'] = df_results['dimensions'].str.contains('oferta|demanda')
    
    categories = []
    for _, row in df_results.iterrows():
        cats = []
        if row['has_geo']:
            cats.append('Geografía')
        if row['has_time']:
            cats.append('Tiempo')
        if row['hasStay']:
            cats.append('Duración')
        if row['hasSupplyDemand']:
            cats.append('Oferta/Demanda')
        categories.append(' + '.join(cats) if cats else 'Otras')
    
    df_results['category'] = categories
    
    cat_stats = df_results.groupby('category')['signal_noise_ratio'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    
    axes[1, 1].barh(range(len(cat_stats)), cat_stats['mean'], color='coral')
    axes[1, 1].set_yticks(range(len(cat_stats)))
    axes[1, 1].set_yticklabels(cat_stats.index, fontsize=10)
    axes[1, 1].set_xlabel('S/N Ratio Promedio')
    axes[1, 1].set_title('S/N Ratio Promedio por Categoría de Dimensión', fontsize=12)
    axes[1, 1].invert_yaxis()
    
    # Agregar conteos
    for i, (cat, row) in enumerate(cat_stats.iterrows()):
        axes[1, 1].text(row['mean'] + 0.01, i, f'n={int(row["count"])}', va='center', fontsize=9)
    
    plt.suptitle('Análisis de Homogeneidad por Combinación de Dimensiones', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'images' / '17_homogeneidad_dimensiones.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráfico guardado en: {output_path}")


def analisis_profundidad():
    """Análisis de profundidad: ¿a qué escala geográfica es más homogéneo?"""
    
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
    with_mapping AS (
        SELECT 
            s.*,
            s.country_code as geo_country,
            s.country_code || ' - ' || s.state as geo_region,
            s.country_code || ' - ' || s.state || ' - ' || s.city as geo_city,
            COALESCE(
                m3.nearest_destination_id,
                m2.nearest_destination_id,
                m1.nearest_destination_id,
                s.city
            ) as destination_final
        FROM standardized s
        LEFT JOIN mapping m3 ON UPPER(s.country_code) || ' - ' || UPPER(s.state) || ' - ' || s.city = m3.reference
        LEFT JOIN mapping m2 ON UPPER(s.country_code) || ' - ' || s.city = m2.reference
        LEFT JOIN mapping m1 ON UPPER(s.country_code) || ' - ' || UPPER(s.state) = m1.reference
    )
    SELECT 
        geo_country,
        geo_region,
        geo_city,
        destination_final,
        month,
        price_std,
        avg_hotel_count,
        count_repeated
    FROM with_mapping
    """
    
    print("\nAnalizando homogeneidad por escala geográfica...")
    start = time.time()
    
    df = pd.read_sql_query(query, conn)
    
    elapsed = time.time() - start
    print(f"Tiempo: {elapsed:.1f}s")
    
    # Calcular CV para cada escala
    scales = ['geo_country', 'geo_region', 'geo_city', 'destination_final']
    
    results = []
    
    for scale in scales:
        # Agrupar por la dimensión + mes
        grouped = df.groupby([scale, 'month']).agg({
            'price_std': ['mean', 'std', 'count'],
            'avg_hotel_count': ['mean', 'std'],
            'count_repeated': ['mean', 'std']
        }).reset_index()
        
        grouped.columns = [scale, 'month', 'price_mean', 'price_std', 'count',
                          'hotel_mean', 'hotel_std', 'demand_mean', 'demand_std']
        
        # Filtrar grupos con suficientes datos
        grouped = grouped[grouped['count'] >= 10]
        
        if len(grouped) > 0:
            cv_price = (grouped['price_std'] / grouped['price_mean'].replace(0, 1)).mean()
            cv_hotel = (grouped['hotel_std'] / grouped['hotel_mean'].replace(0, 1)).mean()
            cv_demand = (grouped['demand_std'] / grouped['demand_mean'].replace(0, 1)).mean()
            
            results.append({
                'scale': scale,
                'n_groups': len(grouped),
                'cv_price': cv_price,
                'cv_hotel': cv_hotel,
                'cv_demand': cv_demand
            })
    
    df_results = pd.DataFrame(results)
    
    print("\nHomogeneidad por escala geográfica:")
    print(f"\n{'Escala':<25} {'Grupos':>8} {'CV Precio':>10} {'CV Oferta':>10} {'CV Demanda':>10}")
    print("-" * 65)
    
    for _, row in df_results.iterrows():
        print(f"{row['scale']:<25} {row['n_groups']:>8} {row['cv_price']:>10.3f} "
              f"{row['cv_hotel']:>10.3f} {row['cv_demand']:>10.3f}")
    
    # Visualización
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(df_results))
    width = 0.25
    
    ax.bar([i - width for i in x], df_results['cv_price'], width, label='CV Precio', color='steelblue')
    ax.bar(x, df_results['cv_hotel'], width, label='CV Oferta', color='coral')
    ax.bar([i + width for i in x], df_results['cv_demand'], width, label='CV Demanda', color='green')
    
    ax.set_xlabel('Escala Geográfica')
    ax.set_ylabel('Coeficiente de Variación (menor = más homogéneo)')
    ax.set_title('Homogeneidad por Escala Geográfica', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df_results['scale'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'images' / '17_homogeneidad_escala_geografica.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráfico guardado en: {output_path}")
    
    conn.close()
    return df_results


if __name__ == "__main__":
    print("="*70)
    print("ANÁLISIS DE HOMOGENEIDAD POR COMBINACIÓN DE DIMENSIONES")
    print("="*70)
    
    # Preparar datos
    df = prepare_base_data()
    
    # Explorar combinaciones
    df_results = explorar_combinaciones(df)
    
    # Visualizar
    visualizar_resultados(df_results)
    
    # Análisis de profundidad geográfica
    df_geo = analisis_profundidad()
    
    # Guardar resultados
    csv_path = OUTPUT_DIR / 'data' / '17_homogeneidad_dimensiones.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(csv_path, index=False)
    print(f"\nResultados guardados en: {csv_path}")
    
    print("\n" + "="*70)
    print("ANÁLISIS COMPLETADO")
    print("="*70)
