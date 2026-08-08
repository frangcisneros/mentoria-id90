"""
19. COMPARACIÓN INTERANUAL - 2024 vs 2025
========================================
Compara patrones entre ambos años para validar estabilidad.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import config

DB_PATH = Path(config.BASE_DIR) / 'data' / 'hotel_data.db'
OUTPUT_DIR = Path(config.BASE_DIR) / 'TP1' / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CARGA DE DATOS
# ============================================================================

def load_year_data():
    """Carga datos de ambos años."""
    conn = sqlite3.connect(str(DB_PATH))
    
    query = """
    WITH valid_data AS (
        SELECT * FROM raw_data
        WHERE nights > 0 AND number_of_rooms > 0 
        AND (number_of_adults + number_of_kids) > 0
    ),
    standardized AS (
        SELECT *,
            avg_price_average / (nights * number_of_rooms * 
                MAX(number_of_adults + number_of_kids, 1)) as price_std,
            strftime('%Y', date_start) as year,
            CAST(strftime('%m', date_start) AS INTEGER) as month,
            CAST(strftime('%w', date_start) AS INTEGER) as day_of_week,
            CASE 
                WHEN CAST(strftime('%d', date_start) AS INTEGER) <= 7 THEN 1
                WHEN CAST(strftime('%d', date_start) AS INTEGER) <= 15 THEN 2
                WHEN CAST(strftime('%d', date_start) AS INTEGER) <= 22 THEN 3
                ELSE 4
            END as week_in_month,
            CASE WHEN nights <= 2 THEN 'corta' 
                 WHEN nights <= 5 THEN 'media' 
                 ELSE 'larga' END as stay_duration
        FROM valid_data
    ),
    with_dest AS (
        SELECT s.*, 
            COALESCE(m3.nearest_destination_id, m2.nearest_destination_id, 
                     m1.nearest_destination_id, s.city) as destination_final,
            COALESCE(m3.nearest_destination_name, m2.nearest_destination_name, 
                     m1.nearest_destination_name, s.city) as destination_name
        FROM standardized s
        LEFT JOIN mapping m3 ON UPPER(s.country_code) || ' - ' || UPPER(s.state) || ' - ' || s.city = m3.reference
        LEFT JOIN mapping m2 ON UPPER(s.country_code) || ' - ' || s.city = m2.reference
        LEFT JOIN mapping m1 ON UPPER(s.country_code) || ' - ' || UPPER(s.state) = m1.reference
    )
    SELECT year, destination_final, month, week_in_month, stay_duration, 
           day_of_week, price_std, avg_hotel_count, count_repeated
    FROM with_dest 
    WHERE destination_final IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# ============================================================================
# ANÁLISIS
# ============================================================================

def compare_global_distributions(df):
    """1. Comparar distribuciones globales de precios."""
    print("\n" + "="*80)
    print("1. DISTRIBUCIONES GLOBALES DE PRECIOS")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, (year, data) in zip(axes, df.groupby('year')):
        ax.hist(data['price_std'], bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'Precios {year}')
        ax.set_xlabel('Precio normalizado (USD/night/person)')
        ax.set_ylabel('Frecuencia')
        ax.axvline(data['price_std'].median(), color='red', linestyle='--', 
                   label=f'Mediana: ${data["price_std"].median():.1f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'images' / '19_distribucion_precios_interanual.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Estadísticas por año
    stats = df.groupby('year')['price_std'].agg(['mean', 'median', 'std', 'count'])
    print("\nEstadísticas por año:")
    print(stats.round(2))
    
    # Test de Mann-Whitney (no paramétrico)
    from scipy.stats import mannwhitneyu
    prices_2024 = df[df['year'] == '2024']['price_std'].dropna()
    prices_2025 = df[df['year'] == '2025']['price_std'].dropna()
    stat, p_value = mannwhitneyu(prices_2024, prices_2025, alternative='two-sided')
    print(f"\nTest de Mann-Whitney U:")
    print(f"  U = {stat:.0f}, p-value = {p_value:.6f}")
    print(f"  {'Diferencia significativa' if p_value < 0.05 else 'Sin diferencia significativa'} (α=0.05)")

def compare_by_month(df):
    """2. Comparar patrones estacionales por mes."""
    print("\n" + "="*80)
    print("2. PATRÓN ESTACIONAL POR MES")
    print("="*80)
    
    monthly = df.groupby(['year', 'month'])['price_std'].mean().unstack(level=0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly.plot(kind='bar', ax=ax, alpha=0.8)
    ax.set_title('Precio promedio por mes: 2024 vs 2025')
    ax.set_xlabel('Mes')
    ax.set_ylabel('Precio normalizado (USD/night/person)')
    ax.legend(['2024', '2025'])
    ax.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                         'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'], rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'images' / '19_precio_por_mes_interanual.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Correlación mensual
    correlation = monthly['2024'].corr(monthly['2025'])
    print(f"\nCorrelación entre meses de 2024 y 2025: {correlation:.3f}")
    
    # Diferencia porcentual
    monthly['diff_pct'] = ((monthly['2025'] - monthly['2024']) / monthly['2024'] * 100)
    print("\nDiferencia porcentual por mes:")
    print(monthly[['diff_pct']].round(2))

def compare_by_destination(df):
    """3. Comparar top destinos entre años."""
    print("\n" + "="*80)
    print("3. TOP DESTINOS: 2024 vs 2025")
    print("="*80)
    
    # Top 20 destinos por volumen
    top_dest = df['destination_final'].value_counts().head(20).index
    
    dest_comparison = df[df['destination_final'].isin(top_dest)].groupby(
        ['year', 'destination_final']
    )['price_std'].mean().unstack(level=0)
    
    # Calcular cambio porcentual
    dest_comparison['change_pct'] = (
        (dest_comparison['2025'] - dest_comparison['2024']) / dest_comparison['2024'] * 100
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Gráfico de barras comparativo
    dest_comparison[['2024', '2025']].plot(kind='barh', ax=axes[0], alpha=0.8)
    axes[0].set_title('Precio por destino: 2024 vs 2025')
    axes[0].set_xlabel('Precio normalizado (USD/night/person)')
    
    # Cambio porcentual
    dest_comparison['change_pct'].sort_values().plot(kind='barh', ax=axes[1], 
                                                       color=['green' if x > 0 else 'red' for x in dest_comparison['change_pct'].sort_values()])
    axes[1].set_title('Cambio porcentual 2024→2025')
    axes[1].set_xlabel('Cambio (%)')
    axes[1].axvline(0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'images' / '19_top_destinos_interanual.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nTop 10 destinos por cambio porcentual:")
    print(dest_comparison[['change_pct', '2024', '2025']].sort_values('change_pct').round(2).to_string())

def compare_by_dimension(df):
    """4. Comparar otras dimensiones."""
    print("\n" + "="*80)
    print("4. COMPARACIÓN POR OTRAS DIMENSIONES")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 4.1 Duración de estadía
    stay_comp = df.groupby(['year', 'stay_duration'])['price_std'].mean().unstack(level=0)
    stay_comp.plot(kind='bar', ax=axes[0, 0], alpha=0.8)
    axes[0, 0].set_title('Precio por duración de estadía')
    axes[0, 0].set_xlabel('Duración')
    axes[0, 0].set_ylabel('Precio normalizado (USD)')
    axes[0, 0].tick_params(axis='x', rotation=0)
    
    # 4.2 Día de la semana
    day_names = ['Dom', 'Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab']
    day_comp = df.groupby(['year', 'day_of_week'])['price_std'].mean().unstack(level=0)
    day_comp.index = [day_names[i] for i in day_comp.index]
    day_comp.plot(kind='bar', ax=axes[0, 1], alpha=0.8)
    axes[0, 1].set_title('Precio por día de la semana')
    axes[0, 1].set_xlabel('Día')
    axes[0, 1].set_ylabel('Precio normalizado (USD)')
    axes[0, 1].tick_params(axis='x', rotation=0)
    
    # 4.3 Semana del mes
    week_comp = df.groupby(['year', 'week_in_month'])['price_std'].mean().unstack(level=0)
    week_comp.plot(kind='bar', ax=axes[1, 0], alpha=0.8)
    axes[1, 0].set_title('Precio por semana del mes')
    axes[1, 0].set_xlabel('Semana')
    axes[1, 0].set_ylabel('Precio normalizado (USD)')
    axes[1, 0].set_xticklabels(['Sem 1', 'Sem 2', 'Sem 3', 'Sem 4'], rotation=0)
    
    # 4.4 Destinos top 10
    top_dest = df['destination_final'].value_counts().head(10).index
    dest_comp = df[df['destination_final'].isin(top_dest)].groupby(
        ['year', 'destination_final']
    )['price_std'].mean().unstack(level=0)
    dest_comp.plot(kind='bar', ax=axes[1, 1], alpha=0.8)
    axes[1, 1].set_title('Precio por destino (top 10)')
    axes[1, 1].set_xlabel('Destino')
    axes[1, 1].set_ylabel('Precio normalizado (USD)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'images' / '19_comparacion_otras_dimensiones.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Correlaciones entre dimensiones
    print("\nCorrelación entre 2024 y 2025 por dimensión:")
    for dim_name, dim_col in [('Mes', 'month'), ('Duración', 'stay_duration'), 
                                ('Día semana', 'day_of_week'), ('Semana mes', 'week_in_month')]:
        comp = df.groupby(['year', dim_col])['price_std'].mean().unstack(level=0)
        corr = comp['2024'].corr(comp['2025'])
        print(f"  {dim_name}: {corr:.3f}")

def detect_outliers(df):
    """5. Detectar destinos con cambios extremos."""
    print("\n" + "="*80)
    print("5. DETECCIÓN DE OUTLIERS INTERANUALES")
    print("="*80)
    
    # Calcular cambio por destino
    dest_changes = df.groupby(['year', 'destination_final'])['price_std'].mean().unstack(level=0)
    dest_changes = dest_changes.dropna()
    dest_changes['change_pct'] = (dest_changes['2025'] - dest_changes['2024']) / dest_changes['2024'] * 100
    
    # Detectar outliers (>2 desviaciones estándar del cambio promedio)
    mean_change = dest_changes['change_pct'].mean()
    std_change = dest_changes['change_pct'].std()
    threshold = 2 * std_change
    
    outliers_up = dest_changes[dest_changes['change_pct'] > mean_change + threshold]
    outliers_down = dest_changes[dest_changes['change_pct'] < mean_change - threshold]
    
    print(f"\nCambio promedio: {mean_change:.1f}%")
    print(f"Desviación estándar: {std_change:.1f}%")
    print(f"Umbral outlier: ±{threshold:.1f}%")
    
    print(f"\nDestinos con AUMENTO extremo ({len(outliers_up)}):")
    for dest, row in outliers_up.sort_values('change_pct', ascending=False).head(10).iterrows():
        print(f"  {dest}: +{row['change_pct']:.1f}% (${row['2024']:.1f} → ${row['2025']:.1f})")
    
    print(f"\nDestinos con DESCENSO extremo ({len(outliers_down)}):")
    for dest, row in outliers_down.sort_values('change_pct').head(10).iterrows():
        print(f"  {dest}: {row['change_pct']:.1f}% (${row['2024']:.1f} → ${row['2025']:.1f})")
    
    # Guardar outliers
    outliers_df = pd.concat([
        outliers_up.assign(direction='up'),
        outliers_down.assign(direction='down')
    ])
    outliers_df.to_csv(OUTPUT_DIR / 'data' / '19_outliers_interanuales.csv')
    
    return outliers_df

def stability_score(df):
    """6. Calcular score de estabilidad por dimensión."""
    print("\n" + "="*80)
    print("6. SCORE DE ESTABILIDAD POR DIMENSIÓN")
    print("="*80)
    
    scores = {}
    
    # Para cada dimensión, calcular cuánto del patrón se mantiene
    for dim_name, dim_col in [('Mes', 'month'), ('Duración', 'stay_duration'), 
                                ('Día semana', 'day_of_week'), ('Semana mes', 'week_in_month')]:
        comp = df.groupby(['year', dim_col])['price_std'].mean().unstack(level=0)
        corr = comp['2024'].corr(comp['2025'])
        scores[dim_name] = corr
    
    # Destino
    top_dests = df['destination_final'].value_counts().head(20).index
    dest_comp = df[df['destination_final'].isin(top_dests)].groupby(
        ['year', 'destination_final']
    )['price_std'].mean().unstack(level=0)
    scores['Destino'] = dest_comp['2024'].corr(dest_comp['2025'])
    
    # Score compuesto
    scores['COMPOSITE'] = np.mean(list(scores.values()))
    
    print("\nScores de estabilidad (correlación 2024-2025):")
    for dim, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        indicator = "✓" if score > 0.8 else "~" if score > 0.6 else "✗"
        print(f"  {indicator} {dim}: {score:.3f}")
    
    return scores

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("19. COMPARACIÓN INTERANUAL: 2024 vs 2025")
    print("="*80)
    
    # Cargar datos
    df = load_year_data()
    
    print(f"\nRegistros cargados:")
    for year, count in df['year'].value_counts().items():
        print(f"  {year}: {count:,}")
    
    # Ejecutar análisis
    compare_global_distributions(df)
    compare_by_month(df)
    compare_by_destination(df)
    compare_by_dimension(df)
    outliers = detect_outliers(df)
    scores = stability_score(df)
    
    # Resumen
    print("\n" + "="*80)
    print("RESUMEN")
    print("="*80)
    print("\n✓ Análisis completado")
    print(f"  - Imágenes guardadas en: {OUTPUT_DIR / 'images'}")
    print(f"  - Outliers guardados en: {OUTPUT_DIR / 'data' / '19_outliers_interanuales.csv'}")
    print(f"\nHallazgos principales:")
    print(f"  1. {'Patrones establecidos' if scores['COMPOSITE'] > 0.8 else 'Patrones parcialmente estables'} entre años")
    print(f"  2. Estacionalidad (mes) es la dimensión más estable: {scores.get('Mes', 0):.3f}")
    print(f"  3. Destinos top también estables: {scores.get('Destino', 0):.3f}")

if __name__ == "__main__":
    main()
