"""
19c. COMPARACIÓN INTERANUAL LIMPIA
===================================
Compara 2024 vs 2025 con datos limpiados.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config

import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

DB_PATH = Path(config.BASE_DIR) / 'data' / 'hotel_data.db'
OUTPUT_DIR = Path(config.BASE_DIR) / 'TP1' / 'outputs'

def load_clean_data():
    """Carga datos limpiando outliers obvios."""
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
                     m1.nearest_destination_id, s.city) as destination_final
        FROM standardized s
        LEFT JOIN mapping m3 ON UPPER(s.country_code) || ' - ' || UPPER(s.state) || ' - ' || s.city = m3.reference
        LEFT JOIN mapping m2 ON UPPER(s.country_code) || ' - ' || s.city = m2.reference
        LEFT JOIN mapping m1 ON UPPER(s.country_code) || ' - ' || UPPER(s.state) = m1.reference
    )
    SELECT * FROM with_dest 
    WHERE destination_final IS NOT NULL
    AND price_std > 0 AND price_std < 500
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def main():
    print("="*80)
    print("19c. COMPARACIÓN INTERANUAL LIMPIA (sin outliers)")
    print("="*80)
    
    df = load_clean_data()
    
    print(f"\nRegistros después de limpieza:")
    for year, count in df['year'].value_counts().items():
        print(f"  {year}: {count:,}")
    
    # 1. Distribuciones
    print("\n" + "="*80)
    print("1. DISTRIBUCIONES DE PRECIOS")
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
    plt.savefig(OUTPUT_DIR / 'images' / '19c_distribucion_precios.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    stats = df.groupby('year')['price_std'].agg(['mean', 'median', 'std'])
    print("\nEstadísticas por año:")
    print(stats.round(2))
    
    prices_2024 = df[df['year'] == '2024']['price_std'].dropna()
    prices_2025 = df[df['year'] == '2025']['price_std'].dropna()
    stat, p_value = mannwhitneyu(prices_2024, prices_2025, alternative='two-sided')
    print(f"\nTest Mann-Whitney U: p-value = {p_value:.6f}")
    print(f"  {'Diferencia significativa' if p_value < 0.05 else 'Sin diferencia significativa'}")
    
    # 2. Por mes
    print("\n" + "="*80)
    print("2. PATRÓN ESTACIONAL POR MES")
    print("="*80)
    
    monthly = df.groupby(['year', 'month'])['price_std'].mean().unstack(level=0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly.plot(kind='bar', ax=ax, alpha=0.8)
    ax.set_title('Precio promedio por mes: 2024 vs 2025')
    ax.set_xlabel('Mes')
    ax.set_ylabel('Precio normalizado (USD)')
    ax.legend(['2024', '2025'])
    ax.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                         'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'], rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'images' / '19c_precio_por_mes.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    corr = monthly['2024'].corr(monthly['2025'])
    print(f"\nCorrelación mensual 2024-2025: {corr:.3f}")
    
    monthly['diff_pct'] = ((monthly['2025'] - monthly['2024']) / monthly['2024'] * 100)
    print("\nDiferencia porcentual por mes:")
    print(monthly[['diff_pct']].round(1))
    
    # 3. Top destinos
    print("\n" + "="*80)
    print("3. TOP DESTINOS: 2024 vs 2025")
    print("="*80)
    
    top_dest = df['destination_final'].value_counts().head(20).index
    dest_comp = df[df['destination_final'].isin(top_dest)].groupby(
        ['year', 'destination_final']
    )['price_std'].mean().unstack(level=0)
    dest_comp['change_pct'] = ((dest_comp['2025'] - dest_comp['2024']) / dest_comp['2024'] * 100)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    dest_comp[['2024', '2025']].plot(kind='barh', ax=axes[0], alpha=0.8)
    axes[0].set_title('Precio por destino: 2024 vs 2025')
    axes[0].set_xlabel('Precio normalizado (USD)')
    
    dest_comp['change_pct'].sort_values().plot(kind='barh', ax=axes[1],
        color=['green' if x > 0 else 'red' for x in dest_comp['change_pct'].sort_values()])
    axes[1].set_title('Cambio porcentual 2024→2025')
    axes[1].set_xlabel('Cambio (%)')
    axes[1].axvline(0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'images' / '19c_top_destinos.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nTop 10 destinos por cambio:")
    print(dest_comp[['change_pct', '2024', '2025']].sort_values('change_pct').round(2).to_string())
    
    # 4. Otras dimensiones
    print("\n" + "="*80)
    print("4. COMPARACIÓN POR OTRAS DIMENSIONES")
    print("="*80)
    
    scores = {}
    for dim_name, dim_col in [('Mes', 'month'), ('Duración', 'stay_duration'), 
                                ('Día semana', 'day_of_week'), ('Semana mes', 'week_in_month')]:
        comp = df.groupby(['year', dim_col])['price_std'].mean().unstack(level=0)
        scores[dim_name] = comp['2024'].corr(comp['2025'])
    
    top_dests = df['destination_final'].value_counts().head(20).index
    dest_corr = df[df['destination_final'].isin(top_dests)].groupby(
        ['year', 'destination_final'])['price_std'].mean().unstack(level=0)
    scores['Destino'] = dest_corr['2024'].corr(dest_corr['2025'])
    
    scores['COMPOSITE'] = np.mean(list(scores.values()))
    
    print("\nScores de estabilidad (correlación 2024-2025):")
    for dim, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        indicator = "✓" if score > 0.8 else "~" if score > 0.6 else "✗"
        print(f"  {indicator} {dim}: {score:.3f}")
    
    # Guardar resumen
    summary = pd.DataFrame({
        'Metric': ['Registros 2024', 'Registros 2025', 'Media 2024', 'Media 2025', 
                   'Mediana 2024', 'Mediana 2025', 'Correlación mensual', 'Score composite'],
        'Value': [len(df[df['year']=='2024']), len(df[df['year']=='2025']),
                  stats.loc['2024','mean'], stats.loc['2025','mean'],
                  stats.loc['2024','median'], stats.loc['2025','median'],
                  corr, scores['COMPOSITE']]
    })
    summary.to_csv(OUTPUT_DIR / 'data' / '19c_resumen_interanual.csv', index=False)
    
    print("\n" + "="*80)
    print("RESUMEN FINAL")
    print("="*80)
    print(f"\nLa comparación interanual muestra:")
    print(f"  • {'Patrones estables' if scores['COMPOSITE'] > 0.8 else 'Patrones parcialmente estables'} entre 2024 y 2025")
    print(f"  • Correlación mensual: {corr:.3f}")
    print(f"  • Score de estabilidad global: {scores['COMPOSITE']:.3f}")

if __name__ == "__main__":
    main()
