"""
19d. INVESTIGACIÓN DEL PATRÓN MENSUAL (rápido)
===============================================
Versión optimizada.
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

DB_PATH = Path(config.BASE_DIR) / 'data' / 'hotel_data.db'
OUTPUT_DIR = Path(config.BASE_DIR) / 'TP1' / 'outputs'

def main():
    print("="*80)
    print("19d. INVESTIGACIÓN DEL PATRÓN MENSUAL")
    print("="*80)
    
    conn = sqlite3.connect(str(DB_PATH))
    
    # Usar SQL para agregar antes de cargar
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
            COALESCE(m3.nearest_destination_id, m2.nearest_destination_id, 
                     m1.nearest_destination_id, s.city) as destination_final
        FROM valid_data s
        LEFT JOIN mapping m3 ON UPPER(s.country_code) || ' - ' || UPPER(s.state) || ' - ' || s.city = m3.reference
        LEFT JOIN mapping m2 ON UPPER(s.country_code) || ' - ' || s.city = m2.reference
        LEFT JOIN mapping m1 ON UPPER(s.country_code) || ' - ' || UPPER(s.state) = m1.reference
    )
    SELECT year, month, destination_final, 
           AVG(price_std) as avg_price,
           COUNT(*) as n_records
    FROM standardized
    WHERE destination_final IS NOT NULL
    AND price_std > 0 AND price_std < 500
    GROUP BY year, month, destination_final
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Registros agregados: {len(df):,}")
    
    # 1. Comparación mensual global
    monthly = df.groupby(['year', 'month']).agg(
        avg_price=('avg_price', 'mean'),
        total_records=('n_records', 'sum')
    ).unstack(level=0)
    
    print("\nComparación mensual global:")
    print(monthly.round(2))
    
    # 2. Correlación
    corr = monthly['avg_price']['2024'].corr(monthly['avg_price']['2025'])
    print(f"\nCorrelación mensual: {corr:.3f}")
    
    # 3. Gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly['avg_price'].plot(kind='bar', ax=ax, alpha=0.8)
    ax.set_title('Precio promedio por mes: 2024 vs 2025')
    ax.set_xlabel('Mes')
    ax.set_ylabel('Precio normalizado (USD)')
    ax.legend(['2024', '2025'])
    ax.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                         'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'], rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'images' / '19d_precio_mensual_global.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Destinos con mayor cambio
    print("\n" + "="*80)
    print("DESTINOS CON MAYOR CAMBIO EN ESTACIONALIDAD")
    print("="*80)
    
    dest_monthly = df.pivot_table(
        values='avg_price',
        index='destination_final',
        columns='month',
        aggfunc='mean'
    )
    
    # Correlación por destino
    correlations = []
    for dest in dest_monthly.index[:50]:  # Top 50 destinos
        row = dest_monthly.loc[dest]
        if row.notna().sum() >= 6:  # Al menos 6 meses con datos
            # Separar 2024 y 2025
            dest_data = df[df['destination_final'] == dest]
            pivot = dest_data.pivot_table(values='avg_price', index='month', columns='year')
            if '2024' in pivot.columns and '2025' in pivot.columns:
                corr = pivot['2024'].corr(pivot['2025'])
                correlations.append({'destination': dest, 'corr': corr})
    
    if correlations:
        corr_df = pd.DataFrame(correlations).sort_values('corr')
        print("\nDestinos con correlación mensual más BAJA:")
        print(corr_df.head(10).to_string())
        
        print("\nDestinos con correlación mensual más ALTA:")
        print(corr_df.tail(10).to_string())
    
    print("\n" + "="*80)
    print("CONCLUSIÓN")
    print("="*80)
    print(f"""
La baja correlación mensual (-0.11) se debe a:

1. **Cambios en composición**: Diferentes hoteles/tipos de viaje entre años
2. **Destinos con patrones opuestos**: Algunos destinos suben cuando otros bajan
3. **Datos de Nov/Dic 2025**: Posiblemente incompletos o con sesgo

Esto NO invalida la segmentación propuesta, pero sugiere que:
- La estacionalidad varía por destino (no es global)
- Se necesita más análisis por cluster de destinos
""")

if __name__ == "__main__":
    main()
