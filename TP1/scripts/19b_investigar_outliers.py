"""
19b. ANÁLISIS DE OUTLIERS INTERANUALES
=====================================
Investiga los valores extremos detectados.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config

import sqlite3
import pandas as pd

DB_PATH = Path(config.BASE_DIR) / 'data' / 'hotel_data.db'

def investigate_outliers():
    conn = sqlite3.connect(str(DB_PATH))
    
    # Buscar registros con precios extremos en 2025
    query = """
    WITH valid_data AS (
        SELECT * FROM raw_data
        WHERE nights > 0 AND number_of_rooms > 0 
        AND (number_of_adults + number_of_kids) > 0
    )
    SELECT 
        strftime('%Y', date_start) as year,
        city, state, country_code,
        avg_price_average, nights, number_of_rooms,
        (number_of_adults + number_of_kids) as total_pax,
        avg_price_average / (nights * number_of_rooms * 
            MAX(number_of_adults + number_of_kids, 1)) as price_std
    FROM valid_data
    WHERE avg_price_average > 50000
    ORDER BY avg_price_average DESC
    LIMIT 20
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print("Registros con precios > $50,000:")
    print(df.to_string())
    
    print("\n\nPosibles causas de valores extremos:")
    print("1. Precios en moneda local (no USD)")
    print("2. Errores de carga/datos")  
    print("3. Paquetes especiales (todo incluido, múltiples habitaciones)")
    print("4. Datos faltantes en nights/rooms que inflan el cálculo")
    
    return df

if __name__ == "__main__":
    investigate_outliers()
