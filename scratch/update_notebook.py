import json
from pathlib import Path

def update_nb(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Cell 3: Verification of DB or CSVs
    cell_3_code = """# Verificar fuentes de datos disponibles (SQLite, CSVs crudos o muestra pre-extraída)
sample_gz_path = Path(config.BASE_DIR) / 'data' / 'sample_data_300k.csv.gz'
hist_files = list((Path(config.BASE_DIR) / 'data').glob('datos_historicos_*.csv'))

if DB_PATH.exists():
    conn = sqlite3.connect(str(DB_PATH))
    count = conn.execute('SELECT COUNT(*) FROM raw_data').fetchone()[0]
    print(f'✓ Base de datos SQLite detectada: {count:,} registros totales')
    conn.close()
elif hist_files:
    print(f'✓ Archivos históricos crudos detectados: {[f.name for f in hist_files]}')
elif sample_gz_path.exists():
    print(f'✓ Archivo de muestra representativa detectado: {sample_gz_path.name} (8 MB)')
else:
    print('⚠ Nota: No se encontró fuente de datos en data/.')
"""
    nb['cells'][3]['source'] = [line + '\n' for line in cell_3_code.splitlines()]

    # Cell 5: Robust Sample Loading (SQLite -> CSVs -> sample_data_300k.csv.gz)
    cell_5_code = """# Cargar muestra representativa y aplicar pipeline de calidad, estandarización y mapeo
mapping_df = af.load_destination_mapping()
sample_gz_path = Path(config.BASE_DIR) / 'data' / 'sample_data_300k.csv.gz'
hist_files = sorted(list((Path(config.BASE_DIR) / 'data').glob('datos_historicos_*.csv')))

if DB_PATH.exists():
    conn = sqlite3.connect(str(DB_PATH))
    query = f'''
    SELECT * FROM raw_data
    WHERE nights > 0 AND number_of_rooms > 0 
    AND (number_of_adults + number_of_kids) > 0
    AND (city IS NOT NULL OR country IS NOT NULL OR country_code IS NOT NULL)
    AND ABS(RANDOM()) % 15 = 0
    LIMIT {SAMPLE_SIZE}
    '''
    df_raw = pd.read_sql_query(query, conn)
    conn.close()
elif hist_files:
    per_file = max(1000, int(SAMPLE_SIZE / len(hist_files)))
    dfs = [pd.read_csv(f, nrows=per_file, dtype={'city': str, 'state': str, 'country': str, 'country_code': str}, low_memory=False) for f in hist_files]
    df_raw = pd.concat(dfs, ignore_index=True)
elif sample_gz_path.exists():
    df_raw = pd.read_csv(sample_gz_path, dtype={'city': str, 'state': str, 'country': str, 'country_code': str}, low_memory=False)
else:
    raise FileNotFoundError("No se encontró hotel_data.db, datos_historicos_*.csv ni sample_data_300k.csv.gz en data/")

# Aplicar validación de datos, estandarización vectorizada y mapeo jerárquico
df_valid = af.validate_data(df_raw)
df_std = af.standardize_prices(df_valid)
df = af.apply_destination_mapping(df_std, mapping_df)

# Features temporales y de duración de estadía
df['date_start'] = pd.to_datetime(df['date_start'])
df['day_of_week'] = df['date_start'].dt.dayofweek
df['month'] = df['date_start'].dt.month
df['year'] = df['date_start'].dt.year

days = df['date_start'].dt.day.values
week_in_month = np.ones(len(days), dtype=int)
week_in_month[(days >= 8) & (days <= 15)] = 2
week_in_month[(days >= 16) & (days <= 22)] = 3
week_in_month[days >= 23] = 4
df['week_in_month'] = week_in_month

df['stay_duration'] = np.where(df['nights'] <= 2, 'corta', np.where(df['nights'] <= 5, 'media', 'larga'))
df['price_std'] = df['avg_price_average_std']

print(f"✓ Registros procesados en muestra:       {len(df):,}")
print(f"  Destinos canónicos únicos:             {df['destination_final'].nunique():,}")
print(f"  Países representados:                  {df['country_code'].nunique():,}")
print(f"  Registros con Mapeo Canónico:          {df['is_mapped'].sum():,} ({df['is_mapped'].mean():.1%})")
print(f"  Rango de fechas:                       {df['date_start'].min().strftime('%Y-%m-%d')} a {df['date_start'].max().strftime('%Y-%m-%d')}")
print(f"\\nEstadísticas de precio estandarizado (price_std = $/hab-noche-persona):")
print(df['price_std'].describe().round(2).to_string())
"""
    nb['cells'][5]['source'] = [line + '\n' for line in cell_5_code.splitlines()]

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Updated {nb_path} successfully!")

update_nb('TP1/TP1_exploracion_mercado.ipynb')
update_nb('TP1_entrega/TP1_exploracion_mercado.ipynb')
