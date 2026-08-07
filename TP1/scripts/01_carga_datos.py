"""
01 - Carga y exploración inicial de datos
Carga una muestra de 100k registros, muestra cobertura y estructura.
"""
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
DATA_DIR = OUTPUT_DIR / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', 20)

# Carga
archivos = sorted(glob.glob(str(config.DATA_DIR / config.HISTORICALS_FILE_PATTERN)))
if not archivos:
    raise FileNotFoundError(f'No se encontraron archivos en {config.DATA_DIR}')

df_raw = pd.read_csv(archivos[0], nrows=100_000)
print(f'Archivo:             {Path(archivos[0]).name}')
print(f'Registros cargados:  {len(df_raw):>12,}  (muestra de 100k)')
print(f'Columnas:            {len(df_raw.columns)}')
print()
print('Lista de columnas:')
for col in df_raw.columns:
    print(f'  - {col}  ({df_raw[col].dtype})')

# Cobertura
print()
print('COBERTURA DEL DATASET')
print('=' * 45)
print(f"  Fecha inicio:       {df_raw['date_start'].min()}")
print(f"  Fecha fin:          {df_raw['date_start'].max()}")
print(f"  Países:             {df_raw['country'].nunique()}")
print(f"  Estados/Provincias: {df_raw['state'].nunique():,}")
print(f"  Ciudades:           {df_raw['city'].nunique():,}")
print()
print('DEMANDA (count_repeated = búsquedas agregadas)')
print('=' * 45)
print(f"  Total búsquedas:    {df_raw['count_repeated'].sum():,.0f}")
print(f"  Media por registro: {df_raw['count_repeated'].mean():.1f}")
print(f"  Máximo:             {df_raw['count_repeated'].max():.0f}")

# Guardar para siguientes scripts
df_raw.to_pickle(DATA_DIR / 'df_raw.pkl')
print()
print(f'✓ df_raw guardado en {DATA_DIR / "df_raw.pkl"}')
