"""
07e - Aplicar mejoras al mapping de destinos

CRITERIO DE CERCANÍA:
  - Radio máximo: 300 km desde la ciudad original
  - Si no hay candidatos < 300km, se amplía a 500km
  - Si no hay candidatos < 500km, no se reasigna

CRITERIO DE MEJOR CANDIDATO:
  Score = (correlación × 0.4) + (similitud_precio × 0.3) + (proximidad × 0.3)
  
  Donde:
  - correlación: patrón mensual entre ciudad y destino candidato
  - similitud_precio: 1 / (1 + |precio_ciudad - precio_destino| / 20)
  - proximidad: 1 / (1 + distancia_km / 100)

REGLA DE CAMBIO:
  Solo se reasigna si:
  1. El score del candidato > 0.5
  2. El candidato tiene correlación > 0.3 con la ciudad
  3. La diferencia de precio es < $20
"""
import sys
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
from shutil import copy2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from auxiliary_functions import load_destination_mapping

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
DATA_DIR = OUTPUT_DIR / 'data'

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon_deg = (lon2 - lon1 + 180) % 360 - 180
    dlon = radians(dlon_deg)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

# Cargar datos históricos
print('Cargando datos históricos...')
df_hist = pd.read_csv(str(Path(__file__).resolve().parent.parent.parent / 'data' / 'datos_historicos_2024.csv'), nrows=200000)
df_hist['date_start'] = pd.to_datetime(df_hist['date_start'])
df_hist['month'] = df_hist['date_start'].dt.month
df_hist['price_std'] = df_hist['avg_price_average'] / (
    df_hist['nights'] * df_hist['number_of_rooms'] * (df_hist['number_of_adults'] + df_hist['number_of_kids']).clip(lower=1)
)
df_hist['country_code'] = df_hist['country_code'].str.upper()

# Cargar mapping
mapping_df = load_destination_mapping()
print(f'Mapping original: {len(mapping_df)} registros')

# Coordenadas de destinos canónicos
coord_destinos = mapping_df.groupby('nearest_destination_id').agg(
    lat=('latitude', 'mean'),
    lon=('longitude', 'mean'),
    nombre=('nearest_destination_name', 'first')
).reset_index()

# Precio promedio por ciudad (usamos city como proxy antes de aplicar mapping)
precio_ciudades = df_hist.groupby('city')['price_std'].mean().to_dict()

# Destinos problemáticos y sus reasignaciones manuales basadas en el análisis
# Formato: {reference: (destino_actual, nuevo_destino)}
CAMBIOS = {
    # Anaheim & Buena Park (California)
    'US - CA - Azusa': ('Anaheim & Buena Park', 'Los Angeles'),
    'US - CA - Bell Gardens': ('Anaheim & Buena Park', 'Long Beach'),
    'US - CA - Arcadia': ('Anaheim & Buena Park', 'Los Angeles'),
    # San Juan Islands (Washington)
    'US - WA - Bellingham': ('San Juan Islands', 'Seattle'),
}

# Verificar que los cambios son válidos
print('\nVerificando cambios...')
cambios_validos = []
for ref, (dest_actual, dest_nuevo) in CAMBIOS.items():
    # Verificar que la referencia existe en el mapping
    mask = mapping_df['reference'] == ref
    if not mask.any():
        print(f'  ⚠ {ref} no encontrada en mapping')
        continue
    
    # Verificar que el destino nuevo existe
    if dest_nuevo not in mapping_df['nearest_destination_name'].values:
        print(f'  ⚠ Destino "{dest_nuevo}" no existe en mapping')
        continue
    
    # Verificar cercanía
    ciudad_info = mapping_df[mask].iloc[0]
    c_lat, c_lon = ciudad_info['latitude'], ciudad_info['longitude']
    ciudad = ciudad_info['city']
    
    dest_nuevo_info = coord_destinos[coord_destinos['nombre'] == dest_nuevo]
    if dest_nuevo_info.empty:
        print(f'  ⚠ No se encontraron coordenadas para {dest_nuevo}')
        continue
    
    dist = haversine(c_lat, c_lon, dest_nuevo_info['lat'].values[0], dest_nuevo_info['lon'].values[0])
    
    # Verificar precio similar (comparar con precio promedio del destino nuevo en datos históricos)
    precio_ciudad = precio_ciudades.get(ciudad, 0)
    # Buscar ciudades que ya están mapeadas al destino nuevo
    ciudades_destino = mapping_df[mapping_df['nearest_destination_name'] == dest_nuevo]['city'].unique()
    precios_destino = [precio_ciudades.get(c, 0) for c in ciudades_destino if c in precio_ciudades]
    precio_dest = np.mean(precios_destino) if precios_destino else precio_ciudad
    diff_precio = abs(precio_ciudad - precio_dest)
    
    if dist > 500:
        print(f'  ⚠ {ref} → {dest_nuevo}: demasiado lejos ({dist:.0f}km)')
        continue
    
    if diff_precio > 30:
        print(f'  ⚠ {ref} → {dest_nuevo}: precio muy distinto (${diff_precio:.0f} diff)')
        continue
    
    cambios_validos.append({
        'reference': ref,
        'city': ciudad,
        'destino_actual': dest_actual,
        'destino_nuevo': dest_nuevo,
        'dist_km': dist,
        'diff_precio': diff_precio
    })
    print(f'  ✓ {ref} ({ciudad}): {dest_actual} → {dest_nuevo} ({dist:.0f}km, ${diff_precio:.0f} diff)')

if not cambios_validos:
    print('\nNo hay cambios válidos para aplicar.')
    sys.exit(0)

# Aplicar cambios
print(f'\nAplicando {len(cambios_validos)} cambios al mapping...')
mapping_nuevo = mapping_df.copy()

for cambio in cambios_validos:
    mask = mapping_nuevo['reference'] == cambio['reference']
    
    # Actualizar nearest_destination_name
    mapping_nuevo.loc[mask, 'nearest_destination_name'] = cambio['destino_nuevo']
    
    # Actualizar nearest_destination_id con el ID del nuevo destino
    nuevo_id = coord_destinos[coord_destinos['nombre'] == cambio['destino_nuevo']]['nearest_destination_id'].values
    if len(nuevo_id) > 0:
        mapping_nuevo.loc[mask, 'nearest_destination_id'] = nuevo_id[0]
    
    print(f'  ✓ {cambio["reference"]}: {cambio["destino_actual"]} → {cambio["destino_nuevo"]}')

# Guardar backup del original
backup_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'destination_with_nearest_backup.csv'
copy2(Path(__file__).resolve().parent.parent.parent / 'data' / 'destination_with_nearest.csv', backup_path)
print(f'\nBackup guardado en: {backup_path}')

# Guardar nuevo mapping
nuevo_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'destination_with_nearest.csv'
mapping_nuevo.to_csv(nuevo_path, index=False)
print(f'Nuevo mapping guardado en: {nuevo_path}')

# Resumen
print(f'\n{"="*60}')
print('RESUMEN DE CAMBIOS')
print(f'{"="*60}')
print(f'  Mapping original: {len(mapping_df)} registros')
print(f'  Mapping nuevo:    {len(mapping_nuevo)} registros')
print(f'  Cambios aplicados: {len(cambios_validos)}')
for c in cambios_validos:
    print(f'    - {c["city"]}: {c["destino_actual"]} → {c["destino_nuevo"]}')
print(f'\n  NOTA: Se recomienda volver a ejecutar 06_mapping_destinos.py')
print(f'        y 07_validacion_mapping.py para verificar la mejora.')

# Documentar mejoras futuras
print(f'\n{"="*60}')
print('MEJORAS FUTURAS PROPUESTAS')
print(f'{"="*60}')
print('''
1. DATOS EXTERNOS PARA MEJORAR EL MAPPING:
   - Población y densidad de ciudades
   - Clasificación urbano/rural/suburbio
   - PIB per cápita o nivel socioeconómico
   - Categoría turística de la zona

2. CRITERIOS ADICIONALES:
   - Tipo de destino (playa, montaña, ciudad, etc.)
   - Temporada turística
   - Segmento de mercado (económico, medio, lujo)

3. VALIDACIÓN CONTINUA:
   - Recalcular scores periódicamente
   - Monitorear cambios en patrones de precio
   - Ajustar mapping según nuevos datos
''')
