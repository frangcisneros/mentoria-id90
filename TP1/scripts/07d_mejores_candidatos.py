"""
07d - Buscar mejores candidatos de mapeo por cercanía + precio
Para cada ciudad de los mappings problemáticos, busca destinos cercanos
y evalúa cuál would be a better match por similitud de precio.
"""
import sys
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2, degrees
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from auxiliary_functions import load_destination_mapping, apply_destination_mapping

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
DATA_DIR = OUTPUT_DIR / 'data'

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon_deg = (lon2 - lon1 + 180) % 360 - 180
    dlon = radians(dlon_deg)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

# --- Cargar datos ---
df_hist = pd.read_csv(str(Path(__file__).resolve().parent.parent.parent / 'data' / 'datos_historicos_2024.csv'), nrows=200000)
df_hist['date_start'] = pd.to_datetime(df_hist['date_start'])
df_hist['month'] = df_hist['date_start'].dt.month
df_hist['price_std'] = df_hist['avg_price_average'] / (
    df_hist['nights'] * df_hist['number_of_rooms'] * (df_hist['number_of_adults'] + df_hist['number_of_kids']).clip(lower=1)
)
df_hist['country_code'] = df_hist['country_code'].str.upper()
mapping_df = load_destination_mapping()
df_hist = apply_destination_mapping(df_hist, mapping_df)

# Coordenadas de cada destino canónico
coord_destinos = mapping_df.groupby('nearest_destination_id').agg(
    lat=('latitude', 'mean'),
    lon=('longitude', 'mean'),
    nombre=('nearest_destination_name', 'first')
).reset_index()

# Destinos problemáticos
PROBLEMATICOS = [
    'Fort Worth', 'Dallas', 'Lafayette', 'Anaheim & Buena Park', 'Seattle',
    'Phoenix', 'St Louis', 'Hayward', 'Detroit', 'San Juan Islands'
]

# Para cada destino problemático, analizar cada ciudad
print('='*100)
print('MEJORES CANDIDATOS POR CERCANÍA + SIMILITUD DE PRECIO')
print('='*100)

for dest_actual in PROBLEMATICOS:
    grp = df_hist[df_hist['destination_name'] == dest_actual]
    if grp.empty:
        continue
    
    # Coordenadas del destino actual
    dest_info = coord_destinos[coord_destinos['nombre'] == dest_actual]
    if dest_info.empty:
        continue
    lat_actual = dest_info['lat'].values[0]
    lon_actual = dest_info['lon'].values[0]
    
    # Precio promedio mensual del destino actual
    precio_dest = grp.groupby(['city', 'month'])['price_std'].mean().unstack(fill_value=np.nan)
    
    print(f'\n{"─"*100}')
    print(f'DESTINO ACTUAL: {dest_actual}')
    print(f'{"─"*100}')
    
    for city in precio_dest.index:
        city_data = grp[grp['city'] == city]
        city_lat = city_data['city'].mode()  # No tenemos lat por ciudad directamente
        precio_city = city_data.groupby('month')['price_std'].mean()
        precio_prom = city_data['price_std'].mean()
        
        # Buscar coords de la ciudad en el mapping
        city_ref = mapping_df[mapping_df['city'] == city]
        if city_ref.empty:
            continue
        c_lat = city_ref['latitude'].values[0]
        c_lon = city_ref['longitude'].values[0]
        
        # Calcular distancias a todos los destinos
        coord_destinos['dist_km'] = coord_destinos.apply(
            lambda r: haversine(c_lat, c_lon, r['lat'], r['lon']), axis=1
        )
        
        # Destinos cercanos (menos de 300km, excluyendo el actual)
        cercanos = coord_destinos[
            (coord_destinos['dist_km'] < 300) & 
            (coord_destinos['nombre'] != dest_actual)
        ].copy()
        
        if cercanos.empty:
            cercanos = coord_destinos[
                (coord_destinos['dist_km'] < 800) & 
                (coord_destinos['nombre'] != dest_actual)
            ].copy()
        
        # Para cada destino cercano, calcular similitud de precio
        candidatos = []
        for _, cand in cercanos.iterrows():
            nombre_cand = cand['nombre']
            grp_cand = df_hist[df_hist['destination_name'] == nombre_cand]
            if len(grp_cand) < 50:
                continue
            precio_cand = grp_cand.groupby('month')['price_std'].mean()
            
            # Similitud: correlación de patrones mensuales + diferencia de precio promedio
            meses_comunes = precio_city.index.intersection(precio_cand.index)
            if len(meses_comunes) < 3:
                continue
            
            corr = precio_city[meses_comunes].corr(precio_cand[meses_comunes])
            diff_precio = abs(precio_prom - grp_cand['price_std'].mean())
            
            # Score: correlación alta + precio similar + cercanía
            cand_lat = cand['lat']
            cand_lon = cand['lon']
            dist = cand['dist_km']
            
            score = (corr * 0.4) + ((1 / (1 + diff_precio/20)) * 0.3) + ((1 / (1 + dist/100)) * 0.3)
            
            candidatos.append({
                'destino': nombre_cand,
                'dist_km': dist,
                'precio_prom': grp_cand['price_std'].mean(),
                'corr': corr,
                'diff_precio': diff_precio,
                'score': score,
                'n_obs': len(grp_cand)
            })
        
        candidatos_df = pd.DataFrame(candidatos).sort_values('score', ascending=False) if candidatos else pd.DataFrame()
        
        print(f'\n  CIUDAD: {city} (precio: ${precio_prom:.2f}, mapeada a: {dest_actual})')
        
        # Info de la ciudad actual
        print(f'    Actual: {dest_actual} → precio prom: ${precio_prom:.2f}')
        
        if not candidatos_df.empty:
            print(f'    Mejores alternativas cercanas:')
            for _, cand in candidatos_df.head(3).iterrows():
                print(f'      → {cand["destino"]:30s}  dist: {cand["dist_km"]:.0f}km  precio: ${cand["precio_prom"]:.2f}  corr: {cand["corr"]:.2f}  score: {cand["score"]:.2f}')
        else:
            print(f'    Sin alternativas cercanas encontradas')

print(f'\n\n✓ Análisis completado')
