"""
06 - Mapping de destinos + mapa interactivo
Carga mapping, calcula centroides, genera mapa con folium.
"""
import sys
import json
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2, degrees
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from auxiliary_functions import load_destination_mapping
import folium

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
MAPS_DIR = OUTPUT_DIR / 'maps'
MAPS_DIR.mkdir(parents=True, exist_ok=True)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon_deg = (lon2 - lon1 + 180) % 360 - 180
    dlon = radians(dlon_deg)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def centroide_geo(lats, lons):
    lats_r = np.radians(lats.astype(float))
    lons_r = np.radians(lons.astype(float))
    x = np.cos(lats_r) * np.cos(lons_r)
    y = np.cos(lats_r) * np.sin(lons_r)
    z = np.sin(lats_r)
    x_m, y_m, z_m = x.mean(), y.mean(), z.mean()
    lon = atan2(y_m, x_m)
    hyp = sqrt(x_m**2 + y_m**2)
    lat = atan2(z_m, hyp)
    return degrees(lat), degrees(lon)

mapping_df = load_destination_mapping()
print(f'Referencias en el mapping: {len(mapping_df):,}')
print(f'Destinos canónicos únicos: {mapping_df["nearest_destination_id"].nunique():,}')

centroides = []
for dest_id, grp in mapping_df.groupby('nearest_destination_id'):
    lat_c, lon_c = centroide_geo(grp['latitude'].values, grp['longitude'].values)
    centroides.append({
        'nearest_destination_id': dest_id,
        'lat_centro': lat_c,
        'lon_centro': lon_c,
        'n_ciudades': len(grp),
        'nombre': grp['nearest_destination_name'].iloc[0]
    })
centroides = pd.DataFrame(centroides)

mapping_con_dist = mapping_df.merge(
    centroides[['nearest_destination_id', 'lat_centro', 'lon_centro']],
    on='nearest_destination_id'
)
mapping_con_dist['dist_km'] = mapping_con_dist.apply(
    lambda r: haversine(r['latitude'], r['longitude'], r['lat_centro'], r['lon_centro']), axis=1
)

dispersion = mapping_con_dist.groupby('nearest_destination_id').agg(
    nombre=('nearest_destination_name', 'first'),
    n_ciudades=('city', 'count'),
    dist_max=('dist_km', 'max'),
    dist_prom=('dist_km', 'mean')
).sort_values('dist_prom', ascending=False)

def get_region(ref):
    if ref.startswith('US'): return 'USA'
    if ref.startswith('MX'): return 'México'
    if ref.startswith(('CA',)): return 'Canadá'
    if ref.startswith(('GB','IE','FR','DE','ES','IT','PT','NL','BE','CH','AT','SE','NO','DK','FI','PL','CZ','GR','TR','HU','RO','BG','HR','RS','BA','SK','SI','LT','LV','EE','IS','LU','MT','CY','AL','MK','ME','XK','AD','MC','LI','SM','VA','GI','FO','GL','SJ','JE','GG','IM')): return 'Europa'
    if ref.startswith(('JP','CN','KR','TH','VN','ID','MY','PH','SG','IN','LK','NP','BD','MM','KH','LA','TW','HK','MO')): return 'Asia'
    if ref.startswith(('AU','NZ','FJ','TO','WS','NC','PG','VU','PF','CK','NU','TK','AS','GU','MP','VI','PR','BM','KY','TC','VG','MS','BL','MF','GP','MQ','GF','RE','YT','PM','WF','PN','KI','NR','TV','MH','FM','PW')): return 'Oceanía/Pacífico'
    if ref.startswith(('BR','AR','CL','CO','PE','EC','VE','BO','PY','UY','CR','PA','GT','HN','SV','NI','CU','JM','HT','DO','TT','BB','BS','BZ','GY','SR')): return 'Latam'
    if ref.startswith(('ZA','MA','EG','TN','DZ','LY','NG','KE','ET','GH','TZ','UG','RW','BW','NA','ZM','ZW','MZ','MG','MU','SC','CV','ST','AO','CM','SN','CI','ML','BF','NE','TD','CF','CD','CG','GA','GQ','BI','DJ','ER','SO','SS','SD','LR','SL','GM','GN','GW','TG','BJ','MR')): return 'África'
    if ref.startswith(('RU','UA','BY','MD','GE','AM','AZ','KZ','UZ','TM','KG','TJ','MN')): return 'Asia Central/East'
    if ref.startswith(('IL','JO','LB','SY','IQ','IR','SA','AE','QA','BH','KW','OM','YE','PS')): return 'Medio Oriente'
    return 'Otro'

mapping_con_dist['region'] = mapping_con_dist['reference'].apply(get_region)
region_map = mapping_con_dist.groupby('nearest_destination_id')['region'].agg(lambda x: x.mode()[0])
dispersion['region'] = dispersion.index.map(region_map)
centroides['region'] = centroides['nearest_destination_id'].map(region_map)

regiones = sorted(dispersion['region'].unique())
region_colors = {
    'USA': '#e6194b', 'México': '#f58231', 'Canadá': '#4363d8',
    'Europa': '#3cb44b', 'Asia': '#911eb4', 'Oceanía/Pacífico': '#42d4f4',
    'Latam': '#f032e6', 'África': '#bfef45', 'Medio Oriente': '#fabed4',
    'Asia Central/East': '#469990', 'Otro': '#dcbeff'
}

m = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB positron')

fgs = {}
for reg in regiones:
    fgs[reg] = folium.FeatureGroup(name=reg, show=True)

for idx, row in dispersion.iterrows():
    dest_id = idx
    reg = row.get('region', 'Otro')
    color = region_colors.get(reg, '#999999')

    lat_c = centroides.loc[centroides['nearest_destination_id'] == dest_id, 'lat_centro'].values
    lon_c = centroides.loc[centroides['nearest_destination_id'] == dest_id, 'lon_centro'].values
    if len(lat_c) == 0:
        continue
    lat_c, lon_c = lat_c[0], lon_c[0]

    folium.Circle(
        location=[lat_c, lon_c],
        radius=row['dist_max'] * 1000,
        color=color, fill=True, fill_opacity=0.08, weight=1,
        popup=f"<b>{row['nombre']}</b><br>Región: {reg}<br>{int(row['n_ciudades'])} ciudades<br>Max: {row['dist_max']:.0f} km<br>Prom: {row['dist_prom']:.0f} km",
        tooltip=row['nombre']
    ).add_to(fgs[reg])

    folium.CircleMarker(
        location=[lat_c, lon_c],
        radius=max(3, min(10, row['n_ciudades'] / 5)),
        color=color, fill=True, fill_opacity=0.8, weight=1,
        tooltip=row['nombre']
    ).add_to(fgs[reg])

for fg in fgs.values():
    fg.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

legend_html = '<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;padding:10px;border-radius:5px;border:2px solid grey;font-size:12px;">'
legend_html += '<b>Filtrar por región</b><br>(usar LayerControl arriba a la derecha)<br><br>'
for reg in regiones:
    c = region_colors.get(reg, '#999')
    n = dispersion[dispersion['region']==reg]['n_ciudades'].sum()
    legend_html += f'<span style="background:{c};width:12px;height:12px;display:inline-block;border-radius:50%;margin-right:5px;"></span>{reg} ({int(n)})<br>'
legend_html += '</div>'
m.get_root().html.add_child(folium.Element(legend_html))

m.save(MAPS_DIR / '06_mapa_mapping.html')
print(f'\nTotal centroides: {len(dispersion)}')
for reg in regiones:
    n = len(dispersion[dispersion['region']==reg])
    print(f'  {reg}: {n} destinos')
print(f'\n✓ Mapa guardado en {MAPS_DIR / "06_mapa_mapping.html"}')
