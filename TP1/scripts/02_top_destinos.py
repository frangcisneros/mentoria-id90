"""
02 - Top destinos por volumen de búsquedas
Gráfico de barras con las 20 ciudades con más demanda.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
IMAGES_DIR = OUTPUT_DIR / 'images'
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

df_raw = pd.read_pickle(OUTPUT_DIR / 'data' / 'df_raw.pkl')

top_dest = (
    df_raw.groupby('city')['count_repeated']
    .sum()
    .sort_values(ascending=False)
    .head(20)
)

fig, ax = plt.subplots(figsize=(13, 5))
top_dest.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
ax.set_title('Top 20 ciudades por volumen de búsquedas', fontsize=13)
ax.set_xlabel('Ciudad')
ax.set_ylabel('Búsquedas totales (count_repeated)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(IMAGES_DIR / '02_top_destinos.png', dpi=150)
plt.show()

top5_pct = top_dest.head(5).sum() / df_raw['count_repeated'].sum() * 100
print(f'Top 5 ciudades concentran el {top5_pct:.1f}% de la demanda total')
