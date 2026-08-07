"""
05 - Estacionalidad: precio promedio por mes
Dos gráficos: barras verticales por mes + barras horizontales ordenadas menor a mayor.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
IMAGES_DIR = OUTPUT_DIR / 'images'
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_pickle(OUTPUT_DIR / 'data' / 'df_expanded.pkl')

precios_mes = (
    df.groupby('month')['avg_price_average_std']
    .mean()
    .reset_index()
)

month_names = {1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',
               7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'}

fig, axes = plt.subplots(2, 1, figsize=(11, 9))

ax = axes[0]
ax.bar(precios_mes['month'], precios_mes['avg_price_average_std'],
       color='steelblue', edgecolor='white')
ax.set_xlabel('Mes')
ax.set_ylabel('Precio normalizado promedio (USD)')
ax.set_title('Estacionalidad: precio promedio normalizado por mes')
ax.set_xticks(range(1, 13))
ax.set_xticklabels([month_names[i] for i in range(1, 13)])

ax = axes[1]
ordenado = precios_mes.sort_values('avg_price_average_std')
labels = [month_names[m] for m in ordenado['month']]
bars = ax.barh(labels, ordenado['avg_price_average_std'],
               color='steelblue', edgecolor='white')
for bar, val in zip(bars, ordenado['avg_price_average_std']):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'${val:.2f}', va='center', fontsize=10)
ax.set_xlabel('Precio normalizado promedio (USD)')
ax.set_title('Meses ordenados por precio promedio (menor → mayor)')
ax.set_xlim(0, ordenado['avg_price_average_std'].max() * 1.12)

plt.tight_layout()
plt.savefig(IMAGES_DIR / '05_estacionalidad.png', dpi=150)
plt.show()
