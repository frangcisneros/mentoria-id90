"""
03 - Normalización de precios
Estandariza precios y muestra distribución antes/después.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from auxiliary_functions import standardize_prices

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'
IMAGES_DIR = OUTPUT_DIR / 'images'
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

df_raw = pd.read_pickle(OUTPUT_DIR / 'data' / 'df_raw.pkl')

df = df_raw.drop_duplicates()
df = standardize_prices(df)

print('Comparación precio total vs. precio normalizado:')
print(df[['avg_price_average', 'avg_price_average_std']].describe().to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

clip_raw = df['avg_price_average'].quantile(0.99)
df['avg_price_average'].clip(upper=clip_raw).hist(
    bins=80, ax=axes[0], color='coral', edgecolor='white'
)
axes[0].set_title('Precio total (sin normalizar)')
axes[0].set_xlabel('USD')

clip_std = df['avg_price_average_std'].quantile(0.99)
df['avg_price_average_std'].clip(upper=clip_std).hist(
    bins=80, ax=axes[1], color='steelblue', edgecolor='white'
)
axes[1].set_title('Precio normalizado (por room-night-person)')
axes[1].set_xlabel('USD')

plt.tight_layout()
plt.savefig(IMAGES_DIR / '03_normalizacion.png', dpi=150)
plt.show()

df.to_pickle(OUTPUT_DIR / 'data' / 'df_norm.pkl')
print()
print(f'✓ df guardado en {OUTPUT_DIR / "data" / "df_norm.pkl"}')
