"""
08 - Aplicar mapping y exploración
"""
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from auxiliary_functions import load_destination_mapping, apply_destination_mapping

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'

df = pd.read_pickle(OUTPUT_DIR / 'data' / 'df_expanded.pkl')
mapping_df = load_destination_mapping()

df = apply_destination_mapping(df, mapping_df)

print('Resultado del mapping:')
print(f'  Destinos únicos finales:         {df["destination_final"].nunique():,}')
sin_mapeo = df['destination_final'].isna().sum()
print(f'  Sin mapeo (fallback a ciudad):   {sin_mapeo:,} registros ({sin_mapeo/len(df):.1%})')
con_mapeo = (~df['destination_final'].isna()).sum()
print(f'  Con mapeo canónico:              {con_mapeo:,} registros ({con_mapeo/len(df):.1%})')

df.to_pickle(OUTPUT_DIR / 'data' / 'df_mapped.pkl')
print()
print(f'✓ df guardado en {OUTPUT_DIR / "data" / "df_mapped.pkl"}')
