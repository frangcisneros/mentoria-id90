"""
04 - Expansión temporal y features
Expande estadías multi-noche a observaciones diarias, agrega month y week_in_month.
"""
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from auxiliary_functions import expand_dates_dataframe, add_temporal_features

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'outputs'

df = pd.read_pickle(OUTPUT_DIR / 'data' / 'df_norm.pkl')

registros_antes = len(df)
df = expand_dates_dataframe(df)
print(f'Registros antes:   {registros_antes:>12,}')
print(f'Registros después: {len(df):>12,}')
print(f'Factor expansión:  {len(df)/registros_antes:.2f}x')

df = add_temporal_features(df)
print(f'\nColumnas agregadas: month, week_in_month')
print(df[['date_start', 'date_end', 'nights', 'date', 'month', 'week_in_month']].head(10).to_string())

df.to_pickle(OUTPUT_DIR / 'data' / 'df_expanded.pkl')
print()
print(f'✓ df guardado en {OUTPUT_DIR / "data" / "df_expanded.pkl"}')
