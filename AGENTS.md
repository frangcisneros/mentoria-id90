# AGENTS.md — Hotel Deals Analysis

## Project Overview

Python project for detecting hotel price deals using statistical z-score analysis on historical search data from ID90Travel. Part of DiploDatos 2026 mentorship.

## Active Work Area

**`TP1_corregido/` is the current working package** (corrected TP1 per instructor feedback — see its `INFORME_CORRECCIONES_TP1.md`). Root-level files are the older app/pipeline.

The working notebook is **`TP1_corregido/tp1_exploracion_marimo.py`** (marimo format). Per owner decision, all new analysis work goes there — `TP1_exploracion_mercado.ipynb` is legacy reference only.

## Environment (this machine)

- **No conda.** System Python 3.14 + user packages in `~/.local`. marimo lives at `~/.local/bin/marimo` (export PATH if missing).
- `requirements.txt` pins are **stale** vs the installed env (installed: pandas 3.x, numpy 2.x). Don't blindly `pip install -r`; check what's already installed first.

## Data Files

Not in the repo (gitignored). Required before running anything:
```
data/datos_historicos_2024.csv   # ~2.5M records
data/datos_historicos_2025.csv
data/hotel_data.db               # SQLite built by database.py from the CSVs
```
`data/destination_with_nearest.csv` IS in the repo (manual corrections applied); `destination_with_nearest_backup.csv` is the pre-correction backup.

## Commands

```bash
# Generate baselines (required before app/tests)
python pipeline_build_baselines.py

# Web app
streamlit run app.py

# Tests (need outputs/*.csv baselines first) — currently 23/23 passing
python3 -m pytest test_system.py -q

# Open the working analysis notebook
marimo edit TP1_corregido/tp1_exploracion_marimo.py

# Execute the whole marimo notebook headless (~50s, runs all cells)
MPLBACKEND=Agg marimo export html TP1_corregido/tp1_exploracion_marimo.py -o out.html
```

⚠️ `marimo export script` does **not** execute cells — it only flattens code. Use `export html` to actually run the notebook headless.

## Marimo Notebook Rules

- Reactivity: a variable must be defined in exactly one cell. `_`-prefixed names are cell-local (safe to reuse).
- After editing, verify conflicts: `python3 scripts/tmp/diag_reactividad_marimo.py TP1_corregido/tp1_exploracion_marimo.py`
- Do NOT wrap the data-loading cell in `mo.persistent_cache` — it silently fails to save with these multi-GB DataFrames (investigated and abandoned; see marimo `_save/hash.py` if ever retried).
- Run marimo from the repo root: the notebook inserts the first directory containing `config.py` into sys.path. From root that resolves to the root pair (`auxiliary_functions.py`, `data/`); from inside `TP1_corregido/` it resolves to the duplicated pair there and falls back to the 300k sample dataset because `TP1_corregido/data/` has no historical CSVs.

## Legacy TP1 Scripts

`TP1/scripts/01..08_*.py` run sequentially; outputs go to `TP1/outputs/`. Superseded by the marimo notebook but kept for provenance. `07_validacion_mapping.py` scores mapping quality; `07e_aplicar_cambios_mapping.py` applies the manual corrections listed below.

## Key Architecture

- **3-stage pipeline**: data extraction → baseline generation (`pipeline_build_baselines.py` → `outputs/*.csv`) → web app (`app.py` reads those outputs at startup). Never run app/tests without baselines.
- **Price standardization**: `price / (nights * rooms * (adults + kids))` — price per room-night-person. Column `price_std`.
- **Temporal expansion**: each booking becomes exactly `nights` rows (check-in + 0..nights-1); checkout day is NOT a paid night. Uses `nights` column, never `(date_end - date_start) + 1`.
- **Baselines**: weighted by `count_repeated` (search demand weight). Three explicit units: `demand_weight` = sum of count_repeated, `n_records` = row count, `count_obs` = alias of demand_weight (backcompat). Confidence threshold: ≥30 demand-weighted units per context.
- **Buckets**: price tiers within destination, controlled by `ENABLE_PRICE_BUCKETS` in `config.py`.
- **Classification**: z-score thresholds in `config.py:THRESHOLDS` (deal < -1.0, good_price < -0.5, normal_upper > 0.5).

## Destination Mapping

`data/destination_with_nearest.csv` maps ~26k raw city names to canonical destinations by geographic proximity. Manual corrections applied (documented criteria): Azusa→Redondo Beach, Bell Gardens→Long Beach, Arcadia→Los Angeles, Bellingham→Seattle, Anacortes→Astoria, Ann Arbor→Columbus, Arlington→Cambridge. ~35% of records remain unmapped (kept as raw city, never auto-assigned).

## Config Location

All thresholds, paths, and feature flags live in `config.py` (root for the app; `TP1_corregido/config.py` is the corrected copy used by the marimo notebook when run from that directory). Check there first when debugging classification behavior.
