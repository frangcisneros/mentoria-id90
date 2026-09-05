# AGENTS.md — Hotel Deals Analysis

## Project Overview

Python project for detecting hotel price deals using statistical z-score analysis on historical search data from ID90Travel. Part of DiploDatos 2026 mentorship.

## Active Work Area

- **`TP2_entrega/`** and `TP2_entrega.zip`: self-contained final TP2 deliverable (notebook with fresh outputs, local `config.py`/`auxiliary_functions.py` copies, `data/` sample + mapping, `reporte_final.md`, `INFORME_CORRECCIONES_TP1.md`, own README/requirements).
- **`TP1_entrega_corregida/`** and `TP1_entrega_corregida.zip`: self-contained final corrected TP1 deliverable (notebook with fresh outputs, local `config.py`/`auxiliary_functions.py`, `data/` sample + mapping, `reporte_final.md`, `INFORME_CORRECCIONES_TP1.md`, own README/requirements).
- Repo root: Streamlit app (`app.py`), baseline pipeline, shared `config.py` / `auxiliary_functions.py` / `test_system.py`.

## Notebooks (Jupytext)

The `.py` (py:percent) is the canonical source; the `.ipynb` is regenerated from it — never edit the `.ipynb` directly.

```bash
jupytext --sync TP1_entrega_corregida/TP1_exploracion_mercado.ipynb
jupytext --sync TP2_entrega/TP2_curacion_mercado.ipynb
python TP2_entrega/TP2_curacion_mercado.py   # standalone run (MPLBACKEND=agg outside Jupyter)
```

## Environment (this machine)

- **No conda.** System Python 3.14 (`/usr/bin/python`) + user packages in `~/.local`. Jupytext at `~/.local/bin/jupytext`.
- A `.venv` (Python 3.11) exists but is NOT the canonical env — do not activate it.
- `pytest` (8.4.2) is installed in system Python 3.14 (`python test_system.py` runs cleanly).
- `requirements.txt` pins may lag the installed env.

## Data Files

Gitignored (required for full runs):
```
data/datos_historicos_2024.csv   # ~2.5M records
data/datos_historicos_2025.csv   # ~2.6M (5.1M total)
data/hotel_data.db               # SQLite built by database.py from the CSVs
```
Tracked in the repo: `data/sample_data_300k.csv.gz` (random 300k sample, seed=42 — default input for the TP1/TP2 notebooks) and `data/destination_with_nearest.csv` (manual corrections applied; `_backup.csv` is the pre-correction original).

Notebooks run on the 300k sample by default; `USE_FULL_HISTORICALS=1` switches TP2 to the full historicals.

## Commands

```bash
# Generate baselines (required before app/tests)
python pipeline_build_baselines.py

# Web app
streamlit run app.py

# Tests (needs pytest installed + baselines present)
python test_system.py
```

`load_baselines()` reads `outputs/market_baselines.csv` and falls back to `outputs/baselines.csv`. The Dockerfile packages only the Streamlit app (python:3.10-slim, port 8501); `data/` must be mounted in.

## Legacy TP1 Scripts

`TP1/scripts/01..19_*.py` run sequentially; outputs go to `TP1/outputs/`. Superseded by the TP1 notebook but kept for provenance. `07_validacion_mapping.py` scores mapping quality; `07e_aplicar_cambios_mapping.py` applies the manual corrections listed below.

## Key Architecture

- **3-stage pipeline**: data extraction → baseline generation (`pipeline_build_baselines.py` → `outputs/*.csv`) → web app (`app.py` reads those outputs at startup). Never run app/tests without baselines.
- **Price standardization**: `price / (nights * rooms * (adults + kids))` — price per room-night-person. Column `price_std`.
- **Temporal expansion**: each booking becomes exactly `nights` rows (check-in + 0..nights-1); checkout day is NOT a paid night. Uses `nights` column, never `(date_end - date_start) + 1`.
- **Baselines**: weighted by `count_repeated` (search demand weight). Three explicit units: `demand_weight` = sum of count_repeated, `n_records` = row count, `count_obs` = alias of demand_weight (backcompat). Confidence threshold: ≥30 demand-weighted units per context.
- **Weighted vs unweighted stats**: production baselines (`auxiliary_functions.calculate_baselines`) are demand-weighted; the TP2 notebook computes its exploratory stats unweighted. Do not mix the two when comparing numbers.
- **Market definition (adopted in TP2)**: `destination_final × month × week_in_month × stay_duration` (corta 1-2n, media 3-5n, larga >5n). On the 300k sample it keeps 77% of demand in cells with N ≥ 30.
- **Detector adopted for TP3**: `is_deal = (z_log < -1.0) AND (1 - price_std/market_median >= 0.20)`, where `z_log` is the z-score of `ln(price_std)` within the market. Evaluated only in markets with N ≥ 30 (percentiles collapse in 1-2 record cells); sparser cells use the fallback cascade: exact week → full month → destination overall.
- **Buckets**: price tiers within destination (p25/p75 of `price_std`), controlled by `ENABLE_PRICE_BUCKETS` in `config.py`.
- **Classification**: z-score thresholds in `config.py:THRESHOLDS` (deal < -1.0, good_price < -0.5, expensive > 0.5).

## Destination Mapping

`data/destination_with_nearest.csv` maps ~26k raw city names to canonical destinations by geographic proximity. Manual corrections applied (documented criteria): Azusa→Redondo Beach, Bell Gardens→Long Beach, Arcadia→Los Angeles, Bellingham→Seattle, Anacortes→Astoria, Ann Arbor→Columbus, Arlington→Cambridge. Unmapped rows keep the raw city (never auto-assigned): on the 300k sample ~30% of rows / ~18% of demand.

## Config Location

All thresholds, paths, and feature flags live in `config.py` (root — used by app, pipeline, tests, and TP2; `TP1_corregido/config.py` is the TP1 package's own copy). Check there first when debugging classification behavior.
