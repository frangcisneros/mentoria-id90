# AGENTS.md — Hotel Deals Analysis

## Project Overview

Python project for detecting hotel price deals using statistical z-score analysis on historical search data from ID90Travel. Part of DiploDatos 2026 mentorship.

## Critical Setup

Data files (~200MB/year) are **not in the repo**. Download from Google Drive before running anything:
```
data/datos_historicos_2024.csv
data/datos_historicos_2025.csv
```
`data/destination_with_nearest.csv` IS in the repo (with manual corrections applied).

## Commands

```bash
# Environment setup
conda create -n hotels python=3.11 && conda activate hotels
pip install -r requirements.txt

# Generate baselines (required before app/tests)
python pipeline_build_baselines.py

# Run web app
streamlit run app.py

# Run tests (requires baselines to exist first)
pytest test_system.py -v
```

## Pipeline Order

1. `pipeline_build_baselines.py` generates `outputs/*.csv` files
2. `app.py` reads those outputs at startup
3. Tests in `test_system.py` also depend on generated baselines

**Never run `app.py` or tests without first generating baselines.**

## TP1 Scripts (Exploration & Validation)

Located in `TP1/scripts/`, numbered for sequential execution:
```bash
cd TP1/scripts
for i in 01 02 03 04 05 06 07 08; do python3 ${i}_*.py; done
```

Outputs go to `TP1/outputs/` (data/, images/, maps/).

Key scripts:
- `07_validacion_mapping.py` — scores mapping quality (CV + correlation)
- `07e_aplicar_cambios_mapping.py` — applies mapping corrections with documented criteria

## Key Architecture

- **3-stage pipeline**: data extraction → baseline generation → web app
- **Price standardization**: `price / (nights * rooms * (adults + kids))` — price per room-night-person
- **Baselines**: statistical summaries grouped by `destination_final × month × week_in_month × price_bucket`
- **Buckets**: price tiers (budget/mid-range/premium) within each destination, controlled by `ENABLE_PRICE_BUCKETS` flag in `config.py`
- **Classification**: z-score thresholds in `config.py:THRESHOLDS` (deal < -1.0, good_price < -0.5, normal_upper > 0.5)

## Destination Mapping

`data/destination_with_nearest.csv` maps ~26k raw city names to canonical destinations by geographic proximity. Manual corrections have been applied:
- Azusa, Bell Gardens, Arcadia → reassigned from Anaheim to nearby destinations
- Bellingham, Anacortes → reassigned from San Juan Islands
- Ann Arbor → reassigned from Detroit to Columbus

Mapping validation script (`07_validacion_mapping.py`) uses composite score combining CV (price variability) and correlation (seasonal pattern alignment).

## Test Issues

`test_system.py` has discrepancies with current code:
- Calls `calcular_total_std` but function is `calculate_price_std`
- Expects Spanish labels ("Buen Precio") but config uses English ("Good Price")
- Tests will fail until fixed

## Config Location

All thresholds, paths, and feature flags live in `config.py`. Check there first when debugging classification behavior.

## Data Notes

- `outputs/` directory is gitignored — must be regenerated locally
- ~40% of records lack destination mapping (fallback to raw city name)
- Historical data covers 2024-2025, ~2.5M records per year
- `data/destination_with_nearest_backup.csv` exists as backup of original mapping
