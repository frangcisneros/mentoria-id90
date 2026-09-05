# TP1 Corregido — Exploración de Mercado y Homogeneidad de Precios
## Diplomatura en Ciencia de Datos · Mentoría ID90Travel

Este directorio contiene la **versión corregida** de la entrega de TP1, aplicando las devoluciones del docente sobre:

- Unidad de análisis y expansión temporal.
- Etiquetado de días de semana.
- Cobertura y trazabilidad del mapping de destinos.
- Separación de `demand_weight`, `n_records` y `count_obs`.
- Política explícita de outliers.
- SNR acompañado de soporte de datos.

Ver `reporte_final.md` para la síntesis ejecutiva y `INFORME_CORRECCIONES_TP1.md` para el detalle técnico completo.

---

## Estructura del paquete de entrega

```text
TP1_entrega_corregida/
├── TP1_exploracion_mercado.ipynb       # Notebook corregido con outputs completos
├── TP1_exploracion_mercado.py          # Script fuente Jupytext (py:percent)
├── auxiliary_functions.py              # Funciones auxiliares corregidas
├── config.py                           # Configuración local autocontenida
├── pipeline_build_baselines.py         # Pipeline de baselines
├── database.py                         # Utilitario SQLite
├── requirements.txt                    # Dependencias
├── README.md                           # Esta guía
├── reporte_final.md                    # Síntesis ejecutiva
├── INFORME_CORRECCIONES_TP1.md         # Informe técnico de correcciones
├── scripts/
│   └── destination_mapping_preprocess.py  # Script de referencia del docente
│
├── data/                               # Datos locales para ejecución inmediata
│   ├── destination_with_nearest.csv
│   ├── destination_with_nearest_backup.csv
│   └── sample_data_300k.csv.gz         # Muestra representativa de 300k (seed=42)
│
└── outputs/                            # Baselines generados por el pipeline
    ├── market_baselines.csv
    ├── price_distribution.csv
    └── bucket_summary.csv
```

---

## Instalación y ejecución

### 1. Entorno virtual y dependencias

```bash
cd TP1_entrega_corregida
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install scikit-learn statsmodels jupyter
```

### 2. Datos

Los archivos `datos_historicos_2024.csv` y `datos_historicos_2025.csv` deben estar en `data/`. En el repo raíz ya existen; si no están, descargarlos del Google Drive de la materia.

### 3. Ejecutar el pipeline completo

```bash
python pipeline_build_baselines.py
```

Esto genera los baselines en `outputs/` usando ambos años completos (2024 + 2025).

### 4. Ejecutar el notebook

```bash
jupyter notebook TP1_exploracion_mercado.ipynb
```

> **Nota:** las conclusiones finales deben basarse en los outputs del pipeline completo, no en la muestra de 300.000 filas.

---

## Resultados principales (versión corregida)

| Métrica | Valor |
|---|---|
| Filas originales (2024 + 2025) | 5,158,190 |
| Filas con mapeo canónico | 69,0% |
| Demanda mapeada | 80,5% |
| Contextos generados | 673.030 |
| Contextos de alta confianza (N≥30) | 157.705 (23,4%) |
| Demanda en contextos de alta confianza | 96,3% |

### Segmentación propuesta

```text
Contexto = destination_final × month × week_in_month × stay_duration
```

`price_bucket` es opcional. Es un proxy de categoría hotelera porque no tenemos estrellas, amenities ni marca.

### Principales correcciones respecto al TP1 original

1. **Expansión temporal:** ahora se generan exactamente `nights` noches pagadas, sin incluir el checkout.
2. **Días de semana:** corregido el diccionario (`0 = Lunes`, ..., `6 = Domingo`).
3. **Mapping:** integrado el pretratamiento del docente con `match_level` para auditar cada match.
4. **Unidad de análisis:** separadas `demand_weight` (suma de `count_repeated`), `n_records` (filas/noches) y `count_obs`.
5. **Outliers:** política explícita con filtro de precios negativos y winsorización al percentil 99.9.
6. **SNR:** reportado junto con soporte de datos (% de demanda en contextos confiables).

---

## Documentación adicional

- `reporte_final.md` — Síntesis ejecutiva de resultados.
- `INFORME_CORRECCIONES_TP1.md` — Detalle técnico de cada corrección y comparación original vs corregido.
