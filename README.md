# Hotel Deals Detection & Market Analysis

**Diplomatura en Ciencia de Datos · Mentoría ID90Travel 2026**

Sistema de detección de ofertas de precios hoteleros basado en análisis estadístico (Z-Score y percentiles) sobre búsquedas históricas de la plataforma ID90Travel.

---

## Estructura del Repositorio

```text
mentoria-id90/
├── app.py                         # Aplicación interactiva en Streamlit para clasificar ofertas
├── auxiliary_functions.py         # Biblioteca de funciones de curación, mapping y baselines
├── config.py                      # Configuración global de umbrales, rutas y esquemas
├── pipeline_build_baselines.py    # Pipeline ETL que genera los baselines estadísticos
├── database.py                    # Gestor de base de datos SQLite para búsquedas
├── test_system.py                 # Suite de tests del sistema
│
├── TP1_corregido/                 # Paquete de entrega TP1 (versión corregida)
│   ├── TP1_exploracion_mercado.ipynb  # Notebook Jupyter interactivo
│   ├── TP1_exploracion_mercado.py     # Script Jupytext sincronizado (py:percent)
│   ├── INFORME_CORRECCIONES_TP1.md    # Devolución docente y correcciones aplicadas
│   ├── reporte_final.md               # Síntesis ejecutiva de la definición de mercado
│   ├── README.md                      # Documentación del TP1
│   └── logs/                          # Logs de ejecución del pipeline de baselines
│
├── TP2/                           # Paquete de entrega TP2 (Curación y Mercado)
│   ├── TP2_curacion_mercado.ipynb     # Notebook Jupyter interactivo
│   ├── TP2_curacion_mercado.py        # Script Jupytext sincronizado (py:percent)
│   └── README.md                      # Documentación, consignas y conclusiones del TP2
│
├── TP1/                           # Scripts exploratorios iniciales de referencia (01..19)
├── TP1_entrega/                   # Archivo histórico de la primera entrega de TP1
├── data/                          # Mapeo de destinos (versionado) y CSVs históricos (gitignored)
├── outputs/                       # Baselines generados por el pipeline (market_baselines.csv, etc.)
└── scripts/                       # Utilidades generales (conversión a PDF, preprocesamiento)
```

---

## Flujo de Trabajo con Jupytext

Los notebooks principales de análisis (`TP1_corregido` y `TP2`) están estructurados con **Jupytext** en formato percent (`py:percent`), permitiendo editarlos tanto como scripts de Python con celdas (`# %%`) en VS Code / PyCharm como en Jupyter Lab / Notebook.

### Sincronización automática
Para sincronizar cambios entre el script `.py` y el notebook `.ipynb`:

```bash
# Sincronizar TP1
jupytext --sync TP1_corregido/TP1_exploracion_mercado.ipynb

# Sincronizar TP2
jupytext --sync TP2/TP2_curacion_mercado.ipynb
```

### Ejecución directa desde consola
```bash
python TP2/TP2_curacion_mercado.py
```

---

## Pipeline y Aplicación Web

### 1. Generación de Baselines
Antes de ejecutar la aplicación Streamlit o los tests, es necesario generar los baselines de mercado:

```bash
python pipeline_build_baselines.py
```
Esto genera los archivos en `outputs/`:
- `market_baselines.csv`
- `price_distribution.csv`
- `bucket_summary.csv`

### 2. Aplicación Streamlit
```bash
streamlit run app.py
```

---

## Definición de Mercado

Tras las evaluaciones de homogeneidad y cobertura del TP1 y TP2, la segmentación adoptada es:

$$\\text{Contexto} = \\text{destination\\_final} \\times \\text{month} \\times \\text{week\\_in\\_month} \\times \\text{stay\\_duration}$$

- **`destination_final`**: Destino canónico consolidado mediante `destination_with_nearest.csv`.
- **`month`**: Mes de check-in (1 a 12).
- **`week_in_month`**: Semana del mes (1 a 4).
- **`stay_duration`**: Estadía corta (1-2 noches), media (3-5 noches) o larga (>5 noches).
- **Umbral de confianza**: $\\ge 30$ unidades de demanda ponderada (`count_repeated`).
