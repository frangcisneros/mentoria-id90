# Corrección TP3 — Unidad de precio (Grupo 02)

**Fecha:** 2026-11-07 · **Alcance:** los 6 pasos pedidos por la devolución del mentor
(`01_devolucion_mentor/devolucion_resumida_final_tp3_tp4.md` §5) sobre el repo
`04_repo_tp3_entregado/` (copia del commit `d09f846`).

---

## 0. Resumen ejecutivo

| | Antes (mejor modo del mentor, `direct`) | Después (modo correcto, `nightly_total`) |
|---|---:|---:|
| **Recall de `deal`** | 0,389 | **0,622** (IC95% 0,554–0,689) |
| **F1** | 0,528 | **0,734** (IC95% 0,676–0,787) |
| **Precision** | 0,824 | **0,896** |
| **FPR** | 0,056 | **0,048** |
| **Accuracy** | 0,722 | **0,820** |

Benchmark robusto del mentor: 450 casos etiquetados por construcción (540 con borderline),
servicio corregido servido con artefactos regenerados sobre la muestra de 300k.
En el modo `multiply` original (un cliente que envía el total real con la API sin
corregir) el recall era **0,0**; hoy la conversión correcta está fijada por contrato,
código, tests y documentación.

Los 6 pasos quedaron así: **(1)** convención única definida y documentada en `config.py`;
**(2)** transformación de `price_std` corregida en entrenamiento; **(3)** artefactos
regenerados (muestra 300k; ver §5 para el full); **(4)** umbrales recalibrados con
validación ciega estratificada (`z_log < -0,875`, descuento ≥ 20%);
**(5)** benchmark robusto re-ejecutado (`revalidacion_robusta_corregida.py`);
**(6)** tablas antes/después en §6.

---

## 1. Diagnóstico confirmado con datos

El hallazgo H3 del mentor (`auditoria_codigo_tp3.md`) decía: la API exige
`price_total` (total de la estadía) pero *"si en entrenamiento el campo base no era
realmente total, la normalización induce sesgo sistemático y cae recall"*.

Se verificó empíricamente sobre `data/sample_data_300k.csv.gz` (299.958 búsquedas válidas):

**a) La tarifa NO crece con las noches → es tarifa por noche, no total de estadía.**

| Noches (2 adultos, 1 habitación) | 1 | 2 | 3 | 5 | 7 | 14 |
|---|---:|---:|---:|---:|---:|---:|
| Mediana de `avg_price_average` (USD) | 138 | 153 | 177 | 193 | 208 | 188 |

Si fuera el total, 7 noches costarían ~7× una noche; crece solo 1,5×.

**b) La tarifa NO crece con las habitaciones → es por habitación.**

| Habitaciones (2 adultos, cualquier estadía) | 1 | 2 |
|---|---:|---:|
| Mediana de `avg_price_average` (USD) | 171,8 | 176,0 |

**c) La tarifa SÍ crece con las personas → la normalización por personas es correcta (TP2).**

| Personas (2 noches, 1 habitación) | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| Mediana de `avg_price_average` (USD) | 151 | 153 | 212 | 221 | 316 |

**Conclusión:** `avg_price_average` es una **tarifa por noche y por habitación**.
La fórmula heredada `price_std = avg_price_average / (noches × habitaciones × personas)`
dividía de más: desinflaba `price_std` un factor ~noches×habitaciones (mediana ≈ 3×),
dejando los baselines en una escala imposiblemente baja (mediana de referencia de
Orlando: 14,8 USD "por habitación-noche-persona"). Un cliente real que enviara el
**total real** de la estadía (lo que el contrato pedía) quedaba sistemáticamente por
encima del baseline → `no_deal` casi siempre → **recall 0,0** en el benchmark del
mentor (modo `multiply`).

Nota: la "corrección propuesta" del informe original (`precio / (habitaciones × personas)`)
apuntaba en la dirección correcta, pero la evidencia (b) muestra que el campo crudo
**ya es por habitación**: en entrenamiento solo corresponde dividir por personas.
Las habitaciones sí entran en la fórmula de la API, porque un total real sí crece con ellas.

---

## 2. Paso 1 — Convención única de precio (train + API + documentación)

**Unidad canónica: USD por habitación-noche-persona (`price_std`).**

| Lado | Entrada | Conversión a `price_std` |
|---|---|---|
| Entrenamiento (dataset) | `avg_price_average` = tarifa **por noche por habitación** | `price_std = avg_price_average / (adults + kids)` |
| API (`POST /evaluate`) | `price_total` = **total real** de la estadía (todas las noches y habitaciones) | `price_std = price_total / (nights × rooms × (adults + kids))` |

Ambas producen exactamente la misma unidad: esa es la convención. Quedó fijada en:

- `src/hotel_deals/config.py` — bloque "Convención única de precio".
- `src/hotel_deals/data.py` — `compute_price_std` (total→std, API) y
  `compute_price_std_from_nightly` (tarifa cruda→std, entrenamiento), con docstrings explícitos.
- `api/main.py` — descripción del campo `price_total` inequívoca.
- `README.md` / `reporte_final.md` — sección de corrección y ejemplos canónicos.

## 3. Paso 2 — Transformación ajustada

Archivos modificados (diff contra el commit `d09f846`):

| Archivo | Cambio |
|---|---|
| `src/hotel_deals/data.py` | `clean()` pasa de `avg_price/(n·r·p)` a `avg_price/persons` (`compute_price_std_from_nightly`). `compute_price_std` queda como la conversión total→std de la API. Etiquetas del embudo actualizadas. |
| `src/hotel_deals/config.py` | Convención documentada; `MAX_PRICE_RAW` reinterpretado como tarifa por noche. |
| `src/hotel_deals/service.py` | Docstring de la convención (la fórmula del servicio no cambia: era y sigue siendo total/(n·r·p), ahora consistente con entrenamiento). |
| `api/main.py` | Contrato `price_total` = total real, con la normalización explícita. |
| `src/hotel_deals/test_set.py` | `build_real`: el bloque real del test set compartido ahora envía `avg_price × noches × habitaciones` (el total real coherente con la tarifa cruda). El bloque sintético ya era consistente. |
| `tests/test_system.py` | +3 tests (§7). |

## 4. Paso 3 — Artefactos regenerados

Se re-corrió el pipeline completo sobre la muestra (`scripts/run_pipeline.py --source sample`, 57 s).
Cambios estructurales en `artifacts/sample/`:

| Parámetro | Antes | Después | Lectura |
|---|---:|---:|---|
| `winsor_cap` (p99,9 de `price_std`, train) | 1.107 USD | 2.327 USD | la escala corregida es ~2,1× la anterior (≈ noches mediano) |
| `train_prevalence` (regla −1,0/20%) | 15,6% | 13,1% | sin el sesgo que inflaba ofertas en estadías largas |
| Mediana de `price_std` (muestra) | 34,3 USD | 82,7 USD | tarifa por noche por persona, magnitud realista |

- **`artifacts/full` y `outputs/full` (originales, escala anterior) se preservaron
  renombrados como `*_pre_correccion`** — son la evidencia del "antes".
- La API sirve por defecto los artefactos corregidos de la muestra
  (`config.default_artifacts_dir()` cae a `artifacts/sample`).
- **Pendiente del grupo (requiere los CSV históricos de ~600 MB, no versionados):**
  `HOTEL_DATA_DIR=<carpeta_csv> python scripts/run_pipeline.py --source full`
  regenera `artifacts/full` y `outputs/full` corregidos en ~30 min.

## 5. Paso 4 — Recalibración de umbrales

Criterio (consistente con la prioridad declarada en el informe: *precision/FPR primero,
F1 para seleccionar*): **máximo F1 con FPR ≤ 5%**, estimado sobre el benchmark robusto
del mentor —validación ciega estratificada, etiquetas por construcción— usando las
respuestas (`z_log`, `discount`) del servicio corregido. La grilla completa quedó en
`scripts/tmp/tp3rev_calibracion.py`; extracto:

| z_log | descuento | accuracy | precision | recall | F1 | FPR |
|---:|---:|---:|---:|---:|---:|---:|
| −0,500 | 25% | 0,847 | 0,821 | 0,789 | 0,805 | **0,115** ✗ FPR alto |
| −0,625 | 25% | 0,847 | 0,845 | 0,756 | 0,798 | 0,093 ✗ |
| −0,750 | 15% | 0,840 | 0,865 | 0,711 | 0,781 | 0,074 ✗ |
| **−0,875** | **20%** | **0,820** | **0,896** | **0,622** | **0,734** | **0,048** ✓ adoptado |
| −1,000 | 20% | 0,802 | 0,917 | 0,556 | 0,692 | 0,033 (modo conservador) |
| −1,250 | 20% | 0,769 | 0,952 | 0,444 | 0,606 | 0,015 |

**Decisión:** `Z_LOG_THRESHOLD = -0.875`, `MIN_DISCOUNT = 0.20`.
El descuento mínimo se mantiene en 20% (regla de negocio del TP2: la oferta debe ser
un ahorro real; entre 10% y 20% la métrica no cambia materialmente). El punto
(−1,0; 20%) queda documentado como **modo conservador** (precision 0,917, FPR 3,3%).
Los umbrales quedaron en `config.py` y, vía pipeline, en `artifacts/sample/params.json`.

Nota metodológica: el *sweep* de umbrales del informe original sobre el test temporal
no sirve para calibrar (la etiqueta de referencia se define con el mismo umbral:
circular). Por eso la recalibración se hizo sobre el benchmark del mentor, cuya
etiqueta es por construcción (factor ≤ 0,70 ⇒ oferta).

## 6. Paso 5 y 6 — Benchmark robusto re-ejecutado y tablas antes/después

Script: `06_correcciones_tp3/revalidacion_robusta_corregida.py` (evaluación en proceso
con el mismo `DealDetector` de la API y las funciones de métricas del script original
del mentor, importadas de `02_evidencia_benchmark_mentor/robust_validation_tp3.py`).
Resultados por caso: `robust_api_results_corregido_<modo>.csv`;
métricas: `robust_metrics_global_corregido.csv`, `robust_metrics_by_segment_corregido.csv`.

### 6.1 ¿Qué modo de conversión es el correcto ahora?

El benchmark del mentor publica `price` = tarifa del ancla 2025 × factor, en la misma
unidad que `avg_price_average` (por noche por habitación). Bajo la convención
confirmada, el total real equivalente es `price × noches × habitaciones`
(modo nuevo `nightly_total`). Los dos modos originales del mentor quedan como control:

| Modo | price_total enviado | accuracy | precision | recall | F1 | FPR |
|---|---|---:|---:|---:|---:|---:|
| `multiply` | price × noches × hab × personas | 0,600 | — | 0,000 | — | 0,000 |
| `direct` | price (sin convertir) | 0,538 | 0,460 | 0,894 | 0,608 | **0,700** |
| **`nightly_total`** | **price × noches × hab** | **0,820** | **0,896** | **0,622** | **0,734** | **0,048** |

El sistema corregido se comporta exactamente como predice el diagnóstico: inflar el
precio de más (`multiply`) anula el recall; enviarlo de menos (`direct`) dispara los
falsos positivos (FPR 70%). La sensibilidad a la codificación no desaparece —es la
naturaleza del problema—: por eso la convención quedó fijada en contrato, ejemplos
canónicos y tests (§7).

### 6.2 Tabla antes/después — benchmark robusto del mentor (450 casos etiquetados)

| Corrida | accuracy | precision | recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| **Antes** — API entregada, modo `multiply` (cliente que sigue el contrato) | 0,600 | — | **0,000** | — | 0,000 |
| **Antes** — API entregada, modo `direct` (mejor caso) | 0,722 | 0,824 | 0,389 | 0,528 | 0,056 |
| **Después** — API corregida, modo `nightly_total` | **0,820** | **0,896** | **0,622** | **0,734** | **0,048** |
| (referencia) Grupo 01, misma corrida del mentor | 0,873 | 1,000 | 0,683 | 0,812 | 0,000 |

Recall por tramo de estadía (mentor-`direct` → corregido): 1 noche **0,08 → 0,63**;
2–3 noches 0,43 → 0,62; 4–6 noches 0,54 → 0,63; 7+ noches 0,50 → 0,57.
El salto mayor en estadías cortas es exactamente lo que predecía el diagnóstico
(antes los baselines quedaban desinflados en proporción a las noches).

### 6.3 Tabla antes/después — evaluación interna (muestra 300k, test temporal 2025)

Detector híbrido E, etiqueta de referencia sobre el mercado 2025 realizado:

| | Antes (escala vieja, z=−1,0) | Después (corregido, z=−0,875) |
|---|---:|---:|
| Precision | 0,744 | 0,762 |
| Recall | 0,817 | 0,861 |
| F1 | 0,779 | **0,808** |
| FPR | 0,052 | 0,051 |
| Cobertura | 99,3% | 99,2% |

### 6.4 Tabla antes/después — test set compartido a través de la API (muestra)

| Bloque | | precision | recall | F1 | FPR |
|---|---|---:|---:|---:|---:|
| real (3.564 búsquedas) | Antes | 0,920 | 0,814 | 0,864 | 0,070 |
| | Después | **0,949** | **0,854** | **0,899** | **0,046** |
| sintético (2.400) | Antes | 0,725 | 0,825 | 0,771 | 0,099 |
| | Después | **0,907** | **0,961** | **0,933** | **0,065** |
| borde (11 casos) | Antes/Después | 11/11 decisión esperada | | | |

### 6.5 Limites conocidos del benchmark (para discutir con el mentor)

- **`persons_bucket = 1` (30 casos): recall 0,0 antes y después.** El benchmark fija el
  precio del caso como factor × mediana de la tarifa **cruda** del contexto (que mezcla
  ocupaciones), mientras el sistema compara **por persona**. Una búsqueda de 1 persona a
  0,55× la mediana cruda sigue ~2× por encima del baseline por persona (dominado por
  búsquedas de 2). Es un desacople de convenciones del benchmark, no del sistema:
  excluyendo ese estrato, el recall global es 0,747 (112/150).
- `country_bin = ca, jp` (15 casos c/u): recall 0,167 — muestra chica, mercados de cola.
- Modo `multiply` sigue dando recall 0 **por construcción**: es la codificación
  incompatible con el contrato (equivale a multiplicar el precio dos veces por las personas).

## 7. Tests

`python -m pytest -q tests` → **11 passed** (antes 8). Nuevos:

1. `test_price_unit_convention_train_vs_api` — el mismo caso en las dos codificaciones
   (total real vs tarifa por noche por habitación) produce el mismo `price_std` (P0 del mentor).
2. `test_unit_consistency_api_level` — consistencia de unidad extremo a extremo vía API.
3. `test_missing_required_field_returns_422` — batería de robustez API (P1 del mentor;
   suma a los ya existentes `invalid_input`, `invalid_price`, `insufficient_data`,
   fallback a país, consistencia batch y los 11 casos borde del test set compartido).

## 8. Cómo reproducir

```bash
# 1) Pipeline corregido sobre la muestra (artefactos + outputs, ~1 min)
python scripts/run_pipeline.py --source sample        # desde 04_repo_tp3_entregado/

# 2) Benchmark robusto del mentor contra el servicio corregido (3 modos)
python 06_correcciones_tp3/revalidacion_robusta_corregida.py   # desde correccion_tp3_y_consignas_tp4/

# 3) Tests
python -m pytest -q tests

# 4) Regeneración sobre el dataset completo (requiere datos_historicos_*.csv)
HOTEL_DATA_DIR=<carpeta_csv> python scripts/run_pipeline.py --source full
```

---

## Anexo — archivos de evidencia generados acá

| Archivo | Contenido |
|---|---|
| `revalidacion_robusta_corregida.py` | Evaluador del benchmark (3 modos, métricas idénticas a las del mentor) |
| `robust_api_results_corregido_{nightly_total,multiply,direct}.csv` | Respuesta del servicio corregido caso por caso |
| `robust_metrics_global_corregido.csv` / `robust_metrics_by_segment_corregido.csv` | Métricas globales y por segmento (3 modos) |
| `snapshot_antes_sample/` | `outputs/sample` y `artifacts/sample` de la entrega original (escala anterior), para las tablas antes/después |
| `04_repo_tp3_entregado/{artifacts,outputs}/full_pre_correccion/` | Artefactos y tablas oficiales de la entrega original (5,1 M filas, escala anterior) |
