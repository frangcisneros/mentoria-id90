# TP2 — Curación de datos y definición de mercado

**Diplomatura en Ciencia de Datos · Mentoría ID90Travel 2026**  
**Fecha de entrega:** 04/09

---

## 1. Qué resuelve esta entrega

El TP2 establece la base analítica y curada para la construcción del clasificador del TP3. Resuelve dos partes fundamentales:

1. **Parte Operativa (Curación y Features)**:
   - **Limpieza de datos sucios**: se descartaron 235 búsquedas con 0 personas, noches inconsistentes ($\le 0$ o $> 30$), 0 habitaciones o precios negativos.
   - **Normalización a unidad común**:
     $$\text{price\_std} = \frac{\text{avg\_price\_average}}{\text{noches} \times \text{habitaciones} \times (\text{adultos} + \text{niños})}$$
   - **Winsorización**: percentil 99.9 ($1.416 USD/noche/hab/persona) para que tarifas atípicas o errores no deformen los desvíos estándar.
   - **Features temporales**: día de check-in (`day_of_week`), mes (`month`), semana del mes (`week_in_month`), año (`year`) y duración de estadía (`stay_duration`: corta 1-2n, media 3-5n, larga >5n).
   - **Mapping geográfico**: se agruparon ~26.000 nombres crudos en destinos canónicos mediante `destination_with_nearest.csv`. Cubre el **82.3% de la demanda** de búsquedas.
   - **Categoría de hotel proxy (`price_bucket`)**: se segmentaron los hoteles en Económico (`low`), Medio (`medium`) y Premium (`high`) según cuartiles de precio por destino.

2. **Parte Analítica (Mercado y Detección de Ofertas)**:
   - **Definición de mercado justificada**: $\text{Mercado} = \text{destination\_final} \times \text{month} \times \text{week\_in\_month} \times \text{stay\_duration}$.
   - Retiene el **77% de la demanda** en celdas con $N \ge 30$ observaciones ponderadas.
   - **Distribución de precios**: asimétrica hacia la derecha (sesgo positivo) con colas largas de lujo y coeficientes de variación muy dispares entre temporadas.
   - **Comparativa de alternativas de detección**: se evaluaron cinco alternativas estadísticas y se diseñó un **detector híbrido** superior que supera las limitaciones del Z-score lineal del docente.

---

## 2. La decisión central: Definición de Mercado

$$\mathbf{\text{Mercado}} = \mathbf{\text{destination\_final} \times \text{month} \times \text{week\_in\_month} \times \text{stay\_duration}}$$

### Por qué esta segmentación y no otra:
- **Destino canónico**: agrupa ciudades satélite y suburbios con su polo metropolitano, evitando dispersión artificial.
- **Mes**: aisla la estacionalidad climática y los picos de vacaciones (enero/julio/diciembre).
- **Semana del mes**: captura el efecto de quincenas de cobro y fines de semana largos.
- **Duración de estadía**: las cadenas hoteleras aplican descuentos por volumen; comparar un fin de semana de 1 noche contra 14 noches distorsionaba el precio diario.
- **Masa estadística verificada**: de 119.253 combinaciones posibles, las celdas densas ($N \ge 30$) concentran 1.67 millones de búsquedas (77.0% del volumen total).

---

## 3. Resultados de la auditoría y curación

### Embudo de datos
| Etapa | Registros | % Retenido |
|---|---|---|
| Datos crudos (`sample_data_300k.csv.gz`) | 300.000 | 100.00% |
| Denominadores válidos y noches $\le 30$ | 299.958 | 99.98% |
| Precios válidos ($> 0$) y winsorizados | 299.765 | 99.92% |

### Auditoría residual (valores inválidos finales = 0)
- `precio_std <= 0`: 0
- `noches <= 0`: 0
- `habitaciones <= 0`: 0
- `ocupacion <= 0`: 0
- `noches > 30`: 0

### Cobertura del mapping geográfico
- **69.9%** de las filas mapeadas a destino canónico.
- **82.3%** de la demanda de búsquedas ponderada cubierta.
- Las ciudades no mapeadas (30.1% de filas, 17.7% de demanda) se conservan con su nombre crudo para no perder registros.

### Categorías de hotel proxy (`price_bucket`)
- **Budget / Económico (`low`)**: 25.6% de las búsquedas (precio promedio: $14.26 USD).
- **Mid-Range / Estándar (`medium`)**: 47.4% de las búsquedas (precio promedio: $38.51 USD).
- **Premium / Lujo (`high`)**: 27.0% de las búsquedas (precio promedio: $122.00 USD).

---

## 4. Comparativa de detectores: ¿qué método adoptamos para el TP3?

Se evaluaron 118.160 observaciones de mercados con masa estadística ($N \ge 30$). Los mercados de 1 o 2 registros no entran: ahí los percentiles colapsan y los z-scores dependen de la salvaguarda, así que medirían ruido y no calidad de detección.

| Método | Filas marcadas | % Detectado | Precio promedio (USD) | Diagnóstico analítico |
|---|---|---|---|---|
| **Detector Híbrido ($z_{\log} < -1.0$ & Ahorro $\ge 20\%$) [ADOPTADO]** | **16.684** | **14.12%** | **$16.16** | **El más efectivo: neutraliza la cola de lujo y asegura ahorro tangible en dólares.** |
| Z-Score Log-Normal ($z_{\log} < -1.0$) [Alternativa 1] | 16.837 | 14.25% | $16.33 | Simetriza la distribución, pero incluye un pequeño grupo sin descuento real. |
| Z-Score Gaussiano ($z < -1.0$) [Docente] | 10.561 | 8.94% | $15.81 | Subestima ofertas: en plazas heterogéneas los hoteles de lujo inflan $\sigma$ y causan falsos negativos. |
| Z-Score Good Price ($z < -0.5$) [Docente] | 37.107 | 31.40% | $24.84 | Buen precio general, no necesariamente una anomalía extrema de precio. |
| Descuento $\ge 30\%$ s/ Mediana [Alternativa 2] | 30.083 | 25.46% | $21.18 | Muy interpretable comercialmente, pero corta un porcentaje fijo sin considerar la dispersión del mercado. |
| Percentil 10 contextual | 16.924 | 14.32% | $18.73 | Forzado: etiqueta una cuota fija incluso en mercados homogéneos sin ofertas reales. |
| IQR inferior ($Q_1 - 1.5 \cdot IQR$) | 1.080 | 0.91% | $24.43 | Casi no se activa: con sesgo positivo la cerca inferior cae por debajo del precio mínimo del mercado. |

---

## 5. Decisión final y articulación con el TP3

A partir de la evidencia empírica, **se descartan las fórmulas ingenuas y se adopta el Detector Híbrido** como Ground Truth para el TP3:

$$\mathbf{\text{is\_deal}} = (z_{\log} < -1.0) \;\land\; \left(1 - \frac{\text{price\_std}}{\text{mediana}_{\text{contexto}}} \ge 0.20\right)$$

### Por qué esta elección:
1. **Frente al Z-Score Gaussiano del docente:**  
   En mercados como Las Vegas o Cancún, hoteles de lujo de $800 USD inflan la desviación estándar muestral ($\sigma$). Cuando $\sigma$ se infla, tarifas genuinamente baratas de $25 USD obtienen un $z = -0.87$ y quedan descartadas como ofertas (falso negativo). Al calcular el Z-score en escala logarítmica ($z_{\log}$), la dispersión se mide de forma proporcional y se elimina este sesgo.
2. **Frente al Percentil 10:**  
   El percentil fuerza que exactamente el 10% de cualquier destino sea oferta, incluso en plazas de moteles con tarifas planas idénticas donde no existe ninguna oportunidad real.
3. **Rol de la confirmación sobre la mediana ($\ge 20\%$):**  
   Filtra ofertas matemáticas en mercados de bajísima dispersión donde un hotel cuesta $40 en vez de $41 (ahorro de $1 USD). Exige que haya una rebaja tangible para el viajero.
4. **Cómo se utilizará en el TP3:**  
   - Esta regla define la etiqueta objetivo binaria `is_deal`.
   - En el TP3 se entrenarán modelos supervisados (LightGBM, XGBoost, Random Forest) sobre las variables de contexto (`day_of_week`, `month`, `stay_duration`, `price_bucket`, `destination_final`) para predecir si una tarifa es una oferta antes de que finalice la ventana de búsqueda.
   - Para celdas con muestra baja ($N < 30$), se mantendrá la cascada de rescate jerárquica:
     $$\text{Semana exacta} \;\longrightarrow\; \text{Mes completo} \;\longrightarrow\; \text{Destino global}$$

---

## 6. Archivos del paquete

| Archivo | Rol |
|---|---|
| [`TP2_curacion_mercado.ipynb`](TP2_curacion_mercado.ipynb) | Notebook ejecutable con todas las celdas, salidas, gráficos y tablas generadas. |
| [`TP2_curacion_mercado.py`](TP2_curacion_mercado.py) | Script gemelo en formato Jupytext (`py:percent`), sincronizado con el `.ipynb`. |
| [`README.md`](README.md) | Informe técnico con la justificación de mercado y la decisión de detección para el TP3. |

---

## 7. Cómo ejecutar

### En Jupyter / VS Code
Abrir [`TP2_curacion_mercado.ipynb`](TP2_curacion_mercado.ipynb) y ejecutar las celdas. Ya tiene todos los outputs y gráficos guardados.

### Desde la terminal
```bash
# Ejecutar script completo
python TP2/TP2_curacion_mercado.py

# Sincronizar cambios entre .py e .ipynb
jupytext --sync TP2/TP2_curacion_mercado.ipynb
```
