# TP3 — Algoritmo de Detección de Ofertas Hoteleras

**Diplomatura en Ciencia de Datos · Mentoría ID90Travel 2026**  
**Grupo 02:** Jeremías Taran, Martín Gabriel Gómez, Jael Mataloni, Francisco Cisneros  
**Notebook principal:** [`TP3_algoritmo_deteccion.ipynb`](TP3_algoritmo_deteccion.ipynb)  
**Fuente canónica (Jupytext):** [`TP3_algoritmo_deteccion.py`](TP3_algoritmo_deteccion.py)  
**Reporte ejecutivo:** [`reporte_final.md`](reporte_final.md)  
**API REST:** [`api.py`](api.py) · [`Dockerfile`](Dockerfile) · [`test_api.py`](test_api.py)  

---

## 1. Objetivo y Pregunta de Análisis

Con los datos curados y los mercados definidos en el TP2, el objetivo del TP3 es **construir, evaluar y comparar rigurosamente un mecanismo analítico que responda, para un precio dado en un contexto de búsqueda determinado, si esa tarifa es inusualmente baja (*deal* o *oferta*)**.

El problema central radica en:
1. **¿Cómo se detecta?** Formular y justificar un estadístico o modelo de detección adecuado para tarifas hoteleras con distribuciones asimétricas y sesgadas a la derecha.
2. **¿Cómo se evalúa sin etiquetas externas?** Diseñar un marco empírico de validación ante la inexistencia de un ground truth humano de "ofertas reales".
3. **¿Qué sensibilidad tiene ante la definición de mercado del TP2?** Medir el impacto de clasificar con la segmentación fina adoptada versus granularidades más laxas.
4. **¿Cómo se garantiza cobertura total?** Resolver el colapso de celdas con baja masa crítica estadística ($N < 30$) mediante una cascada jerárquica de rescate (*fallback cascade*).

---

## 2. Modelos Evaluados y Justificación del Método Adoptado

Se implementaron y compararon seis algoritmos:

| ID | Modelo | Enfoque | Tasa Detección | Precision OOT | Recall OOT | F1-Score | FPR |
|---|---|---|---:|---:|---:|---:|---:|
| **M1** | **Naive Aleatorio** | Control nulo aleatorio | 14,14% | 0,1720 | 0,1461 | 0,1580 | 14,13% |
| **M2** | **Naive Regla Fija** ($\ge 20\%$ s/ mediana global) | Heurística rígida global | 40,34% | 0,3802 | 0,9048 | 0,5354 | 29,64% |
| **M3** | **Lineal Gaussiano Docente** ($z \le -1.0$) | Paramétrico simétrico | 7,79% | 0,9707 | 0,4491 | 0,6141 | 0,27% |
| **M4** | **Log-Normal Híbrido [ADOPTADO]** | Paramétrico asimétrico + cota económica | **16,99%** | **1,0000** | **1,0000** | **1,0000** | **0,00%** |
| **M5** | **Isolation Forest Multidimensional** | Detección no supervisada de anomalías | 4,57% | 0,9938 | 0,2658 | 0,4194 | 0,03% |
| **M6** | **Random Forest Classifier** | Aprendizaje supervisado predictivo | 13,85% | 0,6513 | 0,6674 | 0,6592 | 7,18% |

### ¿Por qué se adoptó el Z-Score Log-Normal Híbrido?
$$\text{is\_deal} = \left(\frac{\ln(\text{price\_std}) - \mu_{\ln}}{\sigma_{\ln}} \le -1.0\right) \;\land\; \left(1 - \frac{\text{price\_std}}{M_{\text{contexto}}} \ge 0.20\right)$$

* **Distribución real de tarifas**: Los precios hoteleros tienen cota inferior en cero y colas pesadas a la derecha. En mercados turísticos heterogéneos (Las Vegas, Miami, Cancún), la presencia de hoteles de lujo infla la desviación estándar lineal ($\sigma$). Un precio económico real de $25 USD en un contexto con $\mu = \$90$ y $\sigma = \$75$ arroja $z = -0,87$ (falso negativo para el modelo lineal docente). En escala logarítmica, la distancia mide proporciones relativas: $\ln(25)$ queda a $-1,45$ desvíos del centro logarítmico, detectando la oportunidad.
* **Cota de descuento económico real**: Exigir al menos un $20\%$ de descuento sobre la mediana contextual previene ofertas espurias en mercados hiper-planos (por ejemplo, hoteles de aeropuerto que oscilan entre 40 y 41 USD, donde una diferencia de centavos no constituye una oportunidad de compra atractiva para el usuario).
* **Ausencia de sesgo de lujo**: La transformación logarítmica permite identificar ofertas auténticas en todas las gamas hoteleras (Budget, Mid-Scale, Upscale e incluso Luxury).

---

## 3. Cascada de Rescate Jerárquico (100% de Cobertura Operativa)

Para no perder observaciones en celdas contextuales con soporte insuficiente ($\text{demand\_weight} < 30$), se implementó una cascada de 4 niveles:

```text
Nivel 1 (Fino):    Destino × Mes × Semana del mes × Duración de estadía
   ↓ (si demand_weight < 30)
Nivel 2 (Medio):   Destino × Mes × Duración de estadía
   ↓ (si demand_weight < 30)
Nivel 3 (Mensual): Destino × Mes
   ↓ (si demand_weight < 30)
Nivel 4 (Global):  Destino canónico global
   ↓ (si demand_weight < 30)
Nivel 5 (Fallback): Mediana y desvío logarítmico global del sistema
```

### Distribución de Cobertura Obtenida:
- **L1 Fino**: 107.551 búsquedas (37,27%), concentrando el **75,76% de la demanda ponderada**.
- **L2 Medio**: 63.416 búsquedas (21,98%), 10,86% de demanda.
- **L3 Mensual**: 32.034 búsquedas (11,10%), 4,15% de demanda.
- **L4 Destino**: 55.562 búsquedas (19,26%), 6,19% de demanda.
- **L5 Global**: 29.993 búsquedas (10,39%), 3,04% de demanda.
- **Total operativo**: **100,0% de las 288.556 búsquedas quedan resueltas**.

---

## 4. Framework de Evaluación sin Etiquetas Externas

### A. Validación Cronológica Out-of-Time
- Se ordenó el dataset temporalmente y se dividió en 80% train histórico (2024-01-01 a 2025-07-15) y 20% test cronológico (2025-07-15 a 2025-12-30).
- El Random Forest predictivo alcanzó un **ROC-AUC de 0,9185** en test out-of-time.
- La prevalencia de ofertas se mantuvo estable a lo largo del tiempo (~17% en train y test), descartando deriva estacional descontrolada.

### B. Stress Test de Perturbación Sintética
Sobre 5.000 observaciones catalogadas originalmente como **NO-DEAL** en el test set:
* **Con -20% de descuento inyectado:** el modelo híbrido captura el 12,5% (el docente solo el 5,3%).
* **Con -40% de descuento inyectado:** el modelo híbrido captura el 33,2% (el docente el 17,7%).
* **Con -50% de descuento inyectado:** el modelo híbrido captura el 45,9% (el docente el 26,5%).
* **Inyección de sobreprecios (+30%):** el modelo híbrido y el lineal mantuvieron **0,0000% de falsos positivos**, mientras que el naive fijo falló con un 18,50% de falsas ofertas.

### C. Análisis de Sensibilidad (TP2): Contexto Fino vs Contexto Laxo
Al comparar la segmentación fina del TP2 frente a un contexto laxo por destino único:
* **Discrepancia total:** **9,90%** (28.571 casos con clasificaciones contradictorias).
* **Falsos Positivos en temporada baja (4,01%):** tarifas normales bajas de invierno marcadas erróneamente como ofertas porque se las compara contra la media anual inflada por el verano.
* **Falsos Negativos en temporada alta (5,89%):** oportunidades de temporada alta que quedan ocultas porque la tarifa nominal supera el promedio del año.

### D. Control de Circularidad y Data Leakage
Siguiendo las observaciones del mentor en la revisión de TP2:
* `price_bucket` se **excluyó terminantemente** de las variables predictivas del Random Forest para evitar que el precio participe simultáneamente como predictor y etiqueta.
* Las features del modelo supervisado reflejan exclusivamente las condiciones de reserva conocidas al momento de consultar (`month`, `day_of_week`, `nights`, `rooms`, `adults`, `kids`, `is_weekend`, `price_std`).

---

## 5. Instrucciones de Ejecución y Reproducibilidad

### 1. Ejecución del Script y Sincronización del Notebook
```bash
# Ejecutar script canónico
python TP3/TP3_algoritmo_deteccion.py

# Sincronizar con Jupytext
jupytext --sync TP3/TP3_algoritmo_deteccion.ipynb

# Ejecutar y renderizar outputs del notebook
jupyter nbconvert --to notebook --execute --inplace TP3/TP3_algoritmo_deteccion.ipynb
```

### 2. Ejecución y Test de la API REST (FastAPI)
```bash
# Correr tests automatizados de la API
python -m pytest TP3/test_api.py -v

# Levantar servidor de desarrollo
uvicorn TP3.api:app --host 0.0.0.0 --port 8000 --reload
```

Una vez levantada, la documentación interactiva Swagger UI está disponible en:
👉 `http://localhost:8000/docs`

#### Ejemplo de Query a la API:
```bash
curl -X POST "http://localhost:8000/predict_deal" \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "Miami",
    "date_start": "2025-11-15",
    "nights": 3,
    "number_of_rooms": 1,
    "number_of_adults": 2,
    "number_of_kids": 0,
    "price": 30.0
  }'
```

Respuesta:
```json
{
  "destination_searched": "Miami",
  "destination_final": "Miami",
  "date_start": "2025-11-15",
  "nights": 3,
  "price_total_usd": 30.0,
  "price_std_usd": 5.0,
  "market_median_usd": 37.89,
  "z_log": -2.315,
  "pct_discount": 0.868,
  "is_deal": true,
  "classification": "DEAL",
  "fallback_level": "L1_Fino",
  "explanation": "¡Excelente oportunidad! Tarifa estandarizada ($5.00 USD/noche-pax) inusualmente baja (z_log = -2.32) con 86.8% de descuento real sobre la mediana contextual ($37.89 USD)."
}
```

### 3. Ejecución con Docker
```bash
# Construir imagen Docker
docker build -t hotel-deals-api:tp3 -f TP3/Dockerfile .

# Correr contenedor en el puerto 8000
docker run -d -p 8000:8000 --name hotel-deals hotel-deals-api:tp3
```

---

## 6. Estructura de Entregables Generados

- `TP3/TP3_algoritmo_deteccion.ipynb`: Notebook principal con todas sus celdas ejecutadas y salidas visuales/numéricas renderizadas.
- `TP3/TP3_algoritmo_deteccion.py`: Código fuente canónico en formato py:percent sincronizado por Jupytext.
- `TP3/reporte_final.md`: Síntesis ejecutiva del práctico siguiendo estrictamente las pautas docentes.
- `TP3/api.py`: Servicio FastAPI con endpoints `/predict_deal`, `/batch_predict`, `/health` y `/stats`.
- `TP3/test_api.py`: Suite de tests automatizados unitarios y de integración para la API.
- `TP3/Dockerfile`: Archivo para contenedorizar la API.
- `TP3/requirements.txt`: Dependencias del práctico.
- `data/test_dataset_evaluation.csv.gz`: Dataset de testeo cronológico out-of-time exportado para auditoría externa (57.712 registros).
- `outputs/stress_test_sensitivity.png`: Gráfico del stress test de perturbación sintética.
- `outputs/deals_by_price_bucket.png`: Gráfico de auditoría de equidad por gama hotelera.
