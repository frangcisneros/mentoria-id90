# TP3 — Algoritmo de Detección de Ofertas Hoteleras

**Diplomatura en Ciencia de Datos · Mentoría ID90Travel 2026**  
**Grupo 02:** Francisco Cisneros  
**Notebook principal:** [`TP3_algoritmo_deteccion.ipynb`](TP3_algoritmo_deteccion.ipynb) (fuente canónica: [`TP3_algoritmo_deteccion.py`](TP3_algoritmo_deteccion.py))

---

## 1. Método Elegido y Justificación

Se evaluaron cuatro alternativas sobre el dataset curado:
1. **Z-Score Gaussiano Lineal (docente):** $z = (x - \mu)/\sigma \le -1.0$.
2. **Z-Score Log-Normal Híbrido (adoptado):**
   $$\text{is\_deal} = \left(\frac{\ln(\text{price\_std}) - \mu_{\ln}}{\sigma_{\ln}} \le -1.0\right) \;\land\; \left(1 - \frac{\text{price\_std}}{\text{mediana}_{\text{contexto}}} \ge 0.20\right)$$
3. **Isolation Forest Multidimensional (no supervisado):** aislamiento de anomalías en el espacio $(\text{z\_log}, \text{pct\_discount})$.
4. **Random Forest Classifier (supervisado):** clasificador entrenado con variables de búsqueda para estimar probabilidad de deal en tiempo real ($ROC\text{-}AUC = 0.985$).

### ¿Por qué se adoptó el Z-Score Log-Normal Híbrido?
* **Distribución real de precios:** Los precios hoteleros tienen cota inferior en cero y cola pesada a la derecha. En plazas heterogéneas (Las Vegas, Miami, Cancún), la presencia de resorts de lujo infla artificialmente la desviación estándar lineal ($\sigma$). Un precio económico real de 25 USD en un contexto con $\mu = 90$ y $\sigma = 75$ arroja $z = -0.87$ (falso negativo). En escala logarítmica, la distancia mide proporciones relativas: $\ln(25)$ queda a $-1.28$ del centro logarítmico, detectando la oferta con solidez.
* **Cobertura económica equilibrada:** Detecta el **13.92%** de las tarifas frente al **6.38%** del modelo gaussiano lineal.
* **Cota de descuento obligatoria:** La condición de ahorro $\ge 20\%$ sobre la mediana previene falsas ofertas en mercados con varianza casi nula (por ejemplo, hoteles de aeropuerto que oscilan entre 40 y 42 USD).

---

## 2. Supuestos del Algoritmo

1. **Unidad homogénea de comparación:** La tarifa base debe estar normalizada a `price_std` (USD por habitación-noche-persona). Comparar tarifas brutas mezcla el tamaño del grupo con el valor del alojamiento.
2. **Masa crítica estadística:** La celda contextual debe contener al menos **$N \ge 30$ observaciones históricas**. En celdas con $N < 30$, se activa la cascada de rescate jerárquica (semana $\to$ mes $\to$ destino global).
3. **Estabilidad temporal intra-mes:** Se asume que las tarifas dentro de una misma semana del mes comparten condiciones similares de oferta hotelera y demanda turística.

---

## 3. Evaluación sin Etiquetas Externas (Ground Truth Inexistente)

Al no existir etiquetas humanas de "esto fue una oferta real", la validación se realizó mediante tres técnicas:

### A. Validación Cronológica Out-of-Time
Se ordenó el dataset temporalmente y se dividió en 80% train / 20% test cronológico.
* El clasificador supervisado alcanzó un **ROC-AUC de 0.9853**, con **Precision de 87.5%** y **Recall de 83.0%** en la detección de las ofertas marcadas por el criterio log-normal híbrido.
* La tasa de ofertas se mantuvo estable a lo largo del tiempo (~13.9% en train y test), descartando deriva estacional descontrolada.

### B. Stress Test de Perturbación Sintética (Recall y Falsos Positivos)
Se tomó una muestra de 5.000 observaciones catalogadas como NO-DEAL y se les inyectaron descuentos forzados:
* **Con 20% de descuento inyectado:** el modelo híbrido captura el 48.2% de los casos; el docente solo el 22.1%.
* **Con 40% de descuento inyectado:** el modelo híbrido captura el 94.6% de los casos; el docente llega al 68.3%.
* **Inyección de precios inflados (+30%):** ambos modelos registraron **0.0000% de falsos positivos**.

---

## 4. Sensibilidad frente a la Definición de Mercado y Contexto

Se contrastó el **Contexto Fino** ($\text{Destino} \times \text{Mes} \times \text{Semana} \times \text{Duración}$) contra el **Contexto Laxo** ($\text{Destino}$ únicamente):
* Hubo una **discrepancia del 6.05%** en las clasificaciones.
* En **temporada baja**, el contexto laxo genera falsos positivos: tarifas normales deprimidas se marcan como deals porque se las compara contra la media anual inflada por el verano.
* En **temporada alta**, el contexto laxo genera falsos negativos: buenas oportunidades de pico quedan ocultas porque superan la media histórica del año.

---

## 5. Limitaciones Abiertas

1. **Destinos de cola larga (Cold-Start):** Ciudades con menos de 30 búsquedas en todo el historial dependen del fallback a destino canónico más cercano (`nearest_destination_id`). Si la distancia geográfica no refleja similitud de mercado, la estimación pierde precisión.
2. **Eventos atípicos no programados:** Congresos masivos, recitales o finales deportivas elevan transitoriamente los precios sin constituir un cambio estructural de temporada.
3. **Inflación y tipo de cambio:** En series multianuales (2024 vs 2025), la inflación del dólar o variaciones cambiarias en destinos fuera de EE.UU. requieren indexación periódica de las medianas de referencia.
