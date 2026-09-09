# TP4 — Presentación Final y Resumen Ejecutivo del Proyecto

**Diplomatura en Ciencia de Datos · Proyecto Final de Mentoría 2026**  
**ID90Travel — Sistema de Detección de Ofertas Hoteleras**  
**Grupo 02:** Francisco Cisneros  

---

## 1. Problema de Negocio
En plataformas de viajes como ID90Travel, los usuarios buscan alojamiento sin saber si la tarifa que observan representa una oportunidad genuina, un valor habitual o un precio inflado.  
El objetivo del proyecto fue desarrollar un sistema capaz de responder automáticamente:
> **¿La tarifa observada para este destino, fechas y grupo de viaje constituye una oferta inusualmente conveniente frente a su contexto histórico?**

---

## 2. Definición de Mercado y Contexto
Un precio no existe en el vacío: comparar un hotel en Las Vegas un viernes de diciembre contra uno un martes de febrero desvirtúa la señal de precio.  
Se definió el **contexto de comparación** mediante cuatro dimensiones ortogonales:
$$\text{Contexto} = \text{destination\_final} \times \text{month} \times \text{week\_in\_month} \times \text{stay\_duration}$$

* **Normalización geográfica:** Se mapearon más de 26.000 nombres de ciudad crudos a destinos canónicos homogéneos usando `destination_with_nearest.csv` y proximidad geográfica.
* **Métrica base homogénea:** Se normalizó la tarifa al precio estándar por habitación-noche-persona:
  $$\text{price\_std} = \frac{\text{avg\_price\_average}}{\text{nights} \times \text{rooms} \times (\text{adults} + 0.5 \times \text{kids})}$$

---

## 3. Decisiones Principales de Limpieza y Curación (TP1 y TP2)
1. **Eliminación de anomalías de ingreso:** Registros con noches $\le 0$, habitaciones $\le 0$, adultos $= 0$ o tarifas fuera del rango operativo $[5, 2500]$ USD fueron filtrados.
2. **Tratamiento del sesgo positivo:** La distribución de tarifas hoteleras no es gaussiana: tiene cota en cero y cola larga a la derecha. Los hoteles de lujo inflan el desvío estándar lineal ($\sigma$), generando falsos negativos en el Z-score tradicional. Se adoptó la modelización log-normal.
3. **Masa crítica y cascada de fallbacks:** Para evitar estimaciones ruidosas en celdas con pocas búsquedas, se exigió un umbral mínimo de $N \ge 30$. Las búsquedas en celdas menores recurren a la cascada: semana del mes $\to$ mes completo $\to$ destino global.

---

## 4. Algoritmo de Detección Adoptado (TP3)
Se seleccionó el **Z-Score Log-Normal Híbrido**:
$$\text{is\_deal} = \left(\frac{\ln(\text{price\_std}) - \mu_{\ln}}{\sigma_{\ln}} \le -1.0\right) \;\land\; \left(1 - \frac{\text{price\_std}}{\text{mediana}_{\text{contexto}}} \ge 0.20\right)$$

* **Tasa de detección:** 13.92% de las búsquedas en mercados consolidados, frente al 6.38% del Z-Score Gaussiano.
* **Garantía de ahorro real:** Exige un descuento mínimo del 20% respecto a la mediana del mercado para evitar marcar como oferta tarifas con diferencias de centavos en mercados de baja dispersión.
* **Modelo predictivo complementario:** Se entrenó un Random Forest sobre variables de búsqueda alcanzando un **ROC-AUC de 0.985**, lo que permite predecir la probabilidad de deal en tiempo real sin recalcular la base histórica en cada petición.

---

## 5. Resultados de Validación y Métricas
* **Stress Test Sintético:** Ante descuentos artificiales del 40%, el modelo captura el **94.6%** de las ofertas inyectadas (vs 68.3% del modelo lineal).
* **Resistencia a Falsos Positivos:** Al inyectar sobreprecios del +30%, la tasa de falsos positivos fue del **0.0000%**.
* **Estabilidad Out-of-Time:** Evaluación cronológica demostró consistencia entre semestres sin drift perjudicial.

---

## 6. Recursos de la Entrega
* **Diapositivas y Guion de Locución:** [`presentacion_final.md`](presentacion_final.md) (formato Marp con notas de orador y speech exacto para el video).
* **Enlace de la Presentación en Video:** [`link_presentacion.md`](link_presentacion.md).
* **Notebooks del Proyecto:**
  * TP1: [`../TP1_corregido/TP1_exploracion_mercado.ipynb`](../TP1_corregido/TP1_exploracion_mercado.ipynb)
  * TP2: [`../TP2/TP2_curacion_mercado.ipynb`](../TP2/TP2_curacion_mercado.ipynb)
  * TP3: [`../TP3/TP3_algoritmo_deteccion.ipynb`](../TP3/TP3_algoritmo_deteccion.ipynb)
