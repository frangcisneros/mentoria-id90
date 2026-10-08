# Plan de Trabajo — TP3: Algoritmo de Detección de Ofertas Hoteleras

**Diplomatura en Ciencia de Datos · Mentoría ID90Travel 2026**  
**Proyecto:** Detección de anomalías de precio y clasificación de ofertas hoteleras

---

## 1. Contexto y Objetivos

El objetivo del Trabajo Práctico 3 es construir y evaluar un mecanismo analítico capaz de responder, para un precio observado en un contexto de búsqueda determinado, si dicha tarifa es inusualmente baja (oferta o *deal*).

### Consigna oficial
- **Enfoque metodológico:** Libre (estadístico, supervisado, no supervisado o híbrido), priorizando la fundamentación conceptual, supuestos y limitaciones por sobre la complejidad algorítmica.
- **Evaluación sin ground truth externo:** Diseñar e implementar métricas que permitan evaluar el desempeño sin etiquetas preexistentes de "oferta real".
- **Sensibilidad a la segmentación:** Cuantificar el impacto de la definición de mercado adoptada en el TP2 sobre la clasificación de ofertas.
- **Entregables:** Implementación reproducible, reporte de evaluación, análisis de sensibilidad y conclusiones técnicas honestas.

---

## 2. Auditoría del Código Base y Correcciones Requeridas

A partir de la revisión del repositorio y las entregas de TP1 y TP2, se identificaron dos inconsistencias conceptuales en el borrador preliminar (`TP3/TP3_algoritmo_deteccion.py`) que deben corregirse:

1. **Cálculo de `price_std` (Normalización):**
   - *Inconsistencia previa:* Se calculó `adultos + 0.5 * niños` para la capacidad de pasajeros.
   - *Alineación canónica:* El estándar validado en TP1/TP2 y en `auxiliary_functions.py` es:
     $$\text{price\_std} = \frac{\text{avg\_price\_average}}{\text{nights} \times \text{number\_of\_rooms} \times (\text{number\_of\_adults} + \text{number\_of\_kids})}$$

2. **Masa Crítica y Cascada Jerárquica de Fallbacks:**
   - *Inconsistencia previa:* Se filtraron celdas usando `n_records >= 30` sin ponderar y descartando celdas ralas (`inner join`), lo que pierde más del 80% del volumen de búsquedas.
   - *Alineación canónica:* Evaluar masa estadística considerando `demand_weight` (suma de `count_repeated`) e implementar la **cascada jerárquica de rescate** (Semana del mes $\to$ Mes completo $\to$ Destino global) para observaciones con $N < 30$.

---

## 3. Arquitectura de Modelos a Comparar

Se compararán cuatro enfoques complementarios:

1. **Modelo 1 — Z-Score Gaussiano Lineal (Baseline Docente):**
   - $Z = \frac{x - \mu}{\sigma} \le -1.0$.
   - *Supuesto:* Distribución normal simétrica de tarifas.
   - *Limitación a demostrar:* En mercados heterogéneos, los hoteles de lujo inflan $\sigma$, desplazando el umbral y generando falsos negativos.

2. **Modelo 2 — Z-Score Log-Normal Híbrido (Propuesta Adoptada):**
   - $\text{is\_deal} = \left(\frac{\ln(\text{price\_std}) - \mu_{\ln}}{\sigma_{\ln}} \le -1.0\right) \;\land\; \left(1 - \frac{\text{price\_std}}{\text{mediana}_{\text{contexto}}} \ge 0.20\right)$.
   - *Ventaja:* Mide dispersión en escala relativa (porcentual) y exige un descuento real tangible ($\ge 20\%$) para evitar falsas ofertas por diferencias mínimas de centavos.

3. **Modelo 3 — Isolation Forest Multidimensional (No Supervisado):**
   - Detección de anomalías en el espacio $(\text{z\_log}, \text{pct\_discount})$.
   - *Ventaja:* No asume una distribución paramétrica a priori; evalúa aislamiento en múltiples dimensiones.

4. **Modelo 4 — Random Forest Classifier (Supervisado / Predictivo):**
   - Entrenado sobre variables del contexto de búsqueda (`month`, `day_of_week`, `nights`, `rooms`, `adults`, `kids`, `price_std`, `is_weekend`).
   - *Utilidad:* Permite estimar probabilidad de deal en tiempo real o en escenarios de baja cobertura histórica (*cold-start*).

---

## 4. Framework de Evaluación sin Etiquetas Externas

Para validar los clasificadores en ausencia de un *ground truth* humano:

1. **Validación Cronológica Out-of-Time (OOT):**
   - División temporal del dataset (80% train histórico / 20% test cronológico).
   - Verificación de estabilidad en la tasa de ofertas detectadas a lo largo de las fechas para descartar deriva estacional o sobreajuste.

2. **Stress Test de Perturbación Sintética:**
   - Muestreo de registros catalogados como *NO-DEAL*.
   - **Inyección de descuentos artificiales:** Aplicar rebajas controladas ($10\%, 20\%, 30\%, 40\%, 50\%$) para medir el *Recall* empírico y la capacidad de reacción de cada modelo.
   - **Inyección de sobreprecios:** Aplicar incrementos artificiales ($+30\%$) para auditar la resistencia ante falsos positivos (la tasa debe ser estrictamente 0%).

3. **Auditoría de Coherencia Económica:**
   - Análisis de la distribución de precios clasificados como ofertas dentro de los diferentes niveles de la categoría proxy (`price_bucket`: Low, Medium, High).

---

## 5. Análisis de Sensibilidad frente a la Granularidad de Mercado

Comparar las clasificaciones obtenidas con:
- **Contexto Fino (TP2):** $\text{Destino} \times \text{Mes} \times \text{Semana} \times \text{Duración}$.
- **Contexto Laxo:** $\text{Destino}$ global (sin temporización ni duración).

### Métricas de sensibilidad:
- Matriz de concordancia y porcentaje de discrepancia.
- Documentación de patologías:
  - *Falsos positivos en temporada baja:* Tarifas habituales deprimidas catalogadas como oferta por compararse contra la media anual inflada por el verano.
  - *Falsos negativos en temporada alta:* Tarifas competitivas en picos de demanda no detectadas porque superan la media anual histórica.

---

## 6. Fases de Ejecución

1. **Fase 1 — Refactor y Curación:** Corregir y sincronizar `TP3/TP3_algoritmo_deteccion.py` con las funciones de `auxiliary_functions.py` y las definiciones de TP2.
2. **Fase 2 — Ejecución y Generación de Resultados:** Correr los experimentos, stress tests y matriz de sensibilidad generando gráficos y métricas reproducibles.
3. **Fase 3 — Sincronización Jupytext:** Sincronizar el script con `TP3_algoritmo_deteccion.ipynb`.
4. **Fase 4 — Paquete de Entrega:** Crear `TP3_entrega/` con su propio `README.md`, `reporte_final.md` ejecutivo, datos requeridos y comprimir en `TP3_entrega.zip`.
