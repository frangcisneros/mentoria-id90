---
marp: true
theme: gaia
_class: lead
paginate: true
backgroundColor: #0f172a
color: #f8fafc
style: |
  section {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    padding: 36px 52px;
    font-size: 24px;
  }
  h1, h2, h3 {
    color: #38bdf8;
  }
  table {
    font-size: 19px;
    width: 100%;
  }
  th {
    background-color: #1e293b;
    color: #38bdf8;
  }
  td {
    background-color: #0f172a;
  }
  .ok {
    color: #34d399;
    font-weight: bold;
  }
  .warn {
    color: #fbbf24;
    font-weight: bold;
  }
  footer {
    font-size: 14px;
    color: #94a3b8;
  }
---

# Sistema de Detección de Ofertas Hoteleras
### Mentoría: Sistema de detección de ofertas hoteleras · Diplomatura en Ciencia de Datos (FAMAF – UNC) 2026

**Grupo 02:** Jeremías Taran · Martín Gabriel Gómez · Jael Mataloni · Francisco Cisneros
**Mentor:** Martín

* **Pregunta de negocio:** Dado el precio de una búsqueda hotelera y la historia de su mercado, ¿estamos frente a una verdadera oferta?
* **Objetivo:** Clasificar en tiempo real (vía API + Docker) si una tarifa es inusualmente baja para su contexto, devolviendo una explicación transparente al usuario.

---

## 2. Dataset, Curación y Normalización Geográfica

| Etapa del embudo de curación (`2024–2025`) | Búsquedas | % Retenido |
|---|---:|---:|
| Búsquedas crudas totales | 5.158.190 | 100,00% |
| Denominadores válidos y estadías $\le 30$ noches | 5.158.163 | 99,99% |
| Tarifa cruda en $(0, 50.000]$ USD | 5.157.438 | 99,98% |
| Piso operativo $\text{price\_std} \ge 1$ USD + corte p99,9 (solo en train) | **5.145.974** | **99,76%** |

* **Fragmentación geográfica resuelta:** +26.000 nombres de ciudad crudos consolidados por proximidad en **2.584 destinos canónicos** (**69,9%** de filas y **82,3%** de la demanda ponderada; el resto preserva la ciudad original sin imputación forzada).
* **Desagregación temporal corregida:** expansión exclusiva por noches pagadas, sin contar el día de check-out.

---

## 3. Unidad Comparable (`price_std`) y Definición de Mercado

* **Unidad canónica:** **USD por habitación-noche-persona (`price_std`)**
* **Contexto de comparación adoptado:**
  $$\text{Mercado} = \text{Destino Canónico} \times \text{Mes} \times \text{Semana del Mes} \times \text{Duración de Estadía}$$

| Segmentación evaluada | Signal-to-Noise Ratio (SNR) | Demanda cubierta ($N \ge 30$ pond.) |
|---|---:|---:|
| Solo Destino | 0,060 | 99,9% |
| Destino × Mes × Semana | 0,191 | 95,4% |
| **Destino × Mes × Semana × Estadía** | **0,538 (9× mejora)** | **91,4%** |

* **Hallazgo crítico (pseudoreplicación):** exigir **30 búsquedas independientes** en vez de repeticiones baja la cobertura del Nivel 1 a **16,8%** en muestra (**44,2%** en full) $\Rightarrow$ **la cascada de fallback es indispensable**.

---

## 4. Cascada Jerárquica de Fallback (L1 $\to$ L6)

| Nivel de Cascada ($\ge 30$ búsquedas únicas en Train 2024) | % Consultas Test 2025 | Acumulado |
|---|---:|---:|
| **L1:** `destino × mes × semana × estadía` | 44,2% | 44,2% |
| **L2:** `destino × mes × estadía` | +23,2% | 67,4% |
| **L3:** `destino × mes` | +10,3% | 77,7% |
| **L4:** `destino × estadía` | +12,2% | 89,9% |
| **L5:** `destino` | +3,5% | 93,4% |
| **L6:** `país × mes × estadía` | +5,5% | **98,9%** |
| **L7:** Sin soporte suficiente (abstención explícita) | 1,1% | 100,0% |

![w:680](assets/cobertura_cascada.png)

---

## 5. Del Z-Score Lineal al Detector Híbrido Log-Normal

* **Por qué falla el Z-Score Gaussiano lineal en hotelería:**
  * Un mismo destino mezcla hoteles económicos y resorts de lujo $\Rightarrow$ distribución con **fuerte cola derecha**.
  * Los valores altos inflan $\sigma$: una oferta real de **25 USD** en una plaza con $\mu = 90, \sigma = 75$ da $z = -0,87$ y **no se detecta** (falso negativo).
* **Detector Híbrido adoptado (Algoritmo E):**
  $$\text{Oferta} \iff \left(z_{\log} < \text{umbral}\right) \;\land\; \left(1 - \frac{\text{price\_std}}{\text{mediana}_{\text{contexto}}} \ge 20\%\right)$$
  * Combina **anomalía estadística proporcional** en escala logarítmica con un **ahorro económico mínimo del 20%**.

---

## 6. Evaluación sin Etiquetas Externas y Anti-Leakage

1. **Split Temporal estricto:**
   * **Entrenamiento:** check-ins **2024** (2,51 M) $\to$ **Test:** check-ins **2025** (2,63 M).
   * Etiqueta de referencia: tarifas que resultaron anómalas dentro del **mercado realizado de 2025** (no visto por los detectores).
2. **Por qué no usar Split Aleatorio (in-period):**
   * En un split 80/20 dentro de 2024 el Híbrido da **F1 = 0,919**; en 2025 da **0,764 – 0,808**. Evaluar en el mismo año oculta la **deriva interanual de precios**.
3. **Blindaje Anti-Leakage:**
   * Recorte superior p99,9, baselines, codificadores y umbrales ajustados **solo en 2024**.
   * El tramo de precio (`price_bucket`) queda **excluido** como feature y dimensión (se construye con el propio precio).
4. **Pruebas con verdad por construcción:** 2.400 sintéticos + Benchmark Robusto Estratificado (540 casos).

---

## 7. Aprendizaje Automático y Comparación de Modelos (A–G)

* **Criterio de negocio:** priorizar **Precision alta y FPR $\le 5\%$** (un falso positivo destruye la confianza del usuario), seleccionando por **F1** con **cobertura $\ge 98\%$**.

| Modelo evaluado en Test Temporal 2025 | Precision (Full / Corr.) | Recall (Full / Corr.) | **F1 (Full / Corr.)** | FPR (Full / Corr.) | Cobertura |
|---|---:|---:|---:|---:|---:|
| **A.** Naive Aleatorio (15%) | 0,151 / 0,161 | 0,151 / 0,164 | 0,151 / 0,162 | 15,0% / 16,1% | 100% |
| **B.** Naive Mediana −20% | 0,375 / 0,486 | 0,965 / 0,974 | 0,540 / 0,648 | **28,5% / 19,5%** | 99,8% / 99,2% |
| **C.** Z-Score Gaussiano lineal | 0,783 / 0,880 | **0,461 / 0,355** | 0,580 / 0,506 | 2,3% / 0,9% | 99,8% / 99,2% |
| **D.** Percentil 10 contextual | 0,760 / 0,847 | 0,718 / 0,680 | 0,738 / 0,755 | 4,0% / 2,3% | 99,8% / 99,2% |
| **E. Híbrido $z_{\log}$ + Cascada (ELEGIDO)** | **0,725 / 0,762** | **0,806 / 0,861** | **0,764 / 0,808** | **5,4% / 5,1%** | **99,8% / 99,2%** |
| **F.** HistGradientBoosting (ML exógeno) | 0,756 / 0,761 | 0,785 / 0,841 | **0,770 / 0,799** | 4,5% / 5,0% | 100% |
| **G.** Random Forest (ML exógeno) | 0,713 / 0,693 | 0,743 / 0,768 | 0,728 / 0,729 | 5,3% / 6,4% | 100% |

---

## 8. Respuesta ante Descuentos Controlados y Sensibilidad

* **Curva sintética (0,3× a 1,5× de la mediana):** tanto **E (Híbrido)** como **F (HistGB)** siguen la curva verdadera; un −30% aún no alcanza por la dispersión de gamas dentro de cada ciudad.
* **Sensibilidad:** quitar mes/estadía resta **12 puntos de F1**; sin cascada la cobertura cae al **44,6%**.

![w:480](assets/respuesta_descuentos_sinteticos.png) ![w:480](assets/sensibilidad_contexto.png)

---

## 9. Decisión de Producto: ¿Por qué Híbrido (E) sobre Boosting (F)?

| Criterio de Producción | **E. Detector Híbrido + Cascada (Elegido)** | **F. HistGradientBoosting** |
|---|---|---|
| **Capacidad predictiva (F1)** | **0,764 (Full) / 0,808 (Muestra corr.)** — F1 API real: **0,899** | 0,770 (Full) / 0,799 (Muestra corr.) |
| **Explicabilidad al usuario** | **Total:** devuelve mediana, % descuento, $z_{\log}$ y nivel de cascada | Caja negra (score de probabilidad) |
| **Disponibilidad de features en query API** | **100% disponibles** en la búsqueda | Requiere `avg_hotel_count` (no viene en la query) |
| **Mantenimiento ante inflación** | Recalcular tabla de medianas (ventana móvil) | Reentrenar modelo + recalibrar umbral |

---

## 10. Radiografía de Errores: Deriva Temporal y Segmentos

* **La principal fuente de error es la deriva de precios 2024 $\to$ 2025 (no el umbral):**
  * **Falsos Positivos (FP):** mercados donde la mediana en 2025 **se abarató un 20%** (`ratio = 0,80`).
  * **Falsos Negativos (FN):** mercados donde la mediana en 2025 **se encareció un 25%** (`ratio = 1,25`).
* **Comportamiento por segmento:**
  * **Top 50 destinos:** F1 = **0,850** (vs 0,735 en cola larga y ciudades sin mapeo).
  * **Niveles de cascada:** FPR de **4,0%** en L1 vs **11%–12%** en L3/L5 (donde se mezclan estadías).
  * **Tramo de precio:** **46% de ofertas en gama baja** vs **0,1% en alta** $\to$ falta la categoría real de estrellas del hotel.

---

## 11. Desafío Clave Resuelto: Unidad de Precio (Train vs API)

* **Diagnóstico empírico (2 adultos, 1 hab.):** la mediana de la tarifa histórica casi no crece con las noches (**1n: 138 USD $\to$ 7n: 208 USD**) ni con las habitaciones (**171,8 vs 176,0 USD**), pero sí con las personas (**151 $\to$ 316 USD**).
* **El problema:** el dato histórico es **tarifa por noche por habitación**, mientras la API recibe `price_total` como **total de la estadía**. Dividir por `noches × hab` en train desinflaba los baselines ~3× y hundía el recall al recibir totales reales.
* **Solución y revalidación en el Benchmark Robusto Ciego (450 casos):**
  * **Train:** $\text{price\_std} = \text{tarifa\_noche\_hab} / \text{personas}$ · **API:** $\text{price\_std} = \text{price\_total} / (\text{noches} \times \text{hab} \times \text{personas})$
  * **Umbral recalibrado** (máx. F1 con FPR $\le 5\%$): $z_{\log} < -0,875$ y descuento $\ge 20\%$.

| Versión en Benchmark Robusto Ciego (450 casos) | Accuracy | Precision | **Recall (`deal`)** | **F1** | FPR |
|---|---:|---:|---:|---:|---:|
| **Antes** (escala desfasada, mejor caso sin ajuste) | 0,722 | 0,824 | 0,389 | **0,528** | 5,6% |
| **Después (convención unificada + recalibración)** | **0,820** | **0,896** | **0,622** | **0,734** | **4,8%** |

---

## 12. Conclusión Ejecutiva y Hoja de Ruta

* **Veredicto de producción:** **Listo para desplegar como MVP / Servicio en Tiempo Real**
  * Microservicio **FastAPI + Docker** operativo (`/health`, `/evaluate`, `/evaluate/batch`).
  * **99% de cobertura**, **FPR < 5%**, justificación auditable en cada respuesta y **11/11 tests** de contrato, unidad y casos borde superados.
* **Próximos pasos recomendados:**
  1. **Baselines con ventana móvil (90–180 días):** corrige la deriva interanual sin reentrenar modelos.
  2. **Sumar categoría real del hotel (estrellas / tier):** habilita detectar ofertas en hoteles 4★ y 5★.
  3. **Simplificar L1 a `destino × mes × estadía` (sin semana del mes):** evita sobre-segmentar (+1,1 pts de F1).
  4. **Monitoreo de tasa de ofertas marcadas por destino:** alerta automática ante cambios bruscos de tarifa.
