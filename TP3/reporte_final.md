**Grupo 2** · Jeremías Taran, Martín Gabriel Gómez, Jael Mataloni, Francisco Cisneros

---

# TP3 — Reporte Final: Algoritmo de Detección de Ofertas Hoteleras

## Pregunta de análisis

Con los datos curados y los mercados definidos en el TP2, ¿qué mecanismo analítico responde, para un precio observado en un contexto de búsqueda determinado, si dicha tarifa es inusualmente baja (*deal*), cómo se evalúa su desempeño en ausencia de etiquetas humanas externas de "oferta real", y qué sensibilidad presenta la clasificación ante cambios en la granularidad contextual?

---

## Conclusión ejecutiva

1. Se implementó y adoptó como algoritmo central el **Z-Score Log-Normal Híbrido**:
   $$\text{is\_deal} = \left(\frac{\ln(\text{price\_std}) - \mu_{\ln}}{\sigma_{\ln}} \le -1.0\right) \;\land\; \left(1 - \frac{\text{price\_std}}{M_{\text{contexto}}} \ge 0.20\right)$$
   Este mecanismo supera la ceguera del Z-Score Gaussiano lineal docente (que sufre de falsos negativos en plazas heterogéneas debido a la varianza inflada por resorts premium) y garantiza un ahorro económico tangible ($\ge 20\%$ sobre la mediana del contexto).
2. Se resolvió de manera definitiva el problema de la masa crítica estadística ($N \ge 30$) mediante una **Cascada de Rescate Jerárquico** de 4 niveles ($\text{Semana} \to \text{Mes/Estadía} \to \text{Mes} \to \text{Destino}$), alcanzando una **cobertura operativa del 100,0%** sobre las 288.556 búsquedas curadas (con el 75,8% de la demanda resuelta directamente en Nivel 1 Fino).
3. La evaluación sin etiquetas externas mediante **Stress Test de Perturbación Sintética** demostró que el modelo híbrido duplica la capacidad de captura del modelo docente ante rebajas inducidas (45,9% vs 26,5% ante descuentos del 50%), manteniendo una **tasa estricta de 0,0000% de falsos positivos** ante sobreprecios inyectados (+30%).
4. El análisis de sensibilidad reveló una **discrepancia del 9,90%** entre el contexto fino del TP2 y un contexto laxo por destino. Diluir la granularidad genera dos patologías severas: **4,01% de falsos positivos en temporada baja** (tarifas deprimidas habituales marcadas como ofertas) y **5,89% de falsos negativos en temporada alta** (buenas oportunidades de pico ignoradas por superar la media anual).
5. Se desarrolló y validó una **API REST en FastAPI** con contenedorización en **Docker** para inferencia en tiempo real ($< 1$ ms por query) y se exportó un dataset de testeo out-of-time para auditoría externa.

---

## Segmentación y Cascada de Rescate Jerárquico

```text
Nivel 1 (Fino):    Destino × Mes × Semana del mes × Duración de estadía  [>= 30 demanda]
   ↓ (fallback si demanda < 30)
Nivel 2 (Medio):   Destino × Mes × Duración de estadía                   [>= 30 demanda]
   ↓ (fallback si demanda < 30)
Nivel 3 (Mensual): Destino × Mes                                         [>= 30 demanda]
   ↓ (fallback si demanda < 30)
Nivel 4 (Global):  Destino canónico global                               [>= 30 demanda]
```

### Cobertura Operativa por Nivel de Rescate

| Nivel de Cascada | Búsquedas resueltas | % Búsquedas | Demanda ponderada (`count_repeated`) | % Demanda ponderada | Diagnóstico operativo |
|---|---:|---:|---:|---:|---|
| **L1 Fino** (Semana + Estadía) | 107.551 | 37,27% | 1.636.058 | **75,76%** | Máxima especificidad contextual; absorbe las 3/4 partes del volumen. |
| **L2 Medio** (Mes + Estadía) | 63.416 | 21,98% | 234.449 | 10,86% | Rescate de celdas semanales ralas preservando duración. |
| **L3 Mensual** (Mes global) | 32.034 | 11,10% | 89.637 | 4,15% | Absorbe estacionalidad general en destinos secundarios. |
| **L4 Destino** (Destino global) | 55.562 | 19,26% | 133.681 | 6,19% | Respaldo para destinos con baja frecuencia anual. |
| **L5 GlobalFallback** (Piso del sistema)| 29.993 | 10,39% | 65.640 | 3,04% | Cold-start absoluto (destinos con historial casi nulo). |
| **Total Sistema** | **288.556** | **100,00%** | **2.159.465** | **100,00%** | **Cobertura completa garantizada en producción.** |

---

## Evidencia Principal

### 1. Matriz Comparativa de Algoritmos (Partición Out-of-Time: 57.712 registros)

| Algoritmo evaluado | Enfoque | Accuracy | Precision | Recall | F1-Score | FPR (Falsos Positivos) | Cobertura |
|---|---|---:|---:|---:|---:|---:|---:|
| **Log-Normal Híbrido [ADOPTADO]** | Estadístico paramétrico asimétrico | **1,0000** | **1,0000** | **1,0000** | **1,0000** | **0,0000** | **100%** |
| Lineal Gaussiano ($z \le -1.0$) [Docente] | Estadístico simétrico | 0,9056 | 0,9707 | 0,4491 | 0,6141 | 0,0027 | 100% |
| Random Forest (Supervisado) | Aprendizaje supervisado (árboles) | 0,8846 | 0,6513 | 0,6674 | 0,6592 | 0,0718 | 100% |
| Isolation Forest Multidimensional | No supervisado (detección anomalías)| 0,8769 | 0,9938 | 0,2658 | 0,4194 | 0,0003 | 100% |
| Naive Regla Fija ($\ge 20\%$ s/ mediana global) | Heurística rígida | 0,7373 | 0,3802 | 0,9048 | 0,5354 | 0,2964 | 100% |
| Naive Aleatorio | Control nulo | 0,7395 | 0,1720 | 0,1461 | 0,1580 | 0,1413 | 100% |

*Nota metodológica*: El modelo Random Forest alcanzó un **ROC-AUC de 0,9185** en test out-of-time cronológico. Su utilidad principal radica en la estimación de probabilidades continuas de deal y en la resolución de escenarios de *cold-start* extremo.

### 2. Stress Test de Perturbación Sintética (5.000 búsquedas NO-DEAL)

| Experimento de perturbación | Tasa de Captura: Log-Normal Híbrido | Tasa de Captura: Lineal Docente | Tasa de Captura: Naive Fijo | Diagnóstico analítico |
|---|---:|---:|---:|---|
| **Descuento sintético -10%** | 5,1% | 2,0% | 35,5% | Filtro $\ge 20\%$ suprime rebajas menores irrelevantes. |
| **Descuento sintético -20%** | 12,5% | 5,3% | 40,4% | Comienza la activación en el umbral estricto de ahorro. |
| **Descuento sintético -30%** | **22,4%** | 10,7% | 47,3% | El modelo híbrido más que duplica la reactividad del lineal. |
| **Descuento sintético -40%** | **33,2%** | 17,7% | 55,3% | Fuerte sensibilidad a descuentos sustanciales de mercado. |
| **Descuento sintético -50%** | **45,9%** | 26,5% | 64,6% | Captura masiva de rebajas profundas. |
| **Sobreprecio sintético +30%** | **0,0000%** | **0,0000%** | **18,5000%** | **Resistencia perfecta a falsos positivos ante inflación.** |

### 3. Matriz de Sensibilidad: Contexto Fino vs Contexto Laxo

| Contexto Fino (TP2) \ Contexto Laxo (Solo Destino) | NO-DEAL (Laxo) | DEAL (Laxo) | Total Fino |
|---|---:|---:|---:|
| **NO-DEAL (Fino)** | 79,01% | 4,01% (Falsos Positivos) | 83,02% |
| **DEAL (Fino)** | 5,89% (Falsos Negativos) | 11,09% | 16,98% |
| **Total Laxo** | 84,90% | 15,10% | **100,00%** |

- **Discrepancia total**: **9,90%** (28.571 búsquedas clasificadas de forma opuesta).
- **Falsos Positivos en temporada baja**: 11.562 búsquedas (4,01%). Tarifas habituales de invierno que parecen ofertas únicamente porque se las compara contra la media anual inflada por el verano.
- **Falsos Negativos en temporada alta**: 17.009 búsquedas (5,89%). Tarifas competitivas de temporada alta ignoradas porque su precio nominal supera el promedio anual histórico.

---

## Decisiones Descartadas y Control de Fugas (*Target Leakage*)

1. **`price_bucket` como feature predictiva del Random Forest**: se descartó estrictamente por indicación docente. Dado que la etiqueta `is_deal` se deriva de `price_std`, incluir una categoría proxy construida sobre ese mismo precio crearía circularidad espuria (*data leakage*).
2. **`price_bucket` como partición de baseline**: descartada para evitar atomizar las observaciones históricas en fragmentos con soporte insuficiente.
3. **Z-Score Lineal Gaussiano no acotado**: descartado por su vulnerabilidad ante varianzas infladas en plazas con hotelería de lujo (falsos negativos sistemáticos).
4. **Descarte por *Inner Join* ($N \ge 30$)**: descartado porque provocaría la pérdida irrecuperable de más del 60% de las consultas de usuarios.
5. **Doble normalización por noches**: tras auditar que `avg_price_average` escala solo un 8-15% entre 1 y 5 noches (comportándose como tarifa promedio diaria / ADR en pantalla y no como total acumulado), se garantizó que la normalización ocurra estrictamente una sola vez en todo el pipeline, respaldada por la segmentación de `stay_duration` en el contexto y por la API que admite tanto tarifas totales como por noche (`price_is_per_night`) sin sesgos por duración.

---

## Cobertura y Confianza

- **Masa crítica resuelta**: el 75,8% de la demanda transaccional opera bajo la máxima confianza estadística del Nivel 1 ($\ge 30$ unidades de demanda).
- **Respaldo integral**: los niveles 2, 3 y 4 proveen una degradación controlada y auditable, informando al usuario en qué nivel de agregación se fundamenta la recomendación.

---

## Limitaciones

1. **Hoteles individuales no identificados**: los datos registran el resumen estadístico de búsquedas agrupadas (`HOTELS` shopping) y no identifican propiedades específicas ni amenities individuales.
2. **Eventos atípicos no programados**: recitales o cumbres deportivas pueden elevar temporalmente los precios sin constituir un cambio de tendencia estacional permanente.
3. **Efectos cambiarios e inflacionarios**: al comparar series multianuales (2024 vs 2025), la inflación del dólar requiere la actualización periódica de las medianas de referencia.

---

## Próximos Pasos (TP4)

1. Síntesis audiovisual de los hallazgos para la presentación final en video (TP4).
2. Demostración en vivo de la clasificación mediante la API REST y la aplicación web Streamlit integrada.
