# TP1 — Reporte Final: Definición de mercado hotelero

## Pregunta de análisis

¿Qué combinación de dimensiones genera grupos de búsquedas hoteleras lo suficientemente homogéneos como para que la comparación de precios dentro de cada grupo sea estadísticamente válida?

La pregunta es previa a la detección de ofertas: antes de clasificar un precio como "barato" o "caro", necesitamos definir contra qué mercado y en qué contexto temporal se lo compara.

---

## Conclusión ejecutiva

El mercado es la geografía canónica (`destination_final`). Comparar precios de destinos distintos —o de ciudades crudas contra destinos canónicos— no tiene sentido.

Sobre `destination_final`, la segmentación que mejor equilibra homogeneidad y cobertura es:

```text
Contexto = destination_final × month × week_in_month × stay_duration
```

`price_bucket` es opcional. Es un proxy de categoría hotelera porque no tenemos estrellas, amenities ni marca.

---

## Segmentación propuesta

| Dimensión | Rol | Justificación |
|---|---|---|
| `destination_final` | Base geográfica principal | Los precios son heterogéneos entre destinos; agrupar por ciudad cruda o país genera mercados no comparables. |
| `month` | Alta | Captura estacionalidad climática y turística. |
| `week_in_month` | Media | Captura dinámicas intra-mes (quincenas, liquidación de sueldos). |
| `stay_duration` | Alta | Existe una curva de descuento por volumen no lineal; estadías largas tienen tarifa por noche menor. |
| `price_bucket` | Opcional | Proxy de categoría hotelera (budget/mid/premium) por falta de estrellas/amenities. |

---

## Evidencia principal

### 1. Signal-to-Noise Ratio (SNR) por combinación de dimensiones

| Segmentación | SNR | Contextos | % Contextos confiables (N≥30) | % Demanda confiable |
|---|---|---|---|---|
| Solo destino | 0.0603 | 1.890 | 69,4% | 99,9% |
| Destino + Mes | 0.1222 | 13.902 | 47,3% | 98,5% |
| Destino + Mes + Semana | 0.1912 | 38.268 | 34,8% | 95,4% |
| Destino + Mes + Semana + Estadía | 0.5384 | 68.487 | 28,5% | 91,4% |

> Valores calculados sobre una muestra representativa de 300.000 filas originales (semilla 42), usando solo destinos canónicos mapeados. El soporte se mide como la suma de `count_repeated` (demanda ponderada) por contexto.

Agregar `month`, `week_in_month` y `stay_duration` sube el SNR, pero fragmenta los datos. La segmentación propuesta conserva más del 90% de la demanda en contextos con al menos 30 unidades de demanda ponderada.

### 2. Estabilidad interanual 2024 vs 2025

- Duración de estadía: correlación ≈ 1.000
- Día de la semana: correlación ≈ 1.000
- Jerarquía de destinos: correlación > 0.90

Esto indica que la estructura de mercado es estable en el tiempo, al menos para las dimensiones geográficas y de duración.

### 3. Cobertura del mapping

| Métrica | Valor |
|---|---|
| Filas mapeadas | 69,0% |
| Demanda mapeada | 80,5% |
| Geografías únicas mapeadas | ~36% |

La mayor parte de la demanda se concentra en destinos que sí logramos mapear. La cola larga de no-matcheados corresponde a barrios, distritos, CDP y variantes administrativas.

---

## Decisiones descartadas
 
| Variable | Motivo de exclusión |
|---|---|
| `country` | Muy agregado; mezcla destinos con perfiles de precio distintos (ej. Miami vs Detroit). |
| `day_of_week` | Efecto real pero heterogéneo entre destinos; agregarlo fragmenta demasiado sin ganar homogeneidad generalizada. |
| `avg_hotel_count` | Variable de control, no de partición; agregarla genera celdas muy pequeñas. |
| `count_repeated` | Es un peso de demanda, no una dimensión de segmentación. |

---

## Cobertura y confianza

Sobre el dataset completo (2024 + 2025, 5,16M filas originales), el pipeline corregido genera:

| Métrica | Valor |
|---|---|
| Contextos totales | ~707.000 |
| Contextos de alta confianza (N≥30) | 157.705 (23,4%) |
| Demanda en contextos de alta confianza | 96,3% del total |

Solo el 23,4% de los contextos supera N≥30, pero concentran el 96,3% de la demanda.

---

## Limitaciones

1. **`price_bucket` como proxy:** al no contar con estrellas, amenities ni marca del hotel, usamos percentiles del precio observado para aproximar la categoría. Esto introduce riesgo de "leakage": el precio define el grupo y luego se evalúa contra ese grupo.

   | Métrica | Sin `price_bucket` | Con `price_bucket` |
   |---|---|---|
   | Contextos totales | 365.989 | 673.030 |
   | Alta confianza (N≥30) | 101.880 (27,8%) | 157.705 (23,4%) |
   | Demanda total | 99.975.146 | 97.984.772 |
   | `mean_price_std` mediana (alta conf.) | 32,28 | 26,45 |
   | `std_price_std` mediana (alta conf.) | 22,29 | 6,98 |

   Con `price_bucket` se generan más contextos y los desvíos internos son menores, porque separa hoteles por rango de precio. La contra es que el precio define el bucket y luego se evalúa contra ese bucket. En TP2 hay que medir si esto reduce la cantidad de ofertas detectadas.

2. **No-matcheados geográficos:** aproximadamente el 20% de la demanda no tiene destino canónico. Son principalmente barrios, CDP y territorios. Quedan como `match_level = no_match` y no se les asigna destino por fallback automático.

3. **Outliers:** se aplicó una política inicial de filtrado (precios > 0, estadías ≤ 30 noches, winsorización al percentil 99.9). Aún quedan valores extremos que requieren auditoría manual.

4. **Muestra del notebook:** el notebook de exploración procesa la muestra representativa de 300.000 filas (seed=42) para viabilidad en memoria; el pipeline de producción procesa los históricos completos.

---

## Próximos pasos (TP2)

1. Implementar el baseline productivo.
2. Medir la sensibilidad del detector de ofertas con y sin `price_bucket`.
3. Definir fallbacks para contextos con poca muestra.
4. Construir la aplicación de consulta.
5. Documentar el comando exacto para regenerar todo desde cero.
