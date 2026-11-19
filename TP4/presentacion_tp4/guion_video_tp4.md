# Guion Completo — Video Final TP4 (Grupo 02 · Sistema de Detección de Ofertas Hoteleras)

**Diplomatura en Ciencia de Datos, Aprendizaje Automático y sus Aplicaciones (FAMAF – UNC) · 2026**  
**Mentoría:** Sistema de detección de ofertas hoteleras  
**Integrantes:** Jeremías Taran · Martín Gabriel Gómez · Jael Mataloni · Francisco Cisneros  
**Mentor:** Martín (`id90-hotel-deals-analysis`)  

---

## 1. Resumen de asignación por integrante (4 bloques · 13:30 min totales)

| Bloque | Orador/a | Slides (`presentacion_tp4.html` / `.pdf`) | Tiempo asignado | Duración | Temas que expone |
|---:|---|---|---|---:|---|
| **Bloque 1** | **Jeremías Taran** | **Slides 01, 02 y 03** | `0:00 – 3:15` | **3 min 15 s** | Presentación del equipo, problema de negocio, embudo de curación (5,15M), normalización geográfica (2.584 destinos), unidad comparable (`price_std`), definición de mercado (SNR 9×) y hallazgo de pseudoreplicación. |
| **Bloque 2** | **Martín Gabriel Gómez** | **Slides 04, 05 y 06** | `3:15 – 6:45` | **3 min 30 s** | Cascada jerárquica de fallback (L1→L6, cobertura 44,2% → 98,9%), por qué falla el Z-score gaussiano lineal, formulación del Detector Híbrido Log-Normal y diseño experimental sin etiquetas externas (Split Temporal 2024→2025 vs Split Aleatorio 2024 y reglas Anti-Leakage). |
| **Bloque 3** | **Jael Mataloni** | **Slides 07, 08, 09 y 10** | `6:45 – 10:15` | **3 min 30 s** | Prioridad de métricas de negocio (Precision, FPR ≤ 5%, F1), comparación de los 7 modelos (A–G, incluyendo HistGradientBoosting y Random Forest), curva de calibración de umbral, pruebas con 2.400 descuentos sintéticos, sensibilidad del contexto y justificación de producto de por qué se elige el Híbrido (E) sobre Boosting (F). |
| **Bloque 4** | **Francisco Cisneros** | **Slides 11, 12, 13 y 14** | `10:15 – 13:30` | **3 min 15 s** | Radiografía de errores (deriva interanual −20%/+25% y segmentos), resolución del desafío de unidad de precio entre Train y API (mejora en Benchmark Ciego de F1 `0,528 → 0,734`), demostración del microservicio en producción (FastAPI + Docker, caso Orlando), hoja de ruta y cierre ejecutivo. |

---

## 2. Mapa detallado Slide por Slide (14 Slides · Quién habla en cada una)

> **Cómo usar el teleprompter integrado en [`presentacion_tp4.html`](presentacion_tp4.html):**
> - Navegá con las flechas `←` / `→` o barra espaciadora.
> - Presioná la tecla **`N`** (o el botón **`Guion [N]`** abajo): se abre una ventana flotante que muestra exactamente **qué integrante habla en esa diapositiva, el minuto y su texto palabra por palabra**.

| Slide | Tiempo | Duración | Quién habla | Título en la diapositiva |
|---:|---|---:|---|---|
| **Slide 01** | `0:00 – 0:55` | 55 s | **Jeremías Taran** | *Sistema de Detección de Ofertas Hoteleras* (Portada) |
| **Slide 02** | `0:55 – 2:05` | 1 min 10 s | **Jeremías Taran** | *01 / Datos y Curación — Curación estructural y normalización geográfica sin imputación forzada* |
| **Slide 03** | `2:05 – 3:15` | 1 min 10 s | **Jeremías Taran** | *02 / Definición de Mercado — El contexto de cuatro dimensiones multiplica 9× la relación señal-ruido* |
| **Slide 04** | `3:15 – 4:25` | 1 min 10 s | **Martín Gabriel Gómez** | *03 / Arquitectura de Contexto — La cascada de 6 niveles eleva la cobertura operativa del 44,2% al 98,9%* |
| **Slide 05** | `4:25 – 5:35` | 1 min 10 s | **Martín Gabriel Gómez** | *04 / Formulación del Detector — Los resorts de lujo inflan la desviación lineal y ocultan ofertas reales* |
| **Slide 06** | `5:35 – 6:45` | 1 min 10 s | **Martín Gabriel Gómez** | *05 / Diseño Experimental — Evaluar en el mismo año infla el F1 a 0,919; el verdadero reto es anticipar 2025* |
| **Slide 07** | `6:45 – 7:40` | 55 s | **Jael Mataloni** | *06 / Aprendizaje Automático y Benchmark — El Detector Híbrido (E) y HistGradientBoosting (F) lideran el test temporal 2025* |
| **Slide 08** | `7:40 – 8:30` | 50 s | **Jael Mataloni** | *07 / Visualización de Modelos — Equilibrio entre F1, Precision y Falsos Positivos a través del umbral z_log* |
| **Slide 09** | `8:30 – 9:25` | 55 s | **Jael Mataloni** | *08 / Estrés Sintético y Sensibilidad — Tanto E como F siguen la curva de verdad; sin mes ni estadía se pierden 12 puntos de F1* |
| **Slide 10** | `9:25 – 10:15` | 50 s | **Jael Mataloni** | *09 / Decisión Arquitectónica — Por qué elegimos el Detector Híbrido (E) sobre HistGradientBoosting (F)* |
| **Slide 11** | `10:15 – 11:20` | 1 min 05 s | **Francisco Cisneros** | *10 / Radiografía de Errores — Los errores provienen de la deriva de precios entre años, no de ruido en el umbral* |
| **Slide 12** | `11:20 – 12:30` | 1 min 10 s | **Francisco Cisneros** | *11 / Desafíos y Solución de Ingeniería — Unificar la convención de precio entre Train y API elevó el F1 de 0,528 a 0,734* |
| **Slide 13** | `12:30 – 13:10` | 40 s | **Francisco Cisneros** | *12 / Producción y Próximos Pasos — Inferencia explicable en tiempo real y hoja de ruta de evolución* |
| **Slide 14** | `13:10 – 13:30` | 20 s | **Francisco Cisneros** | *Conclusión Ejecutiva — Un detector explicable, auditable y listo para producción* (Cierre) |

---

## 3. Guion palabra por palabra (Slide por Slide, con indicación de cada orador)

---

### BLOQUE 1 — JEREMÍAS TARAN (`0:00 – 3:15` · Slides 01 a 03)

#### SLIDE 01 — Portada y Problema de Negocio
- **Orador:** **Jeremías Taran**
- **Tiempo:** `0:00 – 0:55` (55 segundos)
- **Qué se ve en pantalla:** Portada *Sistema de Detección de Ofertas Hoteleras* y las 4 etapas del recorrido del proyecto.

**Dice Jeremías Taran:**
> «Hola a todos. Somos el Grupo 2 de la mentoría *Sistema de detección de ofertas hoteleras* —Jeremías Taran, Martín Gómez, Jael Mataloni y Francisco Cisneros— y hoy les presentamos nuestro trabajo final.
>
> El problema de negocio parte de una necesidad concreta en una plataforma de viajes: cuando un usuario busca un hotel y ve una tarifa, no sabe si ese precio es una oportunidad real o un valor habitual para esa ciudad y esa fecha.
>
> Nuestro objetivo durante toda la mentoría fue diseñar, evaluar y llevar a un servicio productivo en Docker y FastAPI un mecanismo capaz de responder en tiempo real, y con una justificación comprensible para el usuario, si una tarifa observada es inusualmente baja para su contexto. Hacerlo tiene tres grandes desafíos: comparar peras con peras entre destinos muy distintos, detectar anomalías en distribuciones de precio muy asimétricas y evaluar modelos de Aprendizaje Automático cuando no existe una etiqueta externa de qué es una oferta.»

---

#### SLIDE 02 — Dataset, Curación y Normalización Geográfica
- **Orador:** **Jeremías Taran**
- **Tiempo:** `0:55 – 2:05` (1 minuto 10 segundos)
- **Qué se ve en pantalla:** KPIs (`5,15 M` búsquedas válidas, `2.584` destinos canónicos, `82,3%` demanda cubierta, corte `p99,9`), tabla del embudo de limpieza (`99,76%` retenido) y las 3 decisiones clave de curación.

**Dice Jeremías Taran:**
> «Trabajamos con más de 5,15 millones de búsquedas históricas de 2024 y 2025, además de una muestra aleatoria reproducible de 300 mil registros.
>
> En las primeras etapas resolvimos tres problemas estructurales del dato crudo. Primero, la fragmentación geográfica: los proveedores reportaban más de 26 mil variantes de nombres de ciudad. Mediante un mapeo por proximidad geográfica consolidamos las búsquedas en 2.584 destinos canónicos, auditando y corrigiendo manualmente asignaciones dudosas. Así cubrimos el 82,3% de la demanda real y, cuando una ciudad no tiene correspondencia cercana, conservamos la localidad original como salvaguarda sin forzar imputaciones erróneas.
>
> Segundo, corregimos la desagregación temporal para expandir únicamente las noches efectivamente pagadas, excluyendo el día de check-out. Y tercero, aplicamos un embudo de limpieza con corte superior en el percentil 99,9 ajustado exclusivamente en entrenamiento, reteniendo el 99,76% de las búsquedas.»

---

#### SLIDE 03 — Unidad Comparable (`price_std`) y Definición de Mercado
- **Orador:** **Jeremías Taran**
- **Tiempo:** `2:05 – 3:15` (1 minuto 10 segundos)
- **Qué se ve en pantalla:** Fórmula de mercado (`Destino Canónico × Mes × Semana del Mes × Duración de Estadía`), tabla de relación señal-ruido (`SNR` de `0,0603` a `0,5384`, mejora de `9×`) y tabla de pseudoreplicación (`77,0%` con repeticiones vs `16,8% / 44,2%` con búsquedas independientes).

**Dice Jeremías Taran:**
> «Para que dos búsquedas sean comparables, llevamos todo a una misma unidad: dólares por habitación, por noche y por persona, a la que llamamos precio estandarizado (`price_std`).
>
> Luego nos preguntamos: ¿qué define a un mercado comparable? Probamos distintas particiones midiendo la relación señal-ruido. Pasar de mirar solo el destino a combinar destino canónico, mes, semana del mes y tramo de duración de estadía multiplica por nueve la relación señal-ruido.
>
> Ahora bien, descubrimos un fenómeno crítico de pseudoreplicación: muchas celdas parecían tener más de 30 observaciones solo porque una misma búsqueda se repetía cientos de veces. Si exigimos 30 búsquedas independientes reales para tener estimadores estables, el contexto más fino solo alcanza a cubrir el 16,8% en la muestra y el 44,2% en el historial completo. Esa brecha fue la que motivó la cascada jerárquica que presenta ahora Martín.»

---

### BLOQUE 2 — MARTÍN GABRIEL GÓMEZ (`3:15 – 6:45` · Slides 04 a 06)

#### SLIDE 04 — Cascada Jerárquica de Fallback (L1 → L6)
- **Orador:** **Martín Gabriel Gómez**
- **Tiempo:** `3:15 – 4:25` (1 minuto 10 segundos)
- **Qué se ve en pantalla:** Pirámide L1 a L7 a la izquierda y gráfico de barras de cobertura en test 2025 a la derecha (`44,2%` en L1 hasta `98,9%` acumulado en L6, y `1,1%` de abstención explícita en L7).

**Dice Martín Gabriel Gómez:**
> «Gracias, Jeremías. Como el contexto ideal de cuatro dimensiones solo reúne al menos 30 búsquedas independientes en el 44,2% de las consultas de 2025, un sistema sin respaldo dejaría sin respuesta a más de la mitad de los usuarios.
>
> Por eso diseñamos una cascada jerárquica determinista de seis niveles. Cuando llega una consulta, el motor busca primero en el Nivel 1: destino, mes, semana y estadía. Si esa celda no alcanza las 30 búsquedas independientes en la historia de entrenamiento, relaja gradualmente una dimensión: pasa al Nivel 2 quitando la semana del mes —que suma otro 23,2%—, luego a destino y mes, destino y estadía, destino global y, como último rescate regional, país, mes y estadía.
>
> Dos de cada tres búsquedas se resuelven en los dos primeros niveles de alta especificidad, y la cascada completa eleva la cobertura operativa del 44,2% al **98,9%**, absteniéndose de forma explícita y honesta en apenas el 1,1% restante.»

---

#### SLIDE 05 — Del Z-Score Lineal al Detector Híbrido Log-Normal
- **Orador:** **Martín Gabriel Gómez**
- **Tiempo:** `4:25 – 5:35` (1 minuto 10 segundos)
- **Qué se ve en pantalla:** Comparación lado a lado entre el sesgo del Z-score gaussiano lineal (ejemplo empírico: tarifa de `25 USD` con media `90` y desvío `75` da `z = -0,87` y no se detecta) y el Detector Híbrido adoptado (`z_log < umbral` y descuento `≥ 20%`).

**Dice Martín Gabriel Gómez:**
> «Una vez resuelto contra qué contexto comparamos, teníamos que definir la regla estadística. El punto de partida de la mentoría era el Z-score gaussiano lineal. Pero al explorar las distribuciones vimos que los precios hoteleros tienen una fuerte asimetría a la derecha: como no tenemos las estrellas del hotel, un mismo mercado mezcla hospedajes económicos con resorts de lujo de más de 800 dólares. Esos valores altos inflan la desviación estándar lineal y ocultan las ofertas reales: en el ejemplo en pantalla, una tarifa muy baja de 25 dólares en una plaza con media 90 y desvío 75 da un Z lineal de $-0,87$ y queda como falso negativo.
>
> Al pasar a escala logarítmica simetrizamos la distribución y medimos distancias proporcionales ($z_{\log} \approx -2,00$). Y para evitar falsas alarmas en mercados muy uniformes donde 2 dólares activan el Z-score, nuestro **Detector Híbrido (Algoritmo E)** exige una segunda condición simultánea: un ahorro económico real de al menos el 20% por debajo de la mediana histórica del contexto.»

---

#### SLIDE 06 — Evaluación sin Etiquetas Externas y Anti-Leakage
- **Orador:** **Martín Gabriel Gómez**
- **Tiempo:** `5:35 – 6:45` (1 minuto 10 segundos)
- **Qué se ve en pantalla:** Las 3 capas de validación (`Capa 01: Split Temporal 2024 → 2025`, `Capa 02: Control In-Period 80/20 en 2024 con F1 = 0,919`, `Capa 03: 2.400 Sintéticos + Benchmark Ciego de 540 casos`) y el banner con la regla de oro Anti-Leakage en Aprendizaje Automático.

**Dice Martín Gabriel Gómez:**
> «¿Cómo evaluamos de manera justa cuando no tenemos etiquetas humanas de qué es una oferta? Armamos una estrategia de validación en tres capas con control estricto de fuga de información.
>
> Primero, un **split temporal estricto**: todos los baselines, recortes de outliers y modelos supervisados se entrenan únicamente con check-ins de 2024 y se evalúan a ciegas sobre 2025. La etiqueta de referencia mide si el precio resultó efectivamente una oferta dentro del mercado realizado de 2025, que ningún detector puede ver.
>
> Segundo, demostramos por qué no debe evaluarse en el mismo año: en un split aleatorio 80/20 dentro de 2024 el detector alcanza un F1 de **0,919**, mientras que al pasar a 2025 se ubica entre **0,764 y 0,808**. Evaluar dentro del mismo período sobreestima el desempeño porque ignora la deriva inflacionaria y estacional entre años.
>
> Y tercero, blindamos los modelos de Aprendizaje Automático entrenándolos solo con variables exógenas —excluyendo el tramo de precio `price_bucket` porque se calcula con el propio precio observado— y sumamos pruebas sintéticas y el benchmark ciego estratificado. Le doy el pase a Jael para ver los modelos y resultados.»

---

### BLOQUE 3 — JAEL MATALONI (`6:45 – 10:15` · Slides 07 a 10)

#### SLIDE 07 — Modelos Evaluados (A–G) y Tabla de Métricas en Test Temporal 2025
- **Oradora:** **Jael Mataloni**
- **Tiempo:** `6:45 – 7:40` (55 segundos)
- **Qué se ve en pantalla:** Tabla comparativa completa de los 7 modelos (`A. Naive Aleatorio`, `B. Naive Mediana -20%`, `C. Z-Score Gaussiano`, `D. Percentil 10`, `E. Detector Híbrido + Cascada`, `F. HistGradientBoosting`, `G. Random Forest`) con Precision, Recall, F1, FPR y Cobertura (en los 5,1M y en la muestra con unidad corregida).

**Dice Jael Mataloni:**
> «Gracias, Martín. Para elegir el mecanismo de detección entrenamos y comparamos siete enfoques distintos, desde baselines simples hasta modelos supervisados de Aprendizaje Automático.
>
> Primero definimos qué métricas importan para el negocio: en un recomendador de ofertas, el peor error es el **falso positivo** —marcar como oferta un hotel que no lo está destruye la confianza del usuario—. Por eso priorizamos **Precision alta y Tasa de Falsos Positivos cercana o menor al 5%**, usamos **F1** como criterio de selección y exigimos **cobertura mayor al 98%**. No usamos Accuracy porque, con una prevalencia del 15% de ofertas, predecir siempre "no oferta" da 85% de accuracy con F1 igual a cero.
>
> En la tabla vemos resultados muy claros: el baseline B de mediana menos 20% inunda al usuario con falsos positivos (hasta 28,5%); el Z-score gaussiano C pierde más de la mitad de las ofertas reales (recall de 0,36 a 0,46); y en la cima empatan técnicamente nuestro **Detector Híbrido E** y **HistGradientBoosting F**, alcanzando un F1 de **0,764 y 0,770** en la base completa, y de **0,808 frente a 0,799** con la unidad de precio corregida.»

---

#### SLIDE 08 — Visualización Comparada y Curva de Calibración del Umbral
- **Oradora:** **Jael Mataloni**
- **Tiempo:** `7:40 – 8:30` (50 segundos)
- **Qué se ve en pantalla:** Arriba, gráfico de barras de los 7 algoritmos en F1, Precision y FPR; abajo a la izquierda, curva de trade-off de Precision, Recall y FPR según el umbral de $z_{\log}$; abajo a la derecha, lectura ejecutiva.

**Dice Jael Mataloni:**
> «En estos dos gráficos vemos el posicionamiento visual de los siete algoritmos. Arriba, el Detector Híbrido E —en azul— y HistGradientBoosting F son los únicos que superan simultáneamente el 76% de Precision y el 84% de Recall manteniendo la tasa de falsos positivos en torno al 5%.
>
> Y abajo a la izquierda vemos una ventaja clave del detector híbrido: el umbral de $z_{\log}$ funciona como una perilla de negocio gobernable. A medida que exigimos un $z_{\log}$ más negativo hacia la derecha, la precisión supera el 90% y los falsos positivos caen por debajo del 2%. Esto nos permite operar en el punto equilibrado calibrado de $-0,875$ (con 4,8% de falsos positivos) o pasar a un modo conservador de $-1,00$ sin necesidad de reentrenar ningún modelo.»

---

#### SLIDE 09 — Respuesta a Descuentos Sintéticos y Sensibilidad del Contexto
- **Oradora:** **Jael Mataloni**
- **Tiempo:** `8:30 – 9:25` (55 segundos)
- **Qué se ve en pantalla:** Izquierda: curva de respuesta ante 2.400 descuentos y sobreprecios controlados (`0,3×` a `1,5×` de la mediana). Derecha: sensibilidad del F1 y cobertura al quitar la cascada o reducir dimensiones del contexto.

**Dice Jael Mataloni:**
> «Para verificar cómo responden los modelos cuando controlamos exactamente la magnitud del descuento, construimos 2.400 casos sintéticos fijando la tarifa entre el 30% y el 150% de la mediana del mercado.
>
> A la izquierda vemos que tanto el Detector Híbrido E como HistGradientBoosting F siguen de cerca la curva de verdad: activan más del 90% en descuentos del 50% al 70% y caen a cero cuando el precio iguala o supera la mediana, mientras que la regla fija B se equivoca en el 15% de los precios normales. Además, esto deja un hallazgo comercial relevante: como cada ciudad mezcla distintas gamas de hotel, un descuento del 30% todavía está dentro de la variación normal del mercado.
>
> Y a la derecha, el estudio de ablación confirma nuestras decisiones de diseño: usar solo el destino sin mes ni estadía hace perder 12 puntos de F1 (de 0,764 a 0,644), y quitar la cascada derrumba la cobertura al 44,6%.»

---

#### SLIDE 10 — Decisión Arquitectónica: ¿Por qué elegimos el Detector Híbrido (E) sobre HistGradientBoosting (F)?
- **Oradora:** **Jael Mataloni**
- **Tiempo:** `9:25 – 10:15` (50 segundos)
- **Qué se ve en pantalla:** Matriz comparativa entre `E. Detector Híbrido + Cascada` y `F. HistGradientBoosting` en 5 dimensiones (Capacidad predictiva, Explicabilidad, Disponibilidad de variables en la API, Mantenimiento MLOps y Casos borde) + resultados end-to-end del servicio en la API (`F1 = 0,899` en casos reales, `0,933` en sintéticos, `11/11` casos borde).

**Dice Jael Mataloni:**
> «Con estos resultados, la pregunta central frente al cliente es: ante un empate técnico en F1 entre HistGradientBoosting y el Detector Híbrido, ¿cuál llevamos a producción?
>
> Nuestra recomendación es elegir el **Detector Híbrido con cascada de fallback (E)** por cuatro razones decisivas de producto e ingeniería:
> 1. **Explicabilidad total:** informa en lenguaje natural la mediana exacta, el porcentaje de ahorro, el $z_{\log}$ y el nivel de cascada, mientras que el árbol de boosting es una caja negra.
> 2. **Compatibilidad con la API en tiempo real:** HistGradientBoosting necesita la cantidad promedio de hoteles listados (`avg_hotel_count`), una variable que no viaja en una consulta puntual de precio.
> 3. **Mantenimiento de bajo costo:** se actualiza recalculando la tabla de medianas en ventana móvil sin reentrenar modelos ni codificadores.
> 4. **Desempeño probado extremo a extremo:** a través de la API alcanza un F1 de **0,899** en casos reales, **0,933** en sintéticos y supera los **11 de 11** casos borde. Le paso la palabra a Francisco.»

---

### BLOQUE 4 — FRANCISCO CISNEROS (`10:15 – 13:30` · Slides 11 a 14)

#### SLIDE 11 — Radiografía de Errores: Deriva Interanual y Comportamiento por Segmento
- **Orador:** **Francisco Cisneros**
- **Tiempo:** `10:15 – 11:20` (1 minuto 05 segundos)
- **Qué se ve en pantalla:** Izquierda: tabla de perfil de errores (`Falsos Positivos: Mediana 2025/2024 = 0,80`, `Falsos Negativos: Mediana 2025/2024 = 1,25`, `Top 50 destinos F1 0,850 vs 0,735`, `Fallback L3/L5 FPR 11-12%`, `Diciembre Recall 0,70`). Derecha: gráfico de ofertas por `price_bucket` (`53%-61%` en gama baja y `0%` en alta).

**Dice Francisco Cisneros:**
> «Gracias, Jael. Cuando auditamos los errores del detector en 2025 encontramos un diagnóstico muy claro: los errores no provienen de ruido alrededor del umbral, sino de la **deriva de precios entre 2024 y 2025**.
>
> En los falsos positivos vimos que eran destinos donde las tarifas en 2025 se abarataron en mediana un 20% respecto de 2024; por eso un precio normal de 2025 parecía oferta frente a la historia de 2024. Y en los falsos negativos ocurría lo inverso: eran mercados que en 2025 se encarecieron un 25%.
>
> Por segmento, en el Top 50 de destinos el F1 sube a **0,850**, mientras que en los rescates L3 y L5 —que mezclan duraciones de estadía— los falsos positivos suben al 11%. Y a la derecha vemos por qué la categoría proxy `price_bucket` concentra todas las ofertas en gama baja y 0% en alta: sin las estrellas reales del hotel, un cinco estrellas en oferta sigue pareciendo caro frente a la mediana general de la ciudad.»

---

#### SLIDE 12 — Desafío Clave Resuelto: Unidad de Precio (Train vs. API) y Recalibración
- **Orador:** **Francisco Cisneros**
- **Tiempo:** `11:20 – 12:30` (1 minuto 10 segundos)
- **Qué se ve en pantalla:** Izquierda: tabla empírica de mediana de tarifa histórica por noches para 2 adultos y 1 habitación (`1n: 138 USD`, `2n: 153 USD`, `3n: 177 USD`, `5n: 193 USD`, `7n: 208 USD`). Derecha: fórmulas unificadas de `price_std` (Train vs API) y tabla Antes/Después en el Benchmark Ciego Estratificado de 450 casos (`Accuracy 0,722 → 0,820`, `Precision 0,824 → 0,896`, `Recall 0,389 → 0,622`, `F1 0,528 → 0,734`, `FPR 5,6% → 4,8%`).

**Dice Francisco Cisneros:**
> «El desafío de ingeniería más importante surgió al cruzar nuestra investigación sobre la semántica del precio con el benchmark ciego estratificado de la mentoría.
>
> Al analizar el dato histórico para 2 adultos y 1 habitación comprobamos que la mediana pasaba de 138 dólares por 1 noche a apenas 208 dólares por 7 noches, y casi no variaba entre 1 y 2 habitaciones, pero sí se duplicaba con la cantidad de personas. Es decir: la tarifa histórica de entrenamiento ya viene expresada **por noche y por habitación**, mientras que la API recibe `price_total` como el **costo total de toda la estadía**. Como la fórmula heredada dividía por noches y habitaciones también en entrenamiento, desinflaba los baselines a un tercio de su valor real y hundía el recall cuando un cliente enviaba el total de la estadía.
>
> Unificamos la convención para que tanto entrenamiento como inferencia midan exactamente dólares por habitación-noche-persona, regeneramos los baselines y recalibramos el umbral en $z_{\log} < -0,875$. Al revalidar sobre los 450 casos del benchmark ciego, el recall saltó de **0,389 a 0,622** —pasando en estadías de 1 noche de 0,08 a 0,63—, los falsos positivos bajaron a **4,8%** y el **F1 global subió más de 20 puntos, de 0,528 a 0,734**.»

---

#### SLIDE 13 — Servicio en Producción (FastAPI + Docker) y Hoja de Ruta
- **Orador:** **Francisco Cisneros**
- **Tiempo:** `12:30 – 13:10` (40 segundos)
- **Qué se ve en pantalla:** Izquierda: respuesta real de `POST /evaluate` para Orlando (`3 noches, 2 adultos, 1 habitación, price_total = 150 USD → price_std = 25 USD, mediana = 60,85 USD, descuento = 58,9%, z_log = -1,32, decisión = DEAL` y explicación textual). Derecha: los 4 pasos de la hoja de ruta (`01 Ventana móvil 90-180d`, `02 Estrellas del proveedor`, `03 Quitar semana del mes en L1`, `04 Monitoreo de marcado`).

**Dice Francisco Cisneros:**
> «El sistema ya está contenedorizado en Docker con FastAPI y listo para salir a producción como MVP. A la izquierda vemos una respuesta real en Orlando para 3 noches, 2 adultos y 1 habitación con un total de 150 dólares: el servicio lo normaliza a 25 dólares por habitación-noche-persona, lo compara contra la mediana histórica de 60,85 dólares en el Nivel 2, calcula un ahorro del 59% y un $z_{\log}$ de $-1,32$, y devuelve la decisión `deal` con su justificación auditable.
>
> Y a la derecha proponemos cuatro pasos concretos para la siguiente iteración: recalcular baselines en **ventana móvil de 90 a 180 días** para absorber la deriva interanual; incorporar las **estrellas reales del proveedor** para detectar ofertas en hoteles de lujo; promover `destino × mes × estadía` como Nivel 1 (+1,1 puntos de F1); y monitorear la tasa de ofertas marcadas por destino.»

---

#### SLIDE 14 — Cierre Ejecutivo
- **Orador:** **Francisco Cisneros**
- **Tiempo:** `13:10 – 13:30` (20 segundos)
- **Qué se ve en pantalla:** Cierre ejecutivo con los 4 indicadores finales (`98,9%` cobertura operativa, `0,808` F1 en test temporal, `0,734` F1 en benchmark ciego estratificado, `11 / 11` tests superados) y los nombres de los 4 integrantes.

**Dice Francisco Cisneros:**
> «En conclusión: entregamos un sistema de detección de ofertas hoteleras estadísticamente sólido, validado sin etiquetas externas mediante split temporal, pruebas sintéticas y benchmark ciego, que empata con modelos de Gradient Boosting pero con explicabilidad total y mantenimiento mínimo. Muchas gracias al mentor Martín y a la cátedra de la Diplomatura, y quedamos a disposición para las preguntas.»

---

## 4. Banco de preguntas probables para la defensa (Jornadas 04/12 y 05/12)

1. **«¿Por qué no eligieron HistGradientBoosting si en el dataset completo tenía un F1 apenas superior (0,770 vs 0,764)?»**
   - *Respuesta sugerida (Jael):* Porque la diferencia de 0,006 en F1 es marginal (y de hecho se invierte con la unidad corregida: `0,808 vs 0,799`, y en el test set de la API donde el Híbrido llega a `0,899`), mientras que el costo operativo del modelo supervisado es mucho mayor: pierde la explicabilidad directa (mediana, descuento y $z_{\log}$ que viajan en la respuesta al usuario), requiere la variable `avg_hotel_count` que no existe en una consulta individual de precio en tiempo real y obliga a reentrenar ante inflación en lugar de recalcular medianas.
2. **«¿Cómo evaluaron los modelos si no tenían etiquetas reales de qué es una oferta?»**
   - *Respuesta sugerida (Martín):* Combinamos tres validaciones complementarias: (1) un split temporal estricto donde entrenamos en 2024 y medimos si el modelo anticipa qué tarifas resultarán en el ~15% más anómalo de la distribución realizada de 2025; (2) inyección de 2.400 descuentos sintéticos controlados (de `0,3×` a `1,5×` de la mediana) donde la verdad se conoce por construcción; y (3) el benchmark robusto estratificado de 540 casos.
3. **«¿Por qué descartaron `price_bucket` como feature o dimensión del contexto?»**
   - *Respuesta sugerida (Martín / Francisco):* Porque `price_bucket` en este dataset no proviene de una característica externa del hotel (como sus estrellas), sino de los percentiles p25 y p75 del propio precio dentro del destino. En el análisis de errores demostramos que concentra las ofertas en `low` y 0% en `high`: incluirlo como feature en el ML o en el baseline introduce fuga de información (*data leakage*) y circularidad.
4. **«¿Qué pasó con la unidad del precio entre el dataset y la API y cómo lo solucionaron?»**
   - *Respuesta sugerida (Francisco):* Demostramos empíricamente que la tarifa histórica de entrenamiento es una tarifa por noche y por habitación (casi no crece de 1 a 7 noches ni de 1 a 2 habitaciones, pero sí con los huéspedes), mientras que el contrato de la API recibe `price_total` como el costo total de la estadía. Al unificar la transformación para que ambos lados expresen exactamente USD por habitación-noche-persona y recalibrar el umbral en $z_{\log} < -0,875$, el F1 en el benchmark ciego subió de `0,528` a `0,734` y el recall de `0,389` a `0,622`.
5. **«¿Por qué exigen 30 búsquedas independientes (`n_records ≥ 30`) y no 30 unidades de demanda ponderada (`demand_weight ≥ 30`)?»**
   - *Respuesta sugerida (Jeremías):* Porque una sola búsqueda repetida 50 veces (`count_repeated = 50`) tiene varianza cero: si calculamos desviación estándar o percentiles ahí, colapsan a un único valor. Exigir 30 búsquedas independientes garantiza diversidad muestral real, y sigue usando `count_repeated` como peso de frecuencia dentro de esas 30 o más búsquedas.
