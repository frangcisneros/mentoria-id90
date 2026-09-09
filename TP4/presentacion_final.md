---
marp: true
theme: gaia
_class: lead
paginate: true
backgroundColor: #0f172a
color: #f8fafc
style: |
  section {
    font-family: 'Inter', sans-serif;
    padding: 40px 60px;
  }
  h1, h2 {
    color: #38bdf8;
  }
  footer {
    font-size: 0.55rem;
    color: #94a3b8;
  }
  .highlight {
    color: #34d399;
    font-weight: bold;
  }
  .card {
    background-color: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 8px;
    padding: 16px;
    margin-top: 10px;
  }
---

# Sistema de Detección de Ofertas Hoteleras
### ID90Travel · Mentoría DiploDatos 2026

**Grupo 02:** Francisco Cisneros  
**Docente mentor:** Martín  
**Fecha:** Octubre 2026  

---

## 1. El Problema de Negocio

* **Dolor del usuario:** Millones de búsquedas hoteleras diarias se realizan sin certeza de si el precio mostrado es conveniente o inflado.
* **Desafío técnico:**
  * ¿Contra qué se compara una tarifa para que la comparación sea justa?
  * ¿Cómo se detecta estadísticamente una oportunidad sin tener etiquetas humanas previas?
* **Solución desarrollada:** Un pipeline de clasificación contextual en tiempo real con baselines log-normales y garantías de ahorro económico.

<!--
GUION DE LOCUCIÓN (Slide 1 y 2):
"Buenas tardes. Mi nombre es Francisco Cisneros, integrante del Grupo 2 de la mentoría de ID90Travel en la Diplomatura en Ciencia de Datos. Hoy les presento el desarrollo y los resultados del Sistema de Detección Automática de Ofertas Hoteleras.
En plataformas de viajes, los usuarios se enfrentan permanentemente a la incertidumbre: ¿este precio de 120 dólares en Miami es una oportunidad real o es una tarifa inflada? Para responder esto, el desafío no es solo el algoritmo matemático, sino definir contra qué comparamos ese precio. Un hotel en Las Vegas un viernes de diciembre no compite con uno un martes de febrero. Diseñamos un sistema integral que normaliza, contextualiza y clasifica tarifas con rigor estadístico y validación empírica."
-->

---

## 2. El Dataset y la Fragmentación Geográfica

* **Volumen:** 5.1 millones de búsquedas históricas (muestra de trabajo: 300k).
* **El problema de los destinos:** Más de 26.000 nombres de ciudad crudos con tipografías inconsistentes ("NYC", "New York City", "Manhattan").
* **Estandarización implementada:**
  * Algoritmo de mapeo jerárquico (`destination_with_nearest.csv`).
  * Asignación por proximidad geográfica a 2.584 destinos canónicos.
  * Cobertura lograda: **69.9%** mapeada a polos de alta densidad; el resto en fallback directo.

<!--
GUION DE LOCUCIÓN (Slide 3):
"El dataset provisto por ID90Travel contiene más de 5 millones de búsquedas. El primer obstáculo que encontramos fue la fragmentación geográfica: teníamos más de 26.000 nombres de ciudad diferentes debido a la multiplicidad de proveedores de hoteles.
Para que los datos sean comparables, era imprescindible consolidar mercados. Implementamos un pretratamiento de normalización de texto y un mapping jerárquico basado en coordenadas geográficas. Esto nos permitió consolidar la dispersión en 2.584 destinos canónicos con masa crítica suficiente para sostener estimaciones estadísticas confiables."
-->

---

## 3. Curación y Métrica Base: `price_std`

* **El sesgo del tamaño de grupo:** Comparar el precio total de la reserva mezcla familias de 4 personas con viajeros individuales.
* **Métrica homogénea adoptada:**
  $$\text{price\_std} = \frac{\text{avg\_price\_average}}{\text{nights} \times \text{rooms} \times (\text{adults} + 0.5 \times \text{kids})}$$
* **Filtros de calidad operativa:**
  * Eliminación de noches $\le 0$, habitaciones $\le 0$, adultos $= 0$.
  * Rango válido: $[5, 2500]$ USD por habitación-noche-persona.

<!--
GUION DE LOCUCIÓN (Slide 4):
"Una decisión metodológica crucial del Trabajo Práctico 2 fue la definición de la métrica base. Si comparamos el precio total de la reserva, estamos mezclando el costo del hotel con la cantidad de noches y la cantidad de pasajeros.
Definimos 'price_std': el precio en dólares por habitación, por noche y por adulto equivalente, considerando a los menores con un factor de medio adulto. Además, filtramos inconsistencias operativas como búsquedas con cero ocupantes o tarifas astronómicas fuera del rango de 5 a 2.500 dólares."
-->

---

## 4. Definición de Mercado: Contexto Fino

$$\text{Mercado} = \text{destination\_final} \times \text{month} \times \text{week\_in\_month} \times \text{stay\_duration}$$

* **Temporal:** Mes del año y semana del mes (captura estacionalidad de vacaciones y quincenas).
* **Duración:** Escapada corta ($\le 2$ noches), media ($3\text{--}5$) o vacacional ($> 5$).
* **Masa crítica ($N \ge 30$):**
  * Asegura estabilidad muestral de los estimadores.
  * Cascada de rescate para celdas pequeñas: $\text{Semana} \to \text{Mes} \to \text{Destino global}$.

<!--
GUION DE LOCUCIÓN (Slide 5):
"¿Qué es un mercado en este problema? No alcanza con mirar la ciudad. Definimos el mercado como la combinación de cuatro dimensiones: Destino Canónico, Mes, Semana dentro del mes, y Tipo de Estadía.
Esta granularidad retiene el comportamiento homogéneo de oferta y demanda. Para garantizar robustez estadística, exigimos al menos 30 observaciones históricas por celda de mercado. Si una búsqueda cae en una celda con menos historial, el sistema activa una cascada de rescate automática retrocediendo al mes completo o al destino general."
-->

---

## 5. El Algoritmo: Z-Score Lineal vs Log-Normal

* **Limitación del Z-Score Gaussiano Lineal:**
  * Las tarifas hoteleras tienen distribución asimétrica con cola pesada a la derecha.
  * Los hoteles de lujo inflan artificialmente $\sigma$.
  * Tarifas económicas reales (ej. 25 USD con $\mu=90, \sigma=75$) dan $z=-0.87$: <span class="highlight">falso negativo</span>.
* **Z-Score Log-Normal Híbrido (Adoptado):**
  $$\text{is\_deal} = \left(\frac{\ln(\text{price\_std}) - \mu_{\ln}}{\sigma_{\ln}} \le -1.0\right) \;\land\; \left(\text{descuento s/ mediana} \ge 20\%\right)$$
* **Tasa de detección:** **13.92%** de las tarifas (vs 6.38% del modelo lineal).

<!--
GUION DE LOCUCIÓN (Slide 6 y 7):
"En el TP3 abordamos el núcleo algorítmico. El baseline propuesto inicialmente usaba un Z-score gaussiano lineal clásico. Sin embargo, al analizar las distribuciones en plazas heterogéneas como Las Vegas o Cancún, comprobamos que los resorts de lujo de 800 dólares inflan la desviación estándar lineal. Una tarifa de 25 dólares quedaba apenas en z = -0.87 y el sistema no la marcaba como oferta.
La solución fue trabajar en escala logarítmica, midiendo distancias proporcionales relativas, e incorporar una regla de negocio indispensable: la tarifa debe representar un ahorro de al menos el 20% respecto a la mediana del contexto. Esto evita falsos deals en hoteles de tarifa plana donde una diferencia de dos dólares activaría una falsa alarma."
-->

---

## 6. Evaluación sin Etiquetas Externas

1. **Validación Out-of-Time (Split Cronológico):**
   * Train 80% primeros meses / Test 20% meses finales.
   * Tasa de deals estable en el tiempo (~13.9%).
2. **Stress Test de Inyección Sintética de Descuentos:**
   * Ante descuentos del 20%: recupera el **48.2%** de los casos (vs 22.1% lineal).
   * Ante descuentos del 40%: recupera el **94.6%** de los casos (vs 68.3% lineal).
3. **Resistencia a Falsos Positivos:**
   * Con precios inflados un +30%: **0.0000%** de falsos deals.

<!--
GUION DE LOCUCIÓN (Slide 8):
"Al no contar con etiquetas de ground truth humano externo, validamos el sistema con dos experimentos rigurosos.
Primero, una validación cronológica out-of-time para medir si el algoritmo sufría de drift temporal.
Segundo, un stress test de inyección sintética: tomamos miles de tarifas normales y les inyectamos descuentos forzados del 10 al 50 por ciento. El modelo log-normal híbrido demostró una sensibilidad superior, capturando más del 94% de las ofertas reales con descuentos del 40%, mientras que al inyectar sobreprecios del 30%, la tasa de falsos positivos fue exactamente cero."
-->

---

## 7. Modelo Predictivo y Aplicación Web

* **Random Forest Predictivo:**
  * Entrenado sobre variables disponibles al momento de la búsqueda.
  * **ROC-AUC: 0.9853** · Precision: 87.5% · Recall: 83.0%.
  * Permite pre-clasificar ofertas en milisegundos en el backend.
* **Aplicación Streamlit (`app.py`):**
  * Consulta interactiva por destino, fechas y composición.
  * Visualización del Z-Score contextual, ahorro estimado y badge de oportunidad.

<!--
GUION DE LOCUCIÓN (Slide 9 y 10):
"Para llevar esta solución a un entorno de producción de baja latencia, entrenamos un clasificador Random Forest que predice la probabilidad de deal en menos de un milisegundo a partir de las variables de búsqueda, alcanzando un ROC-AUC de 0.985.
Todo este pipeline se encuentra integrado y operativo en la aplicación Streamlit del repositorio, donde cualquier usuario o analista de ID90 puede ingresar un destino, fechas y cantidad de pasajeros, observando en tiempo real si la tarifa encontrada califica como deal, el ahorro proyectado en dólares y la comparativa frente a la distribución histórica."
-->

---

## 8. Conclusiones y Próximos Pasos

* **Logros alcanzados:**
  * Segmentación contextual que retiene la homogeneidad de mercado.
  * Clasificador log-normal que supera ampliamente al Z-score lineal tradicional.
  * Validación empírica sin etiquetas externas demostrada.
* **Próximos pasos para ID90Travel:**
  * Enriquecer destinos de cola larga ($N < 30$) mediante clustering de similitud de precios.
  * Detección automática de eventos atípicos mediante series temporales.

---

# ¡Muchas gracias!
### Preguntas y Comentarios
