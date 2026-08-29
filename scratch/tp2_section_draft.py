@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # TP2 — Preparar los datos y dar forma al mercado

    **Mentoría DiploDatos 2026 · ID90Travel**

    ## Índice de consignas

    1. [Consigna oficial](#consigna-oficial-tp2)
    2. [Dataset curado: limpieza y estandarización](#111-dataset-curado-limpieza-y-estandarización)
    3. [Features de contexto construidas](#112-features-de-contexto-construidas)
    4. [Mapping de destinos aplicado](#113-mapping-de-destinos-aplicado)
    5. [Definición de mercado implementada](#114-definición-de-mercado-implementada)
    6. [Distribución de precios dentro de cada segmento](#115-distribución-de-precios-dentro-de-cada-segmento)
    7. [Estadísticos para detectar precios inusualmente bajos](#116-estadísticos-para-detectar-precios-inusualmente-bajos)
    8. [Conclusiones del TP2](#117-conclusiones-del-tp2)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Consigna oficial TP2

    > #### TP2 — Preparar los datos y dar forma al mercado · Entrega 28/08
    >
    > El TP2 tiene dos partes. La primera es operativa: limpiar y estructurar los datos para que sean comparables. Esto implica normalizar el precio a una unidad común (`precio_total / (noches × habitaciones × personas)`), construir las features de contexto que definen el mercado según lo que mostró el TP1 (día de la semana, mes, categoría de hotel, etc.), y aplicar el mapping de `destination_with_nearest.csv` para consolidar los ~26.000 nombres de ciudad en identificadores únicos — o proponer una agrupación alternativa con evidencia que la soporte. Es también el momento de auditar la calidad de los datos: precios inválidos, noches inconsistentes, búsquedas con ocupación cero.
    >
    > La segunda parte es analítica: con los datos ya segmentados en mercados, explorar cómo se distribuyen los precios dentro de cada uno. ¿Qué forma tiene esa distribución? ¿Hay precios que claramente se alejan del resto? ¿Qué estadístico distingue mejor los precios inusualmente bajos — percentiles, distancia a la media, otra cosa? Este análisis no tiene que llegar a un algoritmo final, pero sí a primeras observaciones concretas que orienten el diseño del TP3.
    >
    > La decisión más importante del TP2 es la definición de mercado que el grupo va a usar. Vale documentarla y justificarla con evidencia — porque condiciona todo lo que viene después.
    >
    > *Entrega: dataset curado con features construidas, definición de mercado implementada, y análisis exploratorio de cómo se distribuyen los precios dentro de cada segmento.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.1 Dataset curado: limpieza y estandarización

    Revisamos el pipeline de calidad aplicado en la sección 1 para verificar que el dataset quedó curado según lo requerido por la consigna.
    """)
    return


@app.cell
def _(df, df_raw, n_estandarizados, n_validos, pd):
    # Resumen del dataset curado
    tp2_quality_summary = pd.DataFrame({
        'etapa': ['raw', 'validados', 'estandarizados', 'mapeados'],
        'n_registros': [len(df_raw), n_validos, n_estandarizados, df['is_mapped'].sum()],
        'pct_del_anterior': [100.0,
                             100.0 * n_validos / len(df_raw),
                             100.0 * n_estandarizados / n_validos if n_validos > 0 else 0,
                             100.0 * df['is_mapped'].sum() / len(df) if len(df) > 0 else 0],
    })
    print('Resumen de curación del dataset')
    print(tp2_quality_summary.to_string(index=False))

    # Verificación de filtros de calidad en el dataframe final
    tp2_invalid = {
        'precio_std <= 0': (df['price_std'] <= 0).sum(),
        'noches <= 0': (df['nights'] <= 0).sum(),
        'habitaciones <= 0': (df['number_of_rooms'] <= 0).sum(),
        'ocupacion <= 0': ((df['number_of_adults'] + df['number_of_kids']) <= 0).sum(),
        'noches > 30': (df['nights'] > 30).sum(),
    }
    print('\nRegistros con problemas de calidad en df final:')
    for k, v in tp2_invalid.items():
        print(f'  {k}: {v:,}')

    print(f"\nPolítica de outliers: precios estandarizados mayores a 0, noches <= 30, y winsorización al percentil 99.9 de `avg_price_average_std`.")
    print(f"Rango final de price_std: [{df['price_std'].min():.2f}, {df['price_std'].max():.2f}]")
    print(f"Percentil 99.9 de price_std: {df['price_std'].quantile(0.999):.2f}")
    return tp2_invalid, tp2_quality_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.2 Features de contexto construidas

    El TP2 pide construir las variables que definen el mercado. Verificamos que están disponibles y documentamos su significado.
    """)
    return


@app.cell
def _(df, pd):
    tp2_features = pd.DataFrame({
        'feature': ['price_std', 'day_of_week', 'month', 'week_in_month', 'year', 'stay_duration', 'destination_final', 'destination_name'],
        'tipo': ['continua', 'ordinal', 'ordinal', 'ordinal', 'ordinal', 'categórica', 'categórica', 'categórica'],
        'descripcion': [
            'Precio por habitación-noche-persona (USD)',
            'Día de la semana del check-in (0=Lunes, 6=Domingo)',
            'Mes del check-in (1-12)',
            'Semana del mes del check-in (1-4)',
            'Año del check-in (2024/2025)',
            'Duración de la estadía: corta (1-2 noches), media (3-5), larga (>5)',
            'ID del destino canónico (post-mapping)',
            'Nombre del destino canónico (post-mapping)',
        ],
    })
    print('Features de contexto disponibles en df:')
    print(tp2_features.to_string(index=False))

    print('\nMuestra de registros con features:')
    print(df[['city', 'country_code', 'date_start', 'nights', 'number_of_rooms',
              'number_of_adults', 'number_of_kids', 'price_std', 'day_of_week',
              'month', 'week_in_month', 'year', 'stay_duration',
              'destination_final', 'destination_name']].head(5).to_string(index=False))
    return tp2_features


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.3 Mapping de destinos aplicado

    Aplicamos `destination_with_nearest.csv` para consolidar los ~26.000 nombres crudos en destinos canónicos. Reportamos cobertura por filas, por demanda ponderada y por geografías únicas, junto con la trazabilidad de cada match.
    """)
    return


@app.cell
def _(df, mapping_df, pd):
    # Cobertura general
    total_rows = len(df)
    mapped_rows = df['is_mapped'].sum()
    total_demand = df['count_repeated'].sum()
    mapped_demand = df.loc[df['is_mapped'], 'count_repeated'].sum()

    # Geografías únicas crudas vs mapeadas
    geo_cols = ['country_code', 'country', 'state', 'city']
    raw_geo = df[geo_cols].drop_duplicates()
    mapped_geo = df.loc[df['is_mapped'], geo_cols].drop_duplicates()

    tp2_mapping_summary = pd.DataFrame({
        'metrica': ['filas', 'demanda_ponderada', 'geografias_unicas'],
        'total': [total_rows, total_demand, len(raw_geo)],
        'mapeadas': [mapped_rows, mapped_demand, len(mapped_geo)],
        'pct_mapeado': [100.0 * mapped_rows / total_rows,
                        100.0 * mapped_demand / total_demand,
                        100.0 * len(mapped_geo) / len(raw_geo)],
    })
    print('Cobertura del mapping de destinos')
    print(tp2_mapping_summary.to_string(index=False))

    # Niveles de match
    if 'match_level' in df.columns:
        tp2_match_levels = df['match_level'].value_counts().reset_index()
        tp2_match_levels.columns = ['match_level', 'n_registros']
        tp2_match_levels['pct'] = 100.0 * tp2_match_levels['n_registros'] / total_rows
        print('\nDistribución por nivel de match:')
        print(tp2_match_levels.to_string(index=False))
    else:
        tp2_match_levels = None
        print('\nColumna match_level no disponible en este df.')

    # Top no mapeados por demanda
    if 'match_level' in df.columns:
        tp2_unmatched = (
            df.loc[df['match_level'] == 'no_match']
            .groupby(geo_cols, dropna=False)
            .agg(demand=('count_repeated', 'sum'), rows=('count_repeated', 'size'))
            .sort_values('demand', ascending=False)
            .head(15)
            .reset_index()
        )
        print('\nTop 15 geografías no mapeadas por demanda:')
        print(tp2_unmatched.to_string(index=False))
    else:
        tp2_unmatched = None
    return tp2_mapping_summary, tp2_match_levels, tp2_unmatched


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.4 Definición de mercado implementada

    Basándonos en la evidencia del TP1, definimos el mercado como:

    $$
    \text{Contexto} = \text{destination\_final} \times \text{month} \times \text{week\_in\_month} \times \text{stay\_duration}
    $$

    Opcionalmente se puede agregar `price_bucket` como proxy de categoría hotelera. A continuación medimos la densidad de cada versión.
    """)
    return


@app.cell
def _(df, pd):
    # Contextos sin price_bucket
    tp2_context_cols = ['destination_final', 'month', 'week_in_month', 'stay_duration']
    tp2_contexts = df.groupby(tp2_context_cols).agg(
        n_records=('price_std', 'count'),
        demand_weight=('count_repeated', 'sum'),
        mean_price=('price_std', 'mean'),
        std_price=('price_std', 'std'),
    ).reset_index()

    # Contextos con price_bucket (si existe)
    if 'price_bucket' in df.columns:
        tp2_contexts_bucket = df.groupby(tp2_context_cols + ['price_bucket']).agg(
            n_records=('price_std', 'count'),
            demand_weight=('count_repeated', 'sum'),
            mean_price=('price_std', 'mean'),
            std_price=('price_std', 'std'),
        ).reset_index()
    else:
        tp2_contexts_bucket = None

    # Resumen de cobertura
    def _summarize(contexts, label):
        total = len(contexts)
        high_conf = (contexts['demand_weight'] >= 30).sum()
        total_demand = contexts['demand_weight'].sum()
        high_conf_demand = contexts.loc[contexts['demand_weight'] >= 30, 'demand_weight'].sum()
        return {
            'segmentacion': label,
            'contextos_totales': total,
            'contextos_n_ge_30': high_conf,
            'pct_contextos_confiables': 100.0 * high_conf / total if total > 0 else 0,
            'demanda_total': total_demand,
            'demanda_en_confiables': high_conf_demand,
            'pct_demanda_confiable': 100.0 * high_conf_demand / total_demand if total_demand > 0 else 0,
        }

    tp2_context_summary = pd.DataFrame([
        _summarize(tp2_contexts, 'destino + mes + semana + estadia'),
    ])
    if tp2_contexts_bucket is not None:
        tp2_context_summary = pd.concat([
            tp2_context_summary,
            pd.DataFrame([_summarize(tp2_contexts_bucket, 'destino + mes + semana + estadia + price_bucket')]),
        ], ignore_index=True)

    print('Densidad de la definición de mercado propuesta')
    print(tp2_context_summary.to_string(index=False))

    # Ejemplo de contextos para un destino popular
    tp2_top_dest = df.groupby('destination_name')['count_repeated'].sum().idxmax()
    tp2_example_contexts = tp2_contexts[
        tp2_contexts['destination_final'] == df.loc[df['destination_name'] == tp2_top_dest, 'destination_final'].iloc[0]
    ].sort_values('demand_weight', ascending=False).head(10)
    print(f'\nTop 10 contextos más densos para {tp2_top_dest}:')
    print(tp2_example_contexts.to_string(index=False))
    return tp2_context_summary, tp2_contexts, tp2_contexts_bucket, tp2_example_contexts, tp2_top_dest


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.5 Distribución de precios dentro de cada segmento

    Exploramos la forma de la distribución de `price_std` dentro de los principales segmentos de mercado. Esto orienta la elección del estadístico de detección para el TP3.
    """)
    return


@app.cell
def _(df, plt, sns, tp2_top_dest):
    # Destinos top por demanda
    tp2_top_dests = (
        df.groupby('destination_name')['count_repeated']
        .sum()
        .sort_values(ascending=False)
        .head(8)
        .index
    )
    tp2_df_top = df[df['destination_name'].isin(tp2_top_dests)].copy()

    # Histograma global vs destinos top
    _fig, _axes = plt.subplots(3, 3, figsize=(15, 12))
    _axes = _axes.flatten()
    sns.histplot(tp2_df_top['price_std'].clip(upper=tp2_df_top['price_std'].quantile(0.99)),
                 bins=60, kde=True, ax=_axes[0], color='steelblue')
    _axes[0].set_title('Global (recortado al p99)')
    _axes[0].set_xlabel('price_std')

    for _i, dest in enumerate(tp2_top_dests[:8], start=1):
        sub = tp2_df_top[tp2_df_top['destination_name'] == dest]
        upper = sub['price_std'].quantile(0.99)
        sns.histplot(sub['price_std'].clip(upper=upper), bins=40, kde=True, ax=_axes[_i], color='teal')
        _axes[_i].set_title(dest)
        _axes[_i].set_xlabel('price_std')

    plt.suptitle('Distribución de price_std por destino canónico (top por demanda)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    # Boxplot por stay_duration y month (global)
    _fig2, _axes2 = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df[df['price_std'] < df['price_std'].quantile(0.99)],
                x='stay_duration', y='price_std', order=['corta', 'media', 'larga'],
                ax=_axes2[0], palette='Set2')
    _axes2[0].set_title('price_std por duración de estadía')

    sns.boxplot(data=df[df['price_std'] < df['price_std'].quantile(0.99)],
                x='month', y='price_std', ax=_axes2[1], palette='coolwarm')
    _axes2[1].set_title('price_std por mes')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, pd, tp2_contexts):
    # Resumen de distribución por segmento de mercado
    tp2_dist_by_segment = tp2_contexts.copy()
    tp2_dist_by_segment['cv'] = tp2_dist_by_segment['std_price'] / tp2_dist_by_segment['mean_price']

    print('Resumen de distribución de precios por contexto de mercado')
    print(tp2_dist_by_segment[['mean_price', 'std_price', 'cv']].describe().round(3).to_string())

    # Contextos con mayor y menor coeficiente de variación
    print('\nContextos con mayor variabilidad relativa (CV):')
    print(tp2_dist_by_segment.nlargest(10, 'cv')[['destination_final', 'month', 'week_in_month', 'stay_duration',
                                                   'n_records', 'demand_weight', 'mean_price', 'std_price', 'cv']]
          .to_string(index=False))

    print('\nContextos con menor variabilidad relativa (CV):')
    print(tp2_dist_by_segment.nsmallest(10, 'cv')[['destination_final', 'month', 'week_in_month', 'stay_duration',
                                                    'n_records', 'demand_weight', 'mean_price', 'std_price', 'cv']]
          .to_string(index=False))
    return tp2_dist_by_segment


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.6 Estadísticos para detectar precios inusualmente bajos

    La consigna del TP2 pide explorar qué estadístico distingue mejor los precios inusualmente bajos. Comparamos cuatro alternativas sobre los datos curados:

    - **Percentil 10 por contexto**: precio por debajo del 10% histórico.
    - **Percentil 25 por contexto**: precio por debajo del cuartil inferior.
    - **Z-score < -1**: precio a más de 1 desvío estándar por debajo de la media del contexto.
    - **IQR (Q1 - 1.5·IQR)**: límite inferior del boxplot.

    No elegimos aún el definitivo; solo comparamos cobertura y correlación entre métodos para orientar el TP3.
    """)
    return


@app.cell
def _(df, np, pd, tp2_contexts):
    # Calcular estadísticos por contexto
    tp2_stats = df.groupby(['destination_final', 'month', 'week_in_month', 'stay_duration']).agg(
        mean_price=('price_std', 'mean'),
        std_price=('price_std', 'std'),
        p10=('price_std', lambda x: x.quantile(0.10)),
        p25=('price_std', lambda x: x.quantile(0.25)),
        p50=('price_std', lambda x: x.quantile(0.50)),
        p75=('price_std', lambda x: x.quantile(0.75)),
        q_lo=('price_std', lambda x: x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25))),
    ).reset_index()

    # Merge con el dataframe original (muestra para performance)
    tp2_sample = df.sample(n=min(200_000, len(df)), random_state=42).copy()
    tp2_eval = tp2_sample.merge(tp2_stats, on=['destination_final', 'month', 'week_in_month', 'stay_duration'], how='left')

    # Máscaras de detección
    tp2_eval['flag_p10'] = tp2_eval['price_std'] <= tp2_eval['p10']
    tp2_eval['flag_p25'] = tp2_eval['price_std'] <= tp2_eval['p25']
    tp2_eval['flag_zscore_lt_minus1'] = (tp2_eval['price_std'] - tp2_eval['mean_price']) / tp2_eval['std_price'] < -1
    tp2_eval['flag_iqr'] = tp2_eval['price_std'] <= tp2_eval['q_lo']

    # Resumen de cobertura
    tp2_method_summary = pd.DataFrame({
        'metodo': ['percentil_10', 'percentil_25', 'z_score_lt_-1', 'iqr_q1_1.5iqr'],
        'n_detectados': [
            tp2_eval['flag_p10'].sum(),
            tp2_eval['flag_p25'].sum(),
            tp2_eval['flag_zscore_lt_minus1'].sum(),
            tp2_eval['flag_iqr'].sum(),
        ],
        'pct_detectados': [
            100.0 * tp2_eval['flag_p10'].mean(),
            100.0 * tp2_eval['flag_p25'].mean(),
            100.0 * tp2_eval['flag_zscore_lt_minus1'].mean(),
            100.0 * tp2_eval['flag_iqr'].mean(),
        ],
    })
    print('Comparación de métodos para detectar precios inusualmente bajos (sobre muestra de 200k)')
    print(tp2_method_summary.to_string(index=False))

    # Correlación entre flags
    tp2_flag_cols = ['flag_p10', 'flag_p25', 'flag_zscore_lt_minus1', 'flag_iqr']
    tp2_corr = tp2_eval[tp2_flag_cols].astype(float).corr()
    print('\nCorrelación entre métodos (sobre registros con contexto estadístico):')
    print(tp2_corr.round(3).to_string())

    # Precio medio detectado por cada método
    tp2_mean_detected = pd.DataFrame({
        'metodo': ['percentil_10', 'percentil_25', 'z_score_lt_-1', 'iqr_q1_1.5iqr'],
        'precio_medio_detectado': [
            tp2_eval.loc[tp2_eval['flag_p10'], 'price_std'].mean(),
            tp2_eval.loc[tp2_eval['flag_p25'], 'price_std'].mean(),
            tp2_eval.loc[tp2_eval['flag_zscore_lt_minus1'], 'price_std'].mean(),
            tp2_eval.loc[tp2_eval['flag_iqr'], 'price_std'].mean(),
        ],
    })
    print('\nPrecio medio de los registros detectados por cada método:')
    print(tp2_mean_detected.to_string(index=False))
    return tp2_corr, tp2_eval, tp2_mean_detected, tp2_method_summary, tp2_stats


@app.cell
def _(plt, sns, tp2_eval):
    # Visualización de la relación entre z-score y percentil dentro de la muestra
    tp2_eval['z_score'] = (tp2_eval['price_std'] - tp2_eval['mean_price']) / tp2_eval['std_price']

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=tp2_eval.sample(n=min(20_000, len(tp2_eval)), random_state=42),
                    x='z_score', y='price_std', alpha=0.3, ax=_axes[0], color='darkblue')
    _axes[0].axvline(-1, color='red', linestyle='--', label='z = -1')
    _axes[0].set_title('Relación entre z-score y price_std')
    _axes[0].legend()

    sns.histplot(tp2_eval['z_score'].dropna().clip(-5, 5), bins=80, kde=True, ax=_axes[1], color='purple')
    _axes[1].axvline(-1, color='red', linestyle='--', label='z = -1')
    _axes[1].set_title('Distribución de z-scores por contexto (clip [-5, 5])')
    _axes[1].legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11.7 Conclusiones del TP2

    1. **Dataset curado**: el pipeline elimina registros con denominador inválido, precios no positivos, noches inconsistentes y aplica winsorización al percentil 99.9. El resultado es un dataset comparable en unidades de precio por habitación-noche-persona.

    2. **Features de contexto**: están construidas las variables que definen el mercado — geografía canónica, temporalidad (mes, semana del mes, día de la semana) y duración de estadía.

    3. **Mapping de destinos**: se aplica `destination_with_nearest.csv` con trazabilidad por `match_level`. La mayor parte de la demanda queda mapeada, aunque una cola larga de geografías administrativas menores permanece como `no_match`.

    4. **Definición de mercado**: usamos `destination_final × month × week_in_month × stay_duration`, opcionalmente con `price_bucket`. La segmentación concentra la mayor parte de la demanda en contextos con suficiente masa estadística (demanda ponderada ≥ 30).

    5. **Distribución de precios**: las distribuciones por destino son asimétricas positivas con cola derecha larga. La variabilidad relativa (CV) cambia fuertemente entre contextos, lo que refuerza la necesidad de segmentar antes de comparar.

    6. **Estadísticos para detección**: el percentil 10 y el z-score < -1 detectan conjuntos parcialmente solapados pero con perfiles distintos. El percentil 10 es más estable en contextos chicos; el z-score es más sensible a la dispersión interna del contexto. La elección final queda para el TP3, donde se evaluará contra baselines ponderados y fallbacks.

    ---

    ### Decisiones metodológicas para el TP3

    - Mantener la geografía canónica como ancla inseparable del mercado.
    - Usar `month`, `week_in_month` y `stay_duration` como dimensiones de contexto.
    - Evaluar `price_bucket` como dimensión opcional, midiendo su impacto en la sensibilidad del detector.
    - Considerar el z-score ponderado por demanda como método principal, con fallback a percentiles cuando el contexto tenga baja muestra.
    """)
    return
