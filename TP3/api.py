# TP3 — FastAPI Service: API de Clasificación de Ofertas Hoteleras
# Mentoría ID90Travel · Grupo 02 (Jeremías Taran, Martín Gabriel Gómez, Jael Mataloni, Francisco Cisneros)

import os
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Resolver ruta raíz
_current_dir = Path(__file__).resolve().parent
ROOT_DIR = _current_dir.parent if (_current_dir.parent / "config.py").exists() else _current_dir
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

import config
import auxiliary_functions as af

app = FastAPI(
    title="Hotel Deals Detection API — ID90Travel Mentorship",
    description="API REST para evaluar en tiempo real si una tarifa hotelera es inusualmente baja (DEAL) utilizando el detector estadístico Z-Score Log-Normal Híbrido con cascada de rescate jerárquico.",
    version="1.0.0"
)

# Variables globales en memoria para respuestas ultra-rápidas (< 1ms)
MAPPING_DF = None
STATS_L1 = None
STATS_L2 = None
STATS_L3 = None
STATS_L4 = None
GLOBAL_STATS = {
    "median_price": 44.75,
    "mean_log": 3.75,
    "std_log": 0.65,
    "mean_lin": 52.30,
    "std_lin": 38.10
}

def init_engine():
    """Inicializa mapping y tablas de estadísticas contextuales jerárquicas."""
    global MAPPING_DF, STATS_L1, STATS_L2, STATS_L3, STATS_L4, GLOBAL_STATS
    
    data_dir = ROOT_DIR / "data"
    mapping_path = data_dir / "destination_with_nearest.csv"
    sample_path = data_dir / "sample_data_300k.csv.gz"
    
    if mapping_path.exists():
        MAPPING_DF = af.load_destination_mapping(mapping_path)
    
    if sample_path.exists():
        df_raw = pd.read_csv(sample_path, dtype={"city": str, "state": str, "country": str, "country_code": str}, low_memory=False)
        df_mapped = af.apply_destination_mapping(df_raw, MAPPING_DF) if MAPPING_DF is not None else df_raw
        
        # Normalizar
        pax = df_mapped["number_of_adults"] + df_mapped["number_of_kids"].clip(lower=0)
        pax = pax.replace(0, 1)
        df_mapped["price_std"] = df_mapped["avg_price_average"] / (df_mapped["nights"] * df_mapped["number_of_rooms"] * pax)
        
        # Filtros
        mask = (
            (df_mapped["nights"] > 0) & (df_mapped["nights"] <= 30) &
            (df_mapped["number_of_rooms"] > 0) &
            (df_mapped["number_of_adults"] > 0) &
            (df_mapped["avg_price_average"] > 0) &
            (df_mapped["price_std"] >= 5.0) &
            (df_mapped["price_std"] <= 2500.0)
        )
        df = df_mapped[mask].copy()
        
        df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
        df = df.dropna(subset=["date_start"]).copy()
        df["month"] = df["date_start"].dt.month
        df["week_in_month"] = ((df["date_start"].dt.day - 1) // 7) + 1
        df["log_price_std"] = np.log(df["price_std"])
        
        def dur(n):
            if n <= 2: return "corta"
            if n <= 5: return "media"
            return "larga"
        df["stay_duration"] = df["nights"].apply(dur)
        
        def calc_level(cols):
            res = df.groupby(cols).agg(
                demand_weight=("count_repeated", "sum"),
                median_price=("price_std", "median"),
                mean_log=("log_price_std", "mean"),
                std_log=("log_price_std", "std"),
                mean_lin=("price_std", "mean"),
                std_lin=("price_std", "std")
            ).reset_index()
            res = res[res["demand_weight"] >= 30]
            res["std_log"] = res["std_log"].fillna(0.3).replace(0, 0.3)
            res["std_lin"] = res["std_lin"].fillna(1.0).replace(0, 1.0)
            return res
        
        STATS_L1 = calc_level(["destination_final", "month", "week_in_month", "stay_duration"]).set_index(["destination_final", "month", "week_in_month", "stay_duration"])
        STATS_L2 = calc_level(["destination_final", "month", "stay_duration"]).set_index(["destination_final", "month", "stay_duration"])
        STATS_L3 = calc_level(["destination_final", "month"]).set_index(["destination_final", "month"])
        STATS_L4 = calc_level(["destination_final"]).set_index(["destination_final"])
        
        GLOBAL_STATS = {
            "median_price": float(df["price_std"].median()),
            "mean_log": float(df["log_price_std"].mean()),
            "std_log": float(df["log_price_std"].std()),
            "mean_lin": float(df["price_std"].mean()),
            "std_lin": float(df["price_std"].std())
        }
        print("✓ Motor analítico de baselines contextuales inicializado exitosamente.")

@app.on_event("startup")
def startup_event():
    init_engine()

# --- MODELOS PYDANTIC ---

class SearchQuery(BaseModel):
    destination: str = Field(..., description="Ciudad o destino consultado (ej: 'Miami', 'Las Vegas', 'Paris')", example="Miami")
    date_start: str = Field(..., description="Fecha de check-in en formato YYYY-MM-DD", example="2025-11-15")
    nights: int = Field(3, ge=1, le=30, description="Cantidad de noches de estadía", example=3)
    number_of_rooms: int = Field(1, ge=1, le=10, description="Número de habitaciones", example=1)
    number_of_adults: int = Field(2, ge=1, le=10, description="Número de pasajeros adultos", example=2)
    number_of_kids: int = Field(0, ge=0, le=10, description="Número de niños", example=0)
    price: float = Field(..., gt=0, description="Tarifa observada en USD", json_schema_extra={"example": 120.0})
    price_is_per_night: bool = Field(False, description="Si es True, la tarifa ingresada ya es por noche. Si es False, representa el total de la estadía (por defecto).", json_schema_extra={"example": False})

class DealResult(BaseModel):
    destination_searched: str
    destination_final: str
    date_start: str
    nights: int
    price_total_usd: float
    price_std_usd: float
    market_median_usd: float
    z_log: float
    pct_discount: float
    is_deal: bool
    classification: str
    fallback_level: str
    explanation: str

class BatchQuery(BaseModel):
    queries: List[SearchQuery]

class BatchResponse(BaseModel):
    total_evaluated: int
    deals_detected: int
    results: List[DealResult]

# --- LÓGICA DE CLASIFICACIÓN ---

def classify_query(q: SearchQuery) -> DealResult:
    # 1. Normalizar fecha
    try:
        dt = datetime.strptime(q.date_start, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de fecha inválido. Usar YYYY-MM-DD.")
    
    month = dt.month
    week_in_month = ((dt.day - 1) // 7) + 1
    
    # 2. Duración
    if q.nights <= 2:
        stay_duration = "corta"
    elif q.nights <= 5:
        stay_duration = "media"
    else:
        stay_duration = "larga"
        
    # 3. Mapeo de destino
    dest_final = q.destination
    if MAPPING_DF is not None:
        dest_norm = af.strip_accents(q.destination) if hasattr(af, "strip_accents") else q.destination.strip().lower()
        match = MAPPING_DF[MAPPING_DF["city"].astype(str).str.lower() == dest_norm]
        if not match.empty and pd.notna(match.iloc[0].get("nearest_destination_name")):
            dest_final = str(match.iloc[0]["nearest_destination_name"])
            
    # 4. Normalizar precio a unidad estándar (USD por habitación-noche-persona)
    pax = q.number_of_adults + max(0, q.number_of_kids)
    if q.price_is_per_night:
        price_total = q.price * q.nights
        price_std = q.price / (q.number_of_rooms * max(1, pax))
    else:
        price_total = q.price
        price_std = q.price / (q.nights * q.number_of_rooms * max(1, pax))
    log_p = np.log(max(1.0, price_std))
    
    # 5. Cascada de fallbacks
    stats = None
    level = "L5_GlobalFallback"
    
    if STATS_L1 is not None and (dest_final, month, week_in_month, stay_duration) in STATS_L1.index:
        stats = STATS_L1.loc[(dest_final, month, week_in_month, stay_duration)]
        level = "L1_Fino"
    elif STATS_L2 is not None and (dest_final, month, stay_duration) in STATS_L2.index:
        stats = STATS_L2.loc[(dest_final, month, stay_duration)]
        level = "L2_Medio"
    elif STATS_L3 is not None and (dest_final, month) in STATS_L3.index:
        stats = STATS_L3.loc[(dest_final, month)]
        level = "L3_Mensual"
    elif STATS_L4 is not None and (dest_final,) in STATS_L4.index:
        stats = STATS_L4.loc[(dest_final,)]
        level = "L4_Destino"
        
    if stats is not None:
        med = float(stats["median_price"])
        m_log = float(stats["mean_log"])
        s_log = float(stats["std_log"])
    else:
        med = GLOBAL_STATS["median_price"]
        m_log = GLOBAL_STATS["mean_log"]
        s_log = GLOBAL_STATS["std_log"]
        
    # 6. Estadísticos de deal
    z_log = float((log_p - m_log) / s_log)
    pct_discount = float((med - price_std) / med)
    
    is_deal = bool((z_log <= -1.0) and (pct_discount >= 0.20))
    
    if is_deal:
        classification = "DEAL"
        explanation = f"¡Excelente oportunidad! Tarifa estandarizada (${price_std:.2f} USD/noche-pax) inusualmente baja (z_log = {z_log:.2f}) con {pct_discount*100:.1f}% de descuento real sobre la mediana contextual (${med:.2f} USD)."
    elif z_log <= -0.5 and pct_discount >= 0.10:
        classification = "GOOD_PRICE"
        explanation = f"Buen precio: tarifa un {pct_discount*100:.1f}% inferior a la mediana del mercado (${med:.2f} USD)."
    elif z_log >= 1.0:
        classification = "EXPENSIVE"
        explanation = f"Tarifa elevada para este mercado (z_log = +{z_log:.2f}), superior a la mediana de referencia (${med:.2f} USD)."
    else:
        classification = "NORMAL"
        explanation = f"Tarifa dentro del rango habitual esperado para {dest_final} en este período (mediana: ${med:.2f} USD)."
        
    return DealResult(
        destination_searched=q.destination,
        destination_final=dest_final,
        date_start=q.date_start,
        nights=q.nights,
        price_total_usd=round(price_total, 2),
        price_std_usd=round(price_std, 2),
        market_median_usd=round(med, 2),
        z_log=round(z_log, 3),
        pct_discount=round(pct_discount, 3),
        is_deal=is_deal,
        classification=classification,
        fallback_level=level,
        explanation=explanation
    )

# --- ENDPOINTS ---

@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Hotel Deals Detection API",
        "group": "Grupo 02 — DiploDatos 2026",
        "authors": ["Jeremías Taran", "Martín Gabriel Gómez", "Jael Mataloni", "Francisco Cisneros"],
        "status": "ready",
        "methodology": "Z-Score Log-Normal Híbrido (z_log <= -1.0 y descuento >= 20%) con Cascada Jerárquica de Rescate (L1 a L5)",
        "endpoints": {
            "GET /health": "Verificación de estado y salud del servicio",
            "POST /predict_deal": "Evaluar una búsqueda hotelera individual",
            "POST /batch_predict": "Evaluar un lote de búsquedas hoteleras",
            "GET /stats": "Estadísticas globales del motor analítico"
        }
    }

@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "mapping_loaded": MAPPING_DF is not None,
        "l1_contexts_loaded": len(STATS_L1) if STATS_L1 is not None else 0
    }

@app.post("/predict_deal", response_model=DealResult, tags=["Detection"])
def predict_deal(query: SearchQuery):
    """Evalúa si una tarifa hotelera es un deal inusualmente bajo para su contexto de búsqueda."""
    return classify_query(query)

@app.post("/batch_predict", response_model=BatchResponse, tags=["Detection"])
def batch_predict(batch: BatchQuery):
    """Evalúa un conjunto de búsquedas simultáneamente."""
    results = [classify_query(q) for q in batch.queries]
    deals_count = sum(1 for r in results if r.is_deal)
    return BatchResponse(
        total_evaluated=len(results),
        deals_detected=deals_count,
        results=results
    )

@app.get("/stats", tags=["Info"])
def stats():
    return {
        "global_baselines": GLOBAL_STATS,
        "l1_contexts_count": len(STATS_L1) if STATS_L1 is not None else 0,
        "l2_contexts_count": len(STATS_L2) if STATS_L2 is not None else 0,
        "l3_contexts_count": len(STATS_L3) if STATS_L3 is not None else 0,
        "l4_contexts_count": len(STATS_L4) if STATS_L4 is not None else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
