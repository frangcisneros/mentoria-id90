#!/usr/bin/env python3
"""Revalidación robusta del TP3 corregido (Grupo 02) — convención única de precio.

Evalúa el servicio corregido (src/hotel_deals de 04_repo_tp3_entregado) sobre los
540 casos del benchmark robusto del mentor
(02_evidencia_benchmark_mentor/robust_test_cases.csv), reutilizando las funciones
de métricas del script original del mentor para que los números sean
estrictamente comparables con su corrida.

Modos de conversión del precio del caso (en el benchmark, price es la tarifa por
noche por habitación del ancla 2025 multiplicada por un factor):

  nightly_total : price_total = price * nights * rooms
                  <- conversión CORRECTA bajo la convención confirmada
                     (avg_price_average = tarifa por noche por habitación)
  multiply      : price_total = price * nights * rooms * persons
                  (modo 'multiply' original del mentor)
  direct        : price_total = price
                  (modo 'direct' original del mentor)

La evaluación es en proceso (DealDetector), equivalente al endpoint
POST /evaluate: api/main.py sólo reenvía el payload al mismo DealDetector.

Uso (desde la raíz de correccion_tp3_y_consignas_tp4):
    python 06_correcciones_tp3/revalidacion_robusta_corregida.py
    python 06_correcciones_tp3/revalidacion_robusta_corregida.py         --artifacts 04_repo_tp3_entregado/artifacts/sample         --out-dir 06_correcciones_tp3
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
REPO = BASE / "04_repo_tp3_entregado"
sys.path.insert(0, str(REPO / "src"))

MODES = ["nightly_total", "multiply", "direct"]
SEGMENTS = ["scenario_type", "stay_bucket", "persons_bucket", "country_bin"]


def load_mentor_module():
    """Importa robust_validation_tp3.py del mentor como módulo (métricas idénticas)."""
    path = BASE / "02_evidencia_benchmark_mentor" / "robust_validation_tp3.py"
    spec = importlib.util.spec_from_file_location("robust_validation_tp3", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # necesario para que @dataclass resuelva el módulo
    spec.loader.exec_module(mod)
    return mod


def build_queries(cases: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Convierte los casos del benchmark a queries de la API bajo cada modo."""
    nights = cases["nights"].fillna(1).astype(int).clip(lower=1)
    rooms = cases["rooms"].fillna(1).astype(int).clip(lower=1)
    adults = cases["adults"].fillna(2).astype(int)
    kids = cases["kids"].fillna(0).astype(int)
    persons = (adults + kids).clip(lower=1)
    price = cases["price"].astype(float)

    if mode == "nightly_total":
        price_total = price * nights * rooms
    elif mode == "multiply":
        price_total = price * nights * rooms * persons
    elif mode == "direct":
        price_total = price
    else:
        raise ValueError(f"modo desconocido: {mode}")

    q = pd.DataFrame({
        "city": cases["city"].astype(object).where(cases["city"].notna(), None),
        "state": cases["state"].astype(object).where(cases["state"].notna(), None),
        "country": cases["country"].astype(object).where(cases["country"].notna(), None),
        "country_code": cases["country_code"].astype(str).str.upper(),
        "date_start": cases["date_start"].astype(str),
        "nights": nights,
        "adults": adults,
        "kids": kids,
        "rooms": rooms,
        "price_total": price_total.round(2),
    })
    return q


def normalize_pred(decision) -> str | None:
    """Misma normalización que normalize_pred_g2 del mentor."""
    d = str(decision).lower()
    if d == "deal":
        return "deal"
    if d in {"no_deal", "insufficient_data", "invalid_price", "invalid_input"}:
        return "no_deal"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(BASE / "02_evidencia_benchmark_mentor" / "robust_test_cases.csv"))
    ap.add_argument("--artifacts", default=str(REPO / "artifacts" / "sample"))
    ap.add_argument("--out-dir", default=str(BASE / "06_correcciones_tp3"))
    args = ap.parse_args()

    from hotel_deals.service import DealDetector

    mentor = load_mentor_module()
    cases = pd.read_csv(args.cases)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = DealDetector(args.artifacts)
    print(f"Artefactos: {args.artifacts}")
    print(f"Parámetros servidos: {detector.params}")

    metrics_rows, seg_rows = [], []
    for mode in MODES:
        q = build_queries(cases, mode)
        resp = detector.evaluate_frame(q)
        res = pd.DataFrame({
            "case_id": cases["case_id"],
            "scenario_type": cases["scenario_type"],
            "expected_label": cases["expected_label"],
            "expected_norm": cases["expected_label"].map(mentor.normalize_label),
            "factor": cases["factor"],
            "ctx_support": cases["ctx_support"],
            "stay_bucket": cases["stay_bucket"],
            "persons_bucket": cases["persons_bucket"],
            "country_bin": cases["country_bin"],
            "g2_price_total_enviado": q["price_total"],
            "g2_decision": resp["decision"],
            "g2_pred_norm": resp["decision"].map(normalize_pred),
            "g2_price_std": resp["price_std"],
            "g2_baseline_median": resp["baseline_median"],
            "g2_discount": resp["discount"],
            "g2_z_log": resp["z_log"],
            "g2_baseline_level": resp["baseline_level"],
            "g2_baseline_n_records": resp["baseline_n_records"],
        })
        res.to_csv(out_dir / f"robust_api_results_corregido_{mode}.csv", index=False)

        st = mentor.calc_stats(res, "g2_pred_norm")
        f1_l, f1_u = mentor.bootstrap_ci(res, "g2_pred_norm", "f1")
        rec_l, rec_u = mentor.bootstrap_ci(res, "g2_pred_norm", "recall")
        metrics_rows.append({
            "group": "grupo_02_corregido",
            "g2_price_mode": mode,
            "n_total_labeled": st.n_total,
            "n_scored": st.n_scored,
            "coverage": st.coverage,
            "tp": st.tp, "tn": st.tn, "fp": st.fp, "fn": st.fn,
            "accuracy": st.accuracy,
            "precision": st.precision,
            "recall": st.recall,
            "f1": st.f1,
            "fpr": st.fpr,
            "f1_ci_low": f1_l, "f1_ci_high": f1_u,
            "recall_ci_low": rec_l, "recall_ci_high": rec_u,
        })

        for seg_col in SEGMENTS:
            for seg_val, seg_df in res.groupby(seg_col, dropna=False):
                s = mentor.calc_stats(seg_df, "g2_pred_norm")
                seg_rows.append({
                    "group": "grupo_02_corregido",
                    "g2_price_mode": mode,
                    "segment": seg_col,
                    "segment_value": seg_val,
                    "n_total_labeled": s.n_total,
                    "n_scored": s.n_scored,
                    "coverage": s.coverage,
                    "accuracy": s.accuracy,
                    "precision": s.precision,
                    "recall": s.recall,
                    "f1": s.f1,
                    "fpr": s.fpr,
                })
        print(f"modo={mode}: acc={st.accuracy} prec={st.precision} rec={st.recall} f1={st.f1} fpr={st.fpr}")

    mdf = pd.DataFrame(metrics_rows)
    sdf = pd.DataFrame(seg_rows)
    mdf.to_csv(out_dir / "robust_metrics_global_corregido.csv", index=False)
    sdf.to_csv(out_dir / "robust_metrics_by_segment_corregido.csv", index=False)
    print("\n== Métricas globales (servicio corregido) ==")
    print(mdf.round(4).to_string(index=False))
    print(f"\nArchivos escritos en {out_dir}")


if __name__ == "__main__":
    main()
