# TP3 — Test suite para la API de Detección de Deals
# Mentoría ID90Travel · Grupo 02

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from api import app, init_engine

# Inicializar motor antes de tests
init_engine()
client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "Grupo 02" in data["group"]

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_predict_deal_low_price():
    # Una tarifa de $30 USD por 3 noches para 2 adultos en Miami ($5 USD/noche-pax) debe ser claramente un DEAL
    query = {
        "destination": "Miami",
        "date_start": "2025-11-15",
        "nights": 3,
        "number_of_rooms": 1,
        "number_of_adults": 2,
        "number_of_kids": 0,
        "price": 30.0
    }
    response = client.post("/predict_deal", json=query)
    assert response.status_code == 200
    data = response.json()
    assert data["is_deal"] is True
    assert data["classification"] == "DEAL"
    assert data["price_std_usd"] == 5.0
    assert data["pct_discount"] >= 0.20
    assert data["z_log"] <= -1.0

def test_predict_deal_expensive_price():
    # Una tarifa de $1200 USD por 2 noches para 1 adulto en Orlando ($600 USD/noche-pax) no debe ser DEAL
    query = {
        "destination": "Orlando",
        "date_start": "2025-08-10",
        "nights": 2,
        "number_of_rooms": 1,
        "number_of_adults": 1,
        "number_of_kids": 0,
        "price": 1200.0
    }
    response = client.post("/predict_deal", json=query)
    assert response.status_code == 200
    data = response.json()
    assert data["is_deal"] is False
def test_predict_deal_price_per_night():
    # Una tarifa de $10 USD por noche para 1 habitación y 2 adultos en Miami ($5 USD/noche-pax)
    query = {
        "destination": "Miami",
        "date_start": "2025-11-15",
        "nights": 3,
        "number_of_rooms": 1,
        "number_of_adults": 2,
        "number_of_kids": 0,
        "price": 10.0,
        "price_is_per_night": True
    }
    response = client.post("/predict_deal", json=query)
    assert response.status_code == 200
    data = response.json()
    assert data["is_deal"] is True
    assert data["price_std_usd"] == 5.0
    assert data["price_total_usd"] == 30.0

def test_batch_predict():
    batch = {
        "queries": [
            {
                "destination": "Miami",
                "date_start": "2025-11-15",
                "nights": 3,
                "number_of_rooms": 1,
                "number_of_adults": 2,
                "number_of_kids": 0,
                "price": 30.0
            },
            {
                "destination": "Orlando",
                "date_start": "2025-08-10",
                "nights": 2,
                "number_of_rooms": 1,
                "number_of_adults": 1,
                "number_of_kids": 0,
                "price": 1200.0
            }
        ]
    }
    response = client.post("/batch_predict", json=batch)
    assert response.status_code == 200
    data = response.json()
    assert data["total_evaluated"] == 2
    assert data["deals_detected"] >= 1

if __name__ == "__main__":
    pytest.main(["-v", __file__])
