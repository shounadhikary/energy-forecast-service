import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parent.parent / "api"))

from main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict():
    payload = {
        "hour": 14, "dayofweek": 2, "month": 6, "dayofyear": 165, "is_weekend": 0,
        "lag_1h": 40000, "lag_24h": 39000, "lag_168h": 41000,
        "roll_mean_24h": 40000, "roll_std_24h": 800,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert "predicted_consumption_mw" in resp.json()