import sys
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parent.parent / "api"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))


def test_health():
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict():
    # API calls predict_one from main's namespace, so patch it there.
    # This avoids needing the real model file in CI.
    with patch("main.predict_one", return_value=40000.0):
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        payload = {
            "hour": 14, "dayofweek": 2, "month": 6, "dayofyear": 165, "is_weekend": 0,
            "lag_1h": 40000, "lag_24h": 39000, "lag_168h": 41000,
            "roll_mean_24h": 40000, "roll_std_24h": 800,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        assert resp.json()["predicted_consumption_mw"] == 40000.0