import sys
from pathlib import Path

# Allow importing from the src-folder
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_one

app = FastAPI(title="Energy Consumption Forecasting API")


class FeaturesInput(BaseModel):
    hour: int
    dayofweek: int
    month: int
    dayofyear: int
    is_weekend: int
    lag_1h: float
    lag_24h: float
    lag_168h: float
    roll_mean_24h: float
    roll_std_24h: float


@app.get("/health")
def health():
    """Simple check to confirm the service is running."""
    return {"status": "ok"}


@app.post("/predict")
def predict(features: FeaturesInput):
    """Return an energy consumption forecast for the given features."""
    prediction = predict_one(features.model_dump())
    return {"predicted_consumption_mw": round(prediction, 2)}
