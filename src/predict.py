import joblib
import pandas as pd
from config import MODEL_PATH

FEATURE_COLS = [
    "hour", "dayofweek", "month", "dayofyear", "is_weekend",
    "lag_1h", "lag_24h", "lag_168h", "roll_mean_24h", "roll_std_24h",
]

_model = None


def get_model():
    """Load the saved model once and reuse it."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_one(features: dict) -> float:
    """Take a dict of features, return a single consumption prediction."""
    model = get_model()
    row = pd.DataFrame([[features[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)
    prediction = model.predict(row)[0]
    return float(prediction)