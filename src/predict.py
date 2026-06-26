import joblib
import pandas as pd
from config import MODEL_PATH, FEATURE_COLS

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