from pathlib import Path

# Auto-detect the project root folder
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "PJME_hourly.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "energy_features.csv"
MODEL_PATH = BASE_DIR / "models" / "energy_model.joblib"

# Feature columns the model is trained on, in order. Defined once here so that
# training (train.py) and inference (predict.py) can never drift out of sync.
FEATURE_COLS = [
    "hour", "dayofweek", "month", "dayofyear", "is_weekend",
    "lag_1h", "lag_24h", "lag_168h", "roll_mean_24h", "roll_std_24h",
]

# The live EIA feed reports demand for the *entire* PJM interconnection, but the
# model was trained on the PJME sub-region. Empirically PJME is ~45% of total PJM
# load, so we scale the live series by this factor to bring it into the range the
# model expects.
PJM_TO_PJME_SCALE = 0.45

# Fallback day-of-year used by the dashboard's manual prediction tool when the
# user supplies feature values by hand (165 ≈ mid-June, the dataset's reference
# point). The feature has only a minor effect on the prediction.
DEFAULT_DAYOFYEAR = 165