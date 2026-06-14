from pathlib import Path

# Auto-detect the project root folder
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "PJME_hourly.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "energy_features.csv"
MODEL_PATH = BASE_DIR / "models" / "energy_model.joblib"