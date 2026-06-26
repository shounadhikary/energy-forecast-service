import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import PROCESSED_DATA_PATH, MODEL_PATH, FEATURE_COLS


TARGET_COL = "consumption"


def load_processed():
    """Load the feature-engineered dataset."""
    df = pd.read_csv(PROCESSED_DATA_PATH)
    return df


def time_based_split(df, test_fraction=0.2):
    """Split chronologically - last 20% is the test set (no shuffling)."""
    split_idx = int(len(df) * (1 - test_fraction))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


def evaluate(name, y_true, y_pred):
    """Print MAE and RMSE for a model."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name:20s} | MAE: {mae:8.2f} | RMSE: {rmse:8.2f}")
    return mae, rmse


def main():
    print("Loading processed data...")
    df = load_processed()

    train, test = time_based_split(df)
    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}\n")

    results = {}

    # --- Baseline: Linear Regression ---
    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    results["LinearRegression"] = (lr, *evaluate("LinearRegression", y_test, lr_pred))

    # --- Strong model: XGBoost ---
    print("Training XGBoost...")
    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        n_jobs=-1,
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    results["XGBoost"] = (xgb, *evaluate("XGBoost", y_test, xgb_pred))

    # --- Pick the best model by RMSE ---
    best_name = min(results, key=lambda k: results[k][2])  # index 2 = rmse
    best_model = results[best_name][0]
    print(f"\nBest model: {best_name}")

    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()