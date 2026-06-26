import pandas as pd
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def load_data(path=RAW_DATA_PATH):
    """Load CSV, parse datetime, sort chronologically."""
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    df = df.rename(columns={"PJME_MW": "consumption"})
    return df


def add_time_features(df):
    """Extract calendar/time-based features."""
    df["hour"] = df["Datetime"].dt.hour
    df["dayofweek"] = df["Datetime"].dt.dayofweek   # 0=Mon, 6=Sun
    df["month"] = df["Datetime"].dt.month
    df["dayofyear"] = df["Datetime"].dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


def add_lag_features(df):
    """Past consumption values - core of time-series forecasting."""
    df["lag_1h"] = df["consumption"].shift(1)      # 1 hour ago
    df["lag_24h"] = df["consumption"].shift(24)    # same time yesterday
    df["lag_168h"] = df["consumption"].shift(168)  # same time last week
    return df


def add_rolling_features(df):
    """Recent trend - rolling mean/std over last 24 hours."""
    df["roll_mean_24h"] = df["consumption"].shift(1).rolling(window=24).mean()
    df["roll_std_24h"] = df["consumption"].shift(1).rolling(window=24).std()
    return df


def build_features(df):
    """Build all features and drop rows with missing values."""
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = df.dropna().reset_index(drop=True)
    return df


def main():
    print("Loading data...")
    df = load_data()
    print(f"Total rows: {len(df)}")

    print("Building features...")
    df = build_features(df)
    print(f"Rows after feature engineering: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()