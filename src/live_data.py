import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()  # read .env (for local use)

EIA_API_KEY = os.getenv("EIA_API_KEY")

# Fallback to Streamlit secrets when deployed
if not EIA_API_KEY:
    try:
        import streamlit as st
        EIA_API_KEY = st.secrets["EIA_API_KEY"]
    except Exception:
        pass
BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

# We use the PJM region to match the model's training data (AEP is part of PJM)
REGION = "PJM"


def fetch_recent_demand(hours=400):
    """Fetch recent hourly electricity demand for the PJM region from EIA."""
    if not EIA_API_KEY:
        raise RuntimeError("EIA_API_KEY not found. Check your .env file.")

    end = datetime.utcnow()
    start = end - timedelta(hours=hours)

    params = {
        "api_key": EIA_API_KEY,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": REGION,
        "facets[type][]": "D",          # D = Demand
        "start": start.strftime("%Y-%m-%dT%H"),
        "end": end.strftime("%Y-%m-%dT%H"),
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    rows = resp.json()["response"]["data"]

    df = pd.DataFrame(rows)
    df["Datetime"] = pd.to_datetime(df["period"])
    df["consumption"] = pd.to_numeric(df["value"], errors="coerce")
    df["consumption"] = df["consumption"] * 0.45  # scale full PJM down to PJME range


    df = df[["Datetime", "consumption"]].dropna()
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = fetch_recent_demand()
    print(f"Fetched {len(df)} rows")
    print(df.tail())