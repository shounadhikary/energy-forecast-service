#  Energy Consumption Forecasting Service

An end-to-end machine learning system that forecasts short-term electricity demand using **near real-time grid data**. The project takes a model from a notebook experiment all the way to a deployed, interactive product - demonstrating the full machine-learning lifecycle: data engineering, modeling, API design, testing, and CI/CD.

---

## Overview

The system fetches near real-time hourly electricity demand from the **U.S. Energy Information Administration (EIA)** API, engineers time-series features, and uses a trained **XGBoost** model to forecast future consumption. Predictions are exposed through both a **REST API** (FastAPI) and an **interactive dashboard** (Streamlit) showing live metrics, recent trends, and a 24-hour forecast.

---

## Key Features

- **Live data pipeline** - pulls recent hourly demand from the EIA API.
- **Time-series feature engineering** - lag features (1h, 24h, 168h), rolling statistics, and calendar features (hour, weekday, month, weekend).
- **Model comparison** - a Linear Regression baseline vs. an XGBoost regressor, evaluated with a chronological train/test split.
- **REST API** - a FastAPI service with automatic Swagger documentation and `/health` and `/predict` endpoints.
- **Interactive dashboard** - a Streamlit app with live KPI cards, a recent-trend chart, a recursive 24-hour forecast, and a manual prediction tool.
- **Engineering practices** - modular code, environment-based secret handling, automated tests (pytest), and a GitHub Actions CI pipeline.

---

## Model Performance

Two models were trained on PJME hourly demand data (~145,000 records) and evaluated on a held-out chronological test set:

| Model              | MAE      | RMSE     |
|--------------------|----------|----------|
| Linear Regression  | 976.23   | 1253.40  |
| **XGBoost**        | **316.47** | **426.39** |

XGBoost was selected as the production model. With consumption in the 14,000–62,000 MW range, a mean absolute error of ~316 MW corresponds to roughly **1% error**.

---

## Architecture

```
EIA API (live data)
        │
        ▼
  Data pipeline  ──►  Feature engineering  ──►  Trained XGBoost model
        │                                              │
        │                                              ▼
        │                                    ┌──────────────────┐
        └───────────────────────────────────►  FastAPI REST API │
                                             └──────────────────┘
                                                       │
                                                       ▼
                                          Streamlit dashboard (live)
```

---

## Tech Stack

- **Language:** Python
- **ML:** scikit-learn, XGBoost, pandas, NumPy
- **API:** FastAPI, Uvicorn, Pydantic
- **Dashboard:** Streamlit, Plotly
- **Data source:** EIA Open Data API
- **Testing / CI:** pytest, GitHub Actions

---

## Project Structure

```
energy-forecast-service/
├── api/
│   └── main.py            # FastAPI service
├── src/
│   ├── config.py          # central paths
│   ├── data_prep.py       # data loading + feature engineering
│   ├── train.py           # model training + comparison
│   ├── predict.py         # prediction logic
│   └── live_data.py       # EIA live data fetcher
├── tests/
│   ├── test_data_prep.py
│   └── test_api.py
├── .github/workflows/ci.yml
├── dashboard.py           # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone and set up

```bash
git clone https://github.com/shounadhikary/energy-forecast-service.git
cd energy-forecast-service
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure environment variables

The project reads secrets from a `.env` file in the project root. An example file
(`.env.example`) lists every variable the project uses. Copy it and fill in your
real values:

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

| Variable      | Required | Description                                                                 |
|---------------|----------|-----------------------------------------------------------------------------|
| `EIA_API_KEY` | Yes      | EIA Open Data API key — get a free key at [eia.gov/opendata](https://www.eia.gov/opendata/). |

`.env` is listed in `.gitignore` and must **never** be committed. When deploying to
Streamlit Community Cloud, set `EIA_API_KEY` as a Streamlit secret instead of using `.env`.

### 3. Prepare data and train the model

The training data (`PJME_hourly.csv`) is from the
[Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
dataset on Kaggle. Place it in `data/raw/`, then run:

```bash
cd src
python data_prep.py
python train.py
```

### 4. Run the API

```bash
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

### 5. Run the dashboard

```bash
streamlit run dashboard.py
```

---

## API Usage

**Health check**

```bash
GET /health
→ {"status": "ok"}
```

**Prediction**

```bash
POST /predict
{
  "hour": 14, "dayofweek": 2, "month": 6, "dayofyear": 165, "is_weekend": 0,
  "lag_1h": 40000, "lag_24h": 39000, "lag_168h": 41000,
  "roll_mean_24h": 40000, "roll_std_24h": 800
}
→ {"predicted_consumption_mw": 40123.45}
```

---

## Testing

```bash
pytest -v
```

Tests cover feature engineering correctness and API endpoint behavior. They run automatically on every push via GitHub Actions.

---

## Notes

- Live demand is sourced from the EIA PJM region and scaled to match the PJME training range, since the model was trained on PJME historical data.
- EIA data reflects the provider's standard reporting delay, so the dashboard is **near** real-time rather than instantaneous — consistent with how most grid dashboards operate.

---

## Possible Extensions

- Retrain directly on full-PJM or European (ENTSO-E) grid data.
- Containerize with Docker for reproducible deployment.
- Add deep-learning models (LSTM) for comparison.
- Deploy the dashboard publicly (e.g., Streamlit Community Cloud).
