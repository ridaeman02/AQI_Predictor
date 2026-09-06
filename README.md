# AirSight AI

**Live Application:** [airsight-ai-njbj5zndomaymcwgdkuuca.streamlit.app](https://airsight-ai-njbj5zndomaymcwgdkuuca.streamlit.app/)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://airsight-ai-njbj5zndomaymcwgdkuuca.streamlit.app/)

AirSight AI is an intelligent air-quality prediction platform that combines machine learning, weather information, 72-hour forecasting, SHAP explainability, AQI alerts, and actionable outdoor-time recommendations. Designed as a high-end environmental intelligence dashboard, it helps users understand real-time pollution metrics and safely plan outdoor activities.

---

## Features

* **Multi-City AQI Monitoring:** Real-time metrics for major Pakistani cities.
* **AI-Powered Predictions:** Machine learning models trained on historical meteorological and pollutant data.
* **72-Hour Forecasting:** Robust 24-hour, 48-hour, and 72-hour predictive forecasting powered by recursive lagging and live weather API integration.
* **Air Quality Windows:** Smart, activity-based outdoor time recommendations that analyze the forecast to find the safest contiguous blocks of time to be outside.
* **SHAP Explainability:** On-demand, lazy-loaded local explanations revealing exactly which environmental factors drove the AI's prediction.
* **Streamlit Dashboard:** A premium, fully responsive, and highly polished SaaS-style user interface.
* **GitHub Actions MLOps:** Automated background workflows for feature pipelines, training pipelines, and email alerts.
* **Hopsworks Feature Store:** Enterprise-grade feature storage and versioning for model training.

---

## Supported Cities

| City | Supported |
| :--- | :--- |
| Lahore | Yes |
| Karachi | Yes |
| Islamabad | Yes |
| Peshawar | Yes |
| Quetta | Yes |

---

## Machine Learning

The platform leverages an ensemble of machine learning models to capture the complex non-linear relationships between atmospheric conditions and AQI.

| Model | Purpose | Used For |
| :--- | :--- | :--- |
| **Random Forest** | High-accuracy non-linear regression | Primary AQI prediction and SHAP explainability |
| **LSTM** | Sequential deep learning | Time-series specific evaluation and alternative predictions |
| **XGBoost** | Gradient boosted decision trees | Ensemble verification and high-performance inference |
| **Ridge Regression** | Linear baseline | Establishing baseline linear metrics |

*Models are securely serialized locally, allowing the dashboard to run without external ML dependencies.*

---

## Forecasting

The application features a custom, recursive time-series forecasting engine.

### 24-Hour Forecast
Calculates the expected AQI and weather variables for the next 24 hours (Today).
### 48-Hour Forecast
Calculates the expected conditions spanning the next 48 hours (Tomorrow).
### 72-Hour Forecast
Extends predictions out to 72 hours using auto-regressive lags and live weather integration.

**Note:** The system seamlessly merges live OpenWeather 5-day weather data with its own internal AQI lag metrics to generate future predictions, falling back to a diurnal baseline if the API is offline.

---

## Air Quality Windows

This signature feature converts complex 72-hour AQI predictions into actionable, human-readable recommendations. It evaluates periods based on distinct physical exertion profiles:
* General Outdoor
* Walking
* Running
* Cycling
* Outdoor Work

It outputs:
* **Best Window:** The single highest-scoring contiguous time block for the selected day.
* **Alternative Windows:** Backup periods that are safe but slightly less optimal.
* **Avoid Periods:** Times where AQI strictly exceeds safety thresholds for the selected activity.
* **Suitability Score:** A calculated 0-100 score prioritizing AQI safety over weather comfort.

*Note: The output remains strictly an environmental suitability recommendation, not medical advice.*

---

## Explainable AI

AirSight AI integrates **SHAP (SHapley Additive exPlanations)** to build trust and transparency. 

* **Lazy Execution:** SHAP values are computationally expensive. To preserve dashboard performance, explanations are evaluated completely on-demand only when a user expands the "View Important Prediction Factors" UI panel.
* **Separation of Concerns:** The UI strictly separates "Environmental Suitability" (the rule-based logic recommending the window) from "Model Prediction Factors" (the exact feature contributions driving the AI's predicted AQI value).

---

## Weather Data

Predictions rely on core meteorological variables tightly coupled with pollutant concentrations:
* Temperature (°C)
* Humidity (%)
* Wind Speed (m/s)

Future predictions utilize the OpenWeather API to retrieve accurate upcoming weather conditions.

---

## Data Pipeline

```text
Data Collection (APIs/Sensors)
          ↓
Historical Data Aggregation
          ↓
Feature Engineering (Lags, Rolling Means)
          ↓
 Hopsworks Feature Store 
          ↓
  Model Training (RF/LSTM/XGB)
          ↓
  Dashboard (Local Fallback Cache)
```
*Note:* The dashboard relies on `data/processed_features.csv` as a lightning-fast local cache baseline for its recursive forecasting, eliminating the need to hit Hopsworks on every user click.

---

## Project Architecture

```text
                    AIRSIGHT AI
                         │
          ┌──────────────┼──────────────┐
          │              │              │
      Data Layer     ML Layer      MLOps Layer
          │              │              │
       AQI/Data      RF/Ridge/       Hopsworks
       Weather       XGBoost/LSTM    GitHub Actions
          │              │
          └───────┬──────┘
                  │
             Forecasting
                  │
        ┌─────────┼─────────┐
        │         │         │
      SHAP     Air Quality  Alerts
               Windows
                  │
                  ▼
            Streamlit UI
```

---

## Project Structure

```text
AirSight-AI/
│
├── src/
│   ├── dashboard/            # Streamlit UI & Components
│   ├── prediction/           # Core forecasting & model loading
│   ├── recommendation/       # Air Quality Windows logic
│   ├── alerts/               # Email alerting system
│   ├── data_pipeline/        # Data ingestion scripts
│   ├── feature_store/        # Hopsworks feature engineering
│   └── explainability/       # SHAP integration
│
├── models/                   # Serialized ML models (.pkl, .keras)
├── data/                     # Local CSV caches (processed_features.csv)
├── tests/                    # Pytest suite
├── .github/
│   └── workflows/            # MLOps & Alert CI/CD pipelines
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

---

## Installation

### Requirements
* Python 3.9+ 

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AirSight-AI

# Create a virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Environment Variables

The project safely segregates secrets based on functionality using a `.env` file or cloud secrets manager. 

### Dashboard Runtime (Required for Live Forecasts)
* `OPENWEATHER_API_KEY`

### MLOps & Training (Required for GitHub Actions / Retraining)
* `HOPSWORKS_API_KEY`

### Alerting (Required for Background Email Cron)
* `SMTP_PASSWORD`

---

## Running Locally

```bash
python -m streamlit run src/dashboard/app.py
```
Upon execution, Streamlit will boot the application locally (typically at `http://localhost:8501`). The dashboard will instantly load utilizing the cached models and data. 

---

## Streamlit Community Cloud Deployment

AirSight AI is live and deployed on Streamlit Community Cloud:
* **Live Application:** [https://airsight-ai-njbj5zndomaymcwgdkuuca.streamlit.app/](https://airsight-ai-njbj5zndomaymcwgdkuuca.streamlit.app/)

To deploy your own instance:

1. Push the project to GitHub (ensure `models/` and `data/` are committed).
2. Log in to Streamlit Community Cloud and click "Create app".
3. Select your repository and branch.
4. Set the Main file path to: `src/dashboard/app.py`
5. Open Advanced Settings -> **Secrets** and add:
   ```toml
   OPENWEATHER_API_KEY = "your-api-key-here"
   ```
6. Click **Deploy**.

*The dashboard will run perfectly without Hopsworks credentials since it utilizes the local cache for lightning-fast inference.*

---

## Other Deployment Options

| Platform | Compatible | Notes |
| :--- | :--- | :--- |
| **Streamlit Community Cloud** | Yes | Native deployment (Recommended) |
| **Hugging Face Spaces** | Yes | Select Streamlit SDK |
| **Render / Railway** | Yes | Standard Web Service |
| **Linux VPS** | Yes | Standard systemd / PM2 setup |

---

## MLOps

The repository contains three automated GitHub Actions workflows:
1. **Feature Pipeline:** Ingests live data and updates Hopsworks.
2. **Training Pipeline:** Retrains models dynamically based on fresh data.
3. **Alert Pipeline:** A cron-based watchdog that runs `src/alerts/check_alerts.py` to notify users of dangerous upcoming AQI spikes.

---

## Alerting

AirSight AI features an independent alerting daemon.
* **Trigger:** The system predicts an AQI value exceeding predefined health thresholds for the next 24 hours.
* **State Management:** It maintains local state to prevent duplicate/spam emails for the same warning window.
* **Decoupling:** Alerting runs entirely independent of the Streamlit UI, typically executed via GitHub Actions.

---

## Testing

The project maintains a rigorous, automated `pytest` suite covering edge cases, time horizons, SHAP lazy loading, and midnight crossings.

```bash
python -m pytest tests/
```
> **30 tests currently pass.**

---

## Security

* All credentials (`OPENWEATHER_API_KEY`, `HOPSWORKS_API_KEY`, `SMTP_PASSWORD`) are strictly loaded via `os.getenv()`.
* The `.env` file is explicitly ignored in `.gitignore`.
* No hardcoded credentials exist in the source code.

---

## Important Files

| File/Directory | Purpose |
| :--- | :--- |
| `src/dashboard/app.py` | Main Streamlit entry point |
| `src/prediction/forecast.py` | Recursive lag-based forecasting engine |
| `src/recommendation/air_quality_windows.py` | AQI outdoor suitability logic |
| `src/explainability/shap_analysis.py` | Lazy-loaded prediction factors |
| `models/` | Serialized model cache |
| `data/processed_features.csv` | Local fallback baseline data |
| `.github/workflows/` | CI/CD automation & scheduled alerts |
| `tests/` | Unit and integration test suite |

---

## Limitations

* **Forecast Accuracy:** The 72-hour forecast depends heavily on the accuracy of the upstream OpenWeather meteorological predictions.
* **API Limits:** Extreme usage of the live forecast tool may require a premium OpenWeather tier if rate limits are exceeded.
* **Disclaimer:** Recommendations provided by AirSight AI are based purely on environmental suitability algorithms. They do not constitute formal medical advice.

---

## Future Improvements

* Automated UI/E2E testing (e.g., `pytest-playwright`).
* Support for additional global cities.
* Integration of broader satellite aerosol data (e.g., Sentinel-5P).

---

## Development

1. Ensure Python 3.9+ is installed.
2. Setup the virtual environment and install dependencies via `requirements.txt`.
3. Create a `.env` file in the root directory for local testing.
4. Run `python -m pytest tests/` before submitting pull requests.
