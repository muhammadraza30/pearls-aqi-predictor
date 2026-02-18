# 🌍 Pearls AQI Predictor

A production-ready **Air Quality Index (AQI) Forecasting System** for Karachi, Pakistan. This project leverages advanced machine learning (SVR, LightGBM, LSTM), a robust feature store (Hopsworks), and a premium Streamlit dashboard to provide accurate 4-day AQI forecasts.

![Live Demo](mraza-aqi-predictor.streamlit.app)
---

## 🚀 Features

- **Automated Forecasting**: Generates 96-hour (4-day) AQI predictions using recursive multi-step forecasting.
- **Advanced Modeling**: Trains and evaluates **SVR**, **LightGBM**, and **LSTM** models, automatically selecting the best performer based on RMSE.
- **Feature Store Integration**: Uses **Hopsworks** to manage historical weather and AQI data, ensuring reliable training and inference pipelines.
- **Premium Dashboard**:
    - **Live AQI Gauge**: Interactive accelerometer-style gauge for real-time status.
    - **Smart Cards**: dynamic forecast cards with health advice and color-coded alerts.
    - **Interactive Charts**: Historical trends, pollutant breakdown, and correlation heatmaps using Plotly.
- **Robust CI/CD**: GitHub Actions pipelines for hourly feature ingestion and daily model retraining.
- **Deployment Ready**: Optimized for **Streamlit Cloud** with minimal configuration.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Custom CSS, Glassmorphism UI)
- **Backend**: FastAPI (Python)
- **ML & Data**: TensorFlow/Keras (LSTM), LightGBM, Scikit-Learn (SVR), Pandas
- **Infrastructure**: Hopsworks (Feature Store & Model Registry)
- **Data Source**: Open-Meteo API (Weather & AQI)
- **DevOps**: GitHub Actions (CI/CD)

---

## 📂 Project Structure

```bash
Pearls_AQI_Predictor/
├── .github/workflows/    # CI/CD Pipelines (Feature & Training)
├── api/                  # FastAPI Backend
├── dashboard/            # Streamlit UI Components & Utils
├── data/                 # Local cache for predictions & metrics
├── models/               # Trained model artifacts (.pkl, .keras)
├── src/
│   ├── features/         # Feature Engineering & Pipeline
│   ├── inference/        # Prediction Logic
│   └── models/           # Model Training & Registry
├── app.py                # Main Streamlit Dashboard Entry
├── run.py                # Unified Launcher (API + UI)
├── requirements.txt      # Python Dependencies
└── README.md             # Project Documentation
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.9+
- A [Hopsworks](https://www.hopsworks.ai/) account (Free)
- API Key for Hopsworks

### 1. Clone the Repository
```bash
git clone https://github.com/muhammadraza30/pearls-aqi-predictor.git
cd pearls-aqi-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file in the root directory:
```ini
HOPSWORKS_API_KEY=your_api_key_here
HOPSWORKS_PROJECT_NAME=your_project_name
OPENWEATHER_API_KEY=your_key (optional)
LATITUDE=24.8608
LONGITUDE=67.0104
API_KEY=your_secret_api_key_for_backend
```

### 4. Run the App
Launch both the Backend API and Streamlit Dashboard:
```bash
python run.py
```
- **Dashboard**: [http://localhost:8501](http://localhost:8501)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔄 Pipelines & Automation

This project uses **GitHub Actions** to automate the ML lifecycle:

1.  **Feature Pipeline** (`feature_pipeline.yml`):
    - Runs **hourly**.
    - Fetches live weather/AQI data.
    - Updates Hopsworks Feature Store.
    - Generates fresh predictions.

2.  **Training Pipeline** (`training_pipeline.yml`):
    - Runs **daily**.
    - Retrains models on the latest data.
    - Registers the best model to Hopsworks.

---

## 📊 Model Performance

Metrics are tracked automatically. The best model is selected based on the lowest **RMSE** (Root Mean Squared Error) on the test set.

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| **SVR** | *Dynamic* | *Dynamic* | *Dynamic* |
| **LightGBM** | *Dynamic* | *Dynamic* | *Dynamic* |
| **LSTM** | *Dynamic* | *Dynamic* | *Dynamic* |

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a Pull Request.

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
