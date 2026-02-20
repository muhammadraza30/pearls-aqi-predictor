"""
Training pipeline for Advanced Time-Series AQI Models (SVR, LightGBM, LSTM).
Implements recursive hourly forecasting strategy with lag features.
"""
import os
import sys
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from lightgbm import LGBMRegressor
# Try importing tensorflow, handles absence if cpu/gpu mismatch
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    HAS_TF = True
except ImportError:
    HAS_TF = False

from src.hopsworks_api import get_project
# from src.features.feature_engineering import create_lag_features
from src.features.preprocessing import fit_scaler, scale_features

# Features to use (including lags)
WEATHER_FEATURES = ['temp', 'humidity', 'wind_speed', 'clouds']
LAG_FEATURES = ['aqi_lag_1', 'aqi_lag_3', 'aqi_lag_6', 'aqi_lag_24']
FEATURES = WEATHER_FEATURES + LAG_FEATURES
TARGET = 'target'  # Next Hour AQI

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def build_lstm_model(input_dim):
    """Builds a simple LSTM model."""
    model = Sequential()
    # Input shape: (timesteps=1, features=input_dim)
    model.add(Input(shape=(1, input_dim)))
    model.add(LSTM(64, activation='tanh', return_sequences=False))  # Tanh is more stable
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    
    # Use Huber loss for regression stability on large values
    # Clipnorm prevents exploding gradients
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mse'])
    return model


def train_and_eval(model_name, model, X_train, y_train, X_test, y_test, is_lstm=False):
    """Trains and evaluates a model."""
    print(f"🚀 Training {model_name}...")
    
    if is_lstm:
        # Reshape for LSTM: [samples, timesteps, features]
        X_train_r = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_r = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        model.fit(X_train_r, y_train, epochs=50, batch_size=32, verbose=0)
        preds = model.predict(X_test_r).flatten()
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"   Shape: Train={X_train.shape}, Test={X_test.shape}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R2: {r2:.4f}")
    
    return model, {"RMSE": rmse, "MAE": mae, "R2": r2}


def run_training():
    project = get_project()
    fs = project.get_feature_store()
    
    print("🔄 Fetching training data (Hourly)...")
    try:
        aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
        query = aqi_fg.select_all()
        # Fetching as pandas df
        df = query.read()
    except Exception as e:
        print(f"❌ Failed to fetch data: {e}")
        return

    print(f"📊 Raw Data Shape: {df.shape}")
    
    # Drop any rows with NaNs that might have slipped through
    df = df.dropna()
    print(f"📊 Data Shape after dropping NaNs: {df.shape}")
    
    # # 1. Feature Engineering
    # df_processed = create_lag_features(df)
    # print(f"📊 Processed Data Shape (with Lags): {df_processed.shape}")
    
    if len(df) < 100:
        print("❌ Not enough data for time-series training. Run backfill!")
        return

    # 2. Train/Test Split (Time Series - no shuffle)
    train_size = int(len(df) * 0.85)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values
    X_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values
    
    # 3. Scaling (Critical for SVR/LSTM)
    scaler, X_train_scaled = fit_scaler(X_train, save_path=f"{MODELS_DIR}/scaler.pkl")
    X_test_scaled = scale_features(X_test, scaler)
    
    metrics = {}
    
    # 4. Train Models
    
    # --- SVR ---
    svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    trained_svr, metric_svr = train_and_eval("SVR", svr_model, X_train_scaled, y_train, X_test_scaled, y_test)
    metrics["SVR"] = metric_svr
    joblib.dump(trained_svr, f"{MODELS_DIR}/SVR_model.pkl")
    
    # --- LightGBM ---
    lgbm_model = LGBMRegressor(n_estimators=1000, learning_rate=0.05)
    trained_lgbm, metric_lgbm = train_and_eval("LightGBM", lgbm_model, X_train_scaled, y_train, X_test_scaled, y_test)
    metrics["LightGBM"] = metric_lgbm
    joblib.dump(trained_lgbm, f"{MODELS_DIR}/LightGBM_model.pkl")
    
    # --- LSTM ---
    if HAS_TF:
        lstm_model = build_lstm_model(len(FEATURES))
        trained_lstm, metric_lstm = train_and_eval("LSTM", lstm_model, X_train_scaled, y_train, X_test_scaled, y_test, is_lstm=True)
        metrics["LSTM"] = metric_lstm
        # Save Keras model
        lstm_model.save(f"{MODELS_DIR}/LSTM_model.keras")
    else:
        print("⚠️ TensorFlow not installed. Skipping LSTM.")

    # 5. Select Best Model
    best_model_name = min(metrics, key=lambda k: metrics[k]["RMSE"])
    print(f"🏆 Best Model: {best_model_name} (RMSE: {metrics[best_model_name]['RMSE']:.4f})")
    
    # Save Metrics
    os.makedirs("data", exist_ok=True)
    with open("data/model_metrics.json", "w") as f:
        json.dump({"models": metrics, "best_model": best_model_name, "updated_at": str(datetime.now())}, f)
    # 6. Register Models to Hopsworks
    print("☁️ Registering models to Hopsworks...")
    
    # Register Scaler
    try:
        mr = project.get_model_registry()
        
        # SVR
        svr_model_meta = mr.python.create_model(
            name="aqi_svr_model",
            metrics=metrics["SVR"],
            description="SVR (RBF) for AQI Forecasting"
        )
        svr_model_meta.save(f"{MODELS_DIR}/SVR_model.pkl")
        
        # LightGBM
        lgbm_model_meta = mr.python.create_model(
            name="aqi_lightgbm_model",
            metrics=metrics["LightGBM"],
            description="LightGBM for AQI Forecasting"
        )
        lgbm_model_meta.save(f"{MODELS_DIR}/LightGBM_model.pkl")
        
        # LSTM
        if HAS_TF and "LSTM" in metrics:
            lstm_model_meta = mr.tensorflow.create_model(
                name="aqi_lstm_model",
                metrics=metrics["LSTM"],
                description="LSTM for AQI Forecasting"
            )
            lstm_model_meta.save(f"{MODELS_DIR}/LSTM_model.keras")
            
        print("✅ Models registered successfully.")
        
    except Exception as e:
        print(f"❌ Failed to register models: {e}")


if __name__ == "__main__":
    run_training()
