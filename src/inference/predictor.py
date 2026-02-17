"""
AQI Inference Engine.
Loads trained models (SVR, LightGBM, LSTM) and Scaler for prediction.
"""
import os
import joblib
import numpy as np
import pandas as pd
from src.hopsworks_api import get_project

# Try importing tensorflow for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as load_keras_model
    HAS_TF = True
except ImportError:
    HAS_TF = False

# Constants matching training
FEATURES = ['temp', 'humidity', 'wind_speed', 'clouds', 'aqi_lag_1', 'aqi_lag_3', 'aqi_lag_6', 'aqi_lag_24']

class AQIInferenceEngine:
    def __init__(self, model_path=None, scaler_path=None):
        self.model = None
        self.scaler = None
        self.is_lstm = False
        
        # Default paths
        # Default paths
        # __file__ = src/inference/predictor.py
        # dirname -> src/inference
        # dirname -> src
        # dirname -> ROOT
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_dir = os.path.join(root_dir, "models")
        
        if not model_path:
            # Try finding best model from JSON
            import json
            metrics_path = os.path.join(root_dir, "data/model_metrics.json")
            try:
                with open(metrics_path, "r") as f:
                    meta = json.load(f)
                    best_name = meta.get("best_model", "SVR")
                    if best_name == "LSTM":
                        model_path = os.path.join(models_dir, "LSTM_model.keras")
                    else:
                        model_path = os.path.join(models_dir, f"{best_name}_model.pkl")
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                model_path = os.path.join(models_dir, "SVR_model.pkl")
        if not scaler_path:
            scaler_path = os.path.join(models_dir, "scaler.pkl")

        self.load_resources(model_path, scaler_path)

    def load_resources(self, model_path, scaler_path):
        """Loads Model and Scaler."""
        print(f"🔄 Loading model from {model_path}...")
        try:
            if model_path.endswith(".keras") or model_path.endswith(".h5"):
                if HAS_TF:
                    self.model = load_keras_model(model_path)
                    self.is_lstm = True
                else:
                    raise ImportError("TensorFlow required for Keras models")
            else:
                self.model = joblib.load(model_path)
                self.is_lstm = False
            print("✅ Model loaded.")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")

        print(f"🔄 Loading scaler from {scaler_path}...")
        try:
            self.scaler = joblib.load(scaler_path)
            print("✅ Scaler loaded.")
        except Exception as e:
            print(f"❌ Failed to load scaler: {e}")

    def predict(self, df):
        """
        Predicts AQI for the given dataframe rows.
        Expects columns: temp, humidity, wind_speed, clouds, aqi_lag_1, aqi_lag_3, aqi_lag_6, aqi_lag_24
        """
        if self.model is None or self.scaler is None:
            return np.zeros(len(df))

        # Ensure correct columns
        try:
            X = df[FEATURES].values
        except KeyError as e:
            print(f"❌ Missing columns for prediction: {e}")
            return np.zeros(len(df))

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        if self.is_lstm:
            # Reshape for LSTM [samples, timesteps, features]
            X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            preds = self.model.predict(X_reshaped, verbose=0).flatten()
        else:
            preds = self.model.predict(X_scaled)
            
        return preds




def categorize_aqi(aqi_value):
    """Returns category, color, and emoji for an AQI value."""
    if aqi_value <= 50:
        return {"category": "Good", "color": "#00e400", "emoji": "🟢", "message": "Air quality is satisfactory.", "is_hazardous": False, "level": 1}
    elif aqi_value <= 100:
        return {"category": "Moderate", "color": "#ffff00", "emoji": "🟡", "message": "Air quality is acceptable.", "is_hazardous": False, "level": 2}
    elif aqi_value <= 150:
        return {"category": "Unhealthy for Sensitive Groups", "color": "#ff7e00", "emoji": "🟠", "message": "Members of sensitive groups may experience health effects.", "is_hazardous": True, "level": 3}
    elif aqi_value <= 200:
        return {"category": "Unhealthy", "color": "#ff0000", "emoji": "🔴", "message": "Everyone may begin to experience health effects.", "is_hazardous": True, "level": 4}
    elif aqi_value <= 300:
        return {"category": "Very Unhealthy", "color": "#8f3f97", "emoji": "🟣", "message": "Health warnings of emergency conditions.", "is_hazardous": True, "level": 5}
    else:
        return {"category": "Hazardous", "color": "#7e0023", "emoji": "⚫", "message": "Health alert: everyone may experience more serious health effects.", "is_hazardous": True, "level": 6}
