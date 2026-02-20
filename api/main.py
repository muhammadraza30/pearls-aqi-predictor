"""
FastAPI backend for AQI Predictor.
Serves prediction data and health check endpoints.
"""
import os
import sys
import json
from pathlib import Path

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(
    title="Karachi AQI Predictor API",
    description="REST API for AQI predictions — Karachi, Pakistan",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PREDICTIONS_CSV = PROJECT_ROOT / "data" / "predictions_3day.csv"
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header:
        return None 
    
    CORRECT_KEY = os.getenv("API_KEY")
    if CORRECT_KEY and api_key_header == CORRECT_KEY:
        return api_key_header
    
    raise HTTPException(status_code=403, detail="Invalid API Key")


@app.get("/api/health")
def health():
    return {"status": "ok", "city": "Karachi"}


@app.get("/api/predictions")
def get_predictions():
    """Return pre-computed AQI predictions for today + next 3 days."""
    if not PREDICTIONS_CSV.exists():
        # Generate on-the-fly
        try:
            from src.inference.predict_3days import main as run_predictions
            run_predictions()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    try:
        df = pd.read_csv(PREDICTIONS_CSV)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read predictions: {e}")
    
    return {"predictions": df.to_dict(orient="records")}


@app.post("/api/predict")
def trigger_prediction(api_key: str = Depends(get_api_key)):
    """Re-generate predictions on demand. Requires X-API-Key header."""
    try:
        from src.inference.predict_3days import main as run_predictions
        run_predictions()
        
        df = pd.read_csv(PREDICTIONS_CSV)
        return {"status": "success", "predictions": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model-version")
def model_version():
    """Return current model version info."""
    version_file = PROJECT_ROOT / "data" / "model_version.json"
    if version_file.exists():
        try:
            with open(version_file) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {"version": 0, "message": "Invalid version file"}
    return {"version": 0, "message": "No models trained yet"}
