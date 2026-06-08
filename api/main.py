from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from contextlib import asynccontextmanager

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model and scaler
    ml_models["model"] = joblib.load("models/xgb_model.pkl")
    try:
        ml_models["scaler"] = joblib.load("models/scaler.pkl")
    except FileNotFoundError:
        print("Warning: scaler.pkl not found. Predictions might be unscaled!")
        ml_models["scaler"] = None
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

class PredictionRequest(BaseModel):
    data: list[float]

@app.post("/predict")
def predict_rul(request: PredictionRequest):
    data = request.data
    model = ml_models.get("model")
    scaler = ml_models.get("scaler")

    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    expected_features = model.n_features_in_

    if len(data) != expected_features:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_features} values but got {len(data)}"
        )

    if scaler:
        num_original_features = scaler.n_features_in_
        window_size = expected_features // num_original_features
        
        # Reshape to (window_size, num_original_features) to apply scaling
        reshaped_data = np.array(data).reshape(window_size, num_original_features)
        
        # Scale using the loaded StandardScaler
        scaled_data = scaler.transform(reshaped_data)
        
        # Flatten back to (1, expected_features) for the model
        input_array = scaled_data.flatten().reshape(1, -1)
    else:
        input_array = np.array(data).reshape(1, -1)

    prediction = model.predict(input_array)

    return {"predicted_rul": float(prediction[0])}
