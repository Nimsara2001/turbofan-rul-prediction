# 🛠️ Predictive Maintenance (NASA C-MAPSS)

An end-to-end Machine Learning pipeline to predict the **Remaining Useful Life (RUL)** of aircraft turbofan engines using time-series sensor data. 

## 🚀 Key Features

- **Dataset:** Real-world [NASA C-MAPSS dataset](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
- **Modeling:** Compares traditional ML (XGBoost, Random Forest) with Deep Learning (LSTM) for time-series forecasting.
- **MLOps:** Uses **MLflow** for experiment tracking and model logging.
- **Deployment:** Serves the best model via a **FastAPI** REST endpoint, containerized with **Docker**.

## 🚦 Quick Start (API)

Run the RUL prediction web service locally using Docker:

```bash
# Build the image
docker build -t rul-service .

# Run the container
docker run -p 8000:8000 rul-service
```

Example Prediction Request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"data": [ ... 420 flattened scaled feature values ... ]}'
```

## 💻 Project Structure

- `data/`: Raw and processed dataset files.
- `notebooks/`: EDA and model baselines.
- `src/`: Training pipeline, feature engineering, and MLOps scripts.
- `api/`: FastAPI application code.
- `Dockerfile`: Container configuration for deployment.

---
*Predictive Maintenance for Real-World Engineering*
