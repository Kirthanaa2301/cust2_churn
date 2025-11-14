from fastapi import FastAPI, UploadFile, File
import pandas as pd
import pickle
from preprocessing_utils import preprocess_churn_data

app = FastAPI(title="Customer Churn Prediction API")

# Load model, scaler, encoders, expected columns
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

import json
with open("expected_columns.json", "r") as f:
    expected_columns = json.load(f)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded CSV
    df = pd.read_csv(file.file)

    # Preprocess
    X_processed, _, _, _, customer_ids, _ = preprocess_churn_data(
        df,
        is_train=False,
        scaler=scaler,
        encoders=encoders,
        expected_columns=expected_columns
    )

    # Make predictions
    model_predictions = model.predict(X_processed)

    # Prepare predictions DataFrame
    predictions = pd.DataFrame({
        "CustomerID": customer_ids,
        "PredictedChurn": model_predictions
    })

    # Save full predictions to CSV
    predictions.to_csv("predictions.csv", index=False)

    # Return small preview to Swagger (first 10 rows)
    return {"predictions_preview": predictions.head(10).to_dict(orient="records"),
            "message": "Full predictions saved to predictions.csv"}
