from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import pickle
import os
from preprocessing_utils import preprocess_churn_data
import json

app = FastAPI(title="Customer Churn Prediction API")

# Load model, scaler, encoders, expected columns
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("expected_columns.json", "r") as f:
    expected_columns = json.load(f)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
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

        # Save full predictions to CSV in current folder
        output_file = os.path.join(os.getcwd(), "predictions.csv")
        predictions.to_csv(output_file, index=False)
        print(f"Predictions saved to: {output_file}")

        # Return small preview to Swagger (first 10 rows)
        preview = predictions.head(10).to_dict(orient="records")
        return JSONResponse(content={
            "message": "Full predictions saved to predictions.csv",
            "predictions_preview": preview
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
