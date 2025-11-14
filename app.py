import pandas as pd
import pickle
import json
from fastapi import FastAPI, UploadFile
import uvicorn

from preprocessing_utils import preprocess_churn_data

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
expected_columns = json.load(open("expected_columns.json"))

app = FastAPI(title="Customer Churn Prediction API")

@app.post("/predict")
async def predict(file: UploadFile):
    df = pd.read_csv(file.file)

    X_processed, _, _, _, customer_ids, _ = preprocess_churn_data(
        df,
        is_train=False,
        scaler=scaler,
        encoders=encoders,
        expected_columns=expected_columns
    )

    predictions = model.predict(X_processed)

    output = pd.DataFrame({
        "CustomerID": customer_ids,
        "ChurnPrediction": predictions
    })

    return output.to_dict(orient="records")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
