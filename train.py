import pandas as pd
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from preprocessing_utils import preprocess_churn_data

df = pd.read_csv("customer_churn_dataset-training-master.csv")

X, y, scaler, encoders, ids, expected_cols = preprocess_churn_data(
    df, is_train=True
)

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))

with open("expected_columns.json", "w") as f:
    json.dump(expected_cols, f)

print("Training complete. Artifacts saved.")
