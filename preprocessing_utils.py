import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_churn_data(df, is_train=True, scaler=None, encoders=None, expected_columns=None):

    df = df.copy()

    customer_ids = df['CustomerID'] if 'CustomerID' in df.columns else None

    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if is_train:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in categorical_cols:
            if col in encoders:
                classes = list(encoders[col].classes_)
                df[col] = df[col].apply(
                    lambda val: encoders[col].transform([val])[0]
                    if val in classes else -1
                )

    y = None
    if 'Churn' in df.columns:
        y = df['Churn']
        df.drop(columns=['Churn'], inplace=True)

    if 'CustomerID' in df.columns:
        df.drop(columns=['CustomerID'], inplace=True)

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if is_train:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    if is_train:
        expected_columns = list(df.columns)
    else:
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_columns]

    return df, y, scaler, encoders, customer_ids, expected_columns
