import pickle
import pandas as pd


def predict_churn(model_path: str, customer_features: dict) -> float:
    model = pickle.load(open(model_path, "rb"))

    df = pd.DataFrame([customer_features])
    df = df.drop(columns=["recency"], errors="ignore")


    return prob
