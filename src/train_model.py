import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def train_xgboost(feature_path: str, model_path: str):
    df = pd.read_csv(feature_path)

    X = df.drop(["customer_id", "churn", "recency"], axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model
