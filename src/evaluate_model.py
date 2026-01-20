import pandas as pd
import pickle
from sklearn.metrics import classification_report, roc_auc_score


def evaluate(model_path: str, feature_path: str):
    model = pickle.load(open(model_path, "rb"))
    df = pd.read_csv(feature_path)

    # IMPORTANT: Use same features as training (no recency)
    X = df.drop(["customer_id", "churn", "recency"], axis=1)
    y = df["churn"]

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    print(classification_report(y, y_pred))
    print("ROC AUC:", roc_auc_score(y, y_prob))
