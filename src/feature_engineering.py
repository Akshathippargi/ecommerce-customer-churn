import pandas as pd
import os


def create_rfm_features(df: pd.DataFrame, churn_days: int = 90) -> pd.DataFrame:
    snapshot_date = df["invoice_date"].max()

    # RFM features
    rfm = df.groupby("customer_id").agg({
        "invoice_date": lambda x: (snapshot_date - x.max()).days,
        "invoice": "nunique",
        "quantity": "sum",
        "total_amount": "sum"
    }).reset_index()

    rfm.columns = [
        "customer_id",
        "recency",
        "frequency",
        "total_units",
        "monetary_value"
    ]

    # Churn definition
    last_purchase = (
        df.groupby("customer_id")["invoice_date"]
        .max()
        .reset_index()
    )

    last_purchase["days_since_last_purchase"] = (
        snapshot_date - last_purchase["invoice_date"]
    ).dt.days

    last_purchase["churn"] = (
        last_purchase["days_since_last_purchase"] > churn_days
    ).astype(int)

    rfm = rfm.merge(
        last_purchase[["customer_id", "churn"]],
        on="customer_id"
    )

    return rfm


def save_features(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
