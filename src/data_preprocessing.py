import pandas as pd
import os


def load_and_clean_data(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path, encoding="ISO-8859-1")

    # Fix BOM + standardize columns
    df.columns = (
        df.columns
          .str.replace("ï»¿", "", regex=False)
          .str.strip()
          .str.lower()
    )

    df = df.rename(columns={
        "invoiceno": "invoice",
        "invoicedate": "invoice_date",
        "unitprice": "price",
        "customerid": "customer_id"
    })

    # Basic cleaning
    df = df.dropna(subset=["customer_id"])
    df = df[df["quantity"] > 0]
    df = df[df["price"] > 0]

    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["total_amount"] = df["quantity"] * df["price"]

    return df


def save_clean_data(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
