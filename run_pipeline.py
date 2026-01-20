from src.data_preprocessing import load_and_clean_data, save_clean_data
from src.feature_engineering import create_rfm_features, save_features
from src.train_model import train_xgboost
from src.evaluate_model import evaluate

RAW_DATA_PATH = "data/raw/online_retail_09_10.csv"
CLEAN_DATA_PATH = "data/processed/clean_transactions.csv"
FEATURE_PATH = "data/processed/customer_features_rfm.csv"
MODEL_PATH = "models/xgboost_churn_model.pkl"

def main():
    print("STEP 1: Loading and cleaning raw data...")
    df = load_and_clean_data(RAW_DATA_PATH)
    save_clean_data(df, CLEAN_DATA_PATH)
    print("✓ Clean data saved")

    print("STEP 2: Feature engineering (RFM + churn)...")
    rfm = create_rfm_features(df)
    save_features(rfm, FEATURE_PATH)
    print("✓ Features saved")

    print("STEP 3: Training XGBoost model...")
    train_xgboost(FEATURE_PATH, MODEL_PATH)
    print("✓ Model trained and saved")

    print("STEP 4: Evaluating model...")
    evaluate(MODEL_PATH, FEATURE_PATH)

    print("\nPIPELINE COMPLETED SUCCESSFULLY 🎉")

if __name__ == "__main__":
    main()
