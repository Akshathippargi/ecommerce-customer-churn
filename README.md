# E-Commerce Customer Churn Prediction (End-to-End ML Project)
## Live Application
The customer churn prediction model is deployed using Streamlit Cloud.

**Live Demo:**  
https://ecommerce-customer-churn-akshat.streamlit.app

## Project Overview
This project implements an end-to-end customer churn prediction system using machine learning. It covers the complete lifecycle from raw data preprocessing to model training, evaluation, and deployment via Streamlit.

The objective is to identify customers at risk of churn using historical purchase behavior and provide a reproducible, production-ready machine learning pipeline.

---

## Business Problem
Customer churn directly impacts revenue in e-commerce businesses. Early identification of churn-prone customers enables proactive retention strategies such as targeted offers, loyalty programs, and personalized engagement.

**Objective:**  
Predict the probability that a customer will churn based on transactional purchase data.

---

## Dataset
- **Source:** UCI Machine Learning Repository – Online Retail Dataset  
- **Type:** Transaction-level e-commerce data  

**Key attributes:**
- Invoice number  
- Quantity  
- Unit price  
- Invoice date  
- Customer ID  

Raw data files are excluded from the repository to keep it lightweight and reproducible.

---

## Churn Definition
A customer is labeled as **churned (1)** if they have **not made any purchase in the last 90 days**, otherwise labeled as **active (0)**.

This definition reflects real-world churn logic based on customer inactivity.

---

## 🛠 Feature Engineering
Customer-level features were engineered using an RFM-inspired approach:

- **frequency** – Number of unique purchase invoices  
- **total_units** – Total quantity of products purchased  
- **monetary_value** – Total spend by the customer  

### Data Leakage Handling
The recency feature was initially identified as a source of target leakage and was removed from model training to ensure realistic, forward-looking predictions.

---

## Model & Technologies Used
- **Model:** XGBoost Classifier  
- **Programming Language:** Python  
- **Data Processing:** Pandas, NumPy  
- **Modeling & Evaluation:** Scikit-learn, XGBoost  
- **Explainability (Offline):** SHAP  
- **Visualization:** Matplotlib, Seaborn  
- **Deployment:** Streamlit  
- **Version Control:** Git, GitHub  

---

## Model Evaluation & Performance
The final model was evaluated using business-relevant metrics.
ROC AUC: ~0.85
Churn Recall: ~82%
Accuracy: ~75%


## Project Structure 
ecommerce-customer-churn/
├── app/
│   └── app.py
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
├── notebooks/
├── run_pipeline.py
├── requirements.txt
└── README.md


### Key Insights
- High recall ensures most churn-prone customers are identified  
- ROC AUC indicates strong ranking performance  
- Metrics are realistic with no target leakage  

---

## Model Explainability
SHAP (SHapley Additive exPlanations) was used in exploratory notebooks to interpret model predictions.

**Key churn drivers identified:**
- Low purchase frequency  
- Lower customer lifetime value  

Explainability was kept offline to maintain Streamlit app stability and performance.

---

## Streamlit Application
A Streamlit web application enables:
- Input of customer purchase behavior  
- Real-time churn probability prediction  
- Business-friendly churn risk categories (Low / Medium / High)

### Run the app locally
```bash
python -m streamlit run app/app.py
