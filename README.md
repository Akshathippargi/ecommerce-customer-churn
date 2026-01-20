# 📉 E-Commerce Customer Churn Prediction (End-to-End ML Project)

## 📌 Project Overview
This project implements an **end-to-end customer churn prediction system** using **machine learning**, covering the full lifecycle from **raw data preprocessing** to **model training, evaluation, and deployment via Streamlit**.

The goal is to identify customers at risk of churn using historical purchase behavior and provide a **reproducible, production-ready ML pipeline**.

---

## 🎯 Business Problem
Customer churn leads to direct revenue loss in e-commerce businesses. Early identification of high-risk customers enables proactive retention strategies such as targeted discounts, loyalty programs, and personalized outreach.

**Objective:**  
Predict the probability that a customer will churn based on transactional purchase data.

---

## 🗂 Dataset
- **Source:** UCI Machine Learning Repository – Online Retail Dataset  
- **Type:** Transaction-level e-commerce data  
- **Key attributes:**
  - Invoice number
  - Quantity
  - Unit price
  - Invoice date
  - Customer ID

> Raw data files are excluded from the repository to keep it lightweight and reproducible.

---

## 🧠 Churn Definition
A customer is labeled as **churned (1)** if they have **not made any purchase in the last 90 days**, otherwise labeled as **active (0)**.

This mirrors real-world business churn definitions based on customer inactivity.

---

## 🛠 Feature Engineering
Customer-level features were engineered using an **RFM-inspired approach**:

| Feature | Description |
|------|-----------|
| `frequency` | Number of unique purchase invoices |
| `total_units` | Total number of units purchased |
| `monetary_value` | Total spend by the customer |

### ⚠️ Data Leakage Handling
The `recency` feature was initially identified as a source of **target leakage** and was **removed from model training**, ensuring realistic and forward-looking predictions.

---

## 🤖 Model & Techniques
- **Algorithm:** XGBoost Classifier
- **Class imbalance handling:** `scale_pos_weight`
- **Evaluation metrics:**
  - ROC AUC
  - Recall (Churn class)
  - Precision
  - F1-score

---

## 📊 Final Model Performance
