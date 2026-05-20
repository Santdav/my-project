# 📉 Customer Churn Prediction

> A full end-to-end machine learning pipeline that predicts customer churn from telecom data — covering EDA, feature engineering, model training, explainability, and a REST API serving layer.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [📉 Customer Churn Prediction](#-customer-churn-prediction)
  - [📋 Table of Contents](#-table-of-contents)
  - [Overview](#overview)
  - [Dataset](#dataset)
  - [Pipeline](#pipeline)
    - [1. Exploratory Data Analysis](#1-exploratory-data-analysis)
    - [2. Feature Engineering](#2-feature-engineering)
    - [3. Modeling](#3-modeling)
    - [4. Threshold Selection](#4-threshold-selection)
  - [Results](#results)
  - [Explainability](#explainability)
  - [API](#api)
    - [Run locally](#run-locally)
    - [Endpoint](#endpoint)
  - [Setup \& Usage](#setup--usage)
  - [Next Steps](#next-steps)
  - [License](#license)

---

## Overview

<!--
Briefly explain the business problem:
- Why does churn matter?
- What does this model predict?
- What is the output and how would it be used in practice?
-->

---

## Dataset

| Property | Value |
|---|---|
| Source | [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Rows | 7,043 |
| Features | 20 |
| Target | `Churn` (Yes / No) |
| Class balance | ~26% churn, ~74% retained |

<!--
Add any notes about data quality issues you found during EDA:
- Missing values
- Encoding inconsistencies (e.g. TotalCharges as string)
- Any rows dropped and why
-->


---

## Pipeline

### 1. Exploratory Data Analysis
<!--
Summarize key findings:
- Churn rate overall and by segment (contract type, tenure, charges)
- Distributions of key features
- Notable correlations
-->

### 2. Feature Engineering
<!--
Document engineered features:
- e.g. tenure buckets, charge-per-month ratio, service count aggregation
- Encoding strategy (OHE vs ordinal)
- Scaling approach (StandardScaler / MinMaxScaler)
-->

### 3. Modeling
<!--
Models trained and compared:
- Baseline: Logistic Regression
- Tree-based: Random Forest, XGBoost
- Tuning: GridSearchCV / Optuna
- Imbalance strategy: class_weight, SMOTE, threshold adjustment
-->

| Model | AUC-ROC | F1 (churn) | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |

### 4. Threshold Selection
<!--
Explain why you chose your decision threshold:
- Default 0.5 vs tuned threshold
- Precision-Recall tradeoff in business context
- Cost matrix if applicable
-->

---

## Results

<!--
Highlight the best model:
- Final metrics on test set
- Confusion matrix summary
- ROC curve (embed image from reports/figures/)
-->

---

## Explainability

<!--
SHAP analysis summary:
- Top features driving churn (global importance)
- Example of a single prediction explained (local SHAP waterfall)
- Business interpretation of findings
-->

---

## API

The trained model is served via a **FastAPI** endpoint.

### Run locally

```bash
uvicorn api.main:app --reload
```

### Endpoint

```
POST /predict
```

**Request body:**
```json
{
  "tenure": 12,
  "monthly_charges": 70.5,
  "contract": "Month-to-month",
  "...": "..."
}
```

**Response:**
```json
{
  "churn_probability": 0.83,
  "churn_prediction": true
}
```

---

## Setup & Usage

```bash
# Clone the repo
git clone https://github.com/Santdav/churn-prediction.git
cd churn-prediction

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter lab
```

---

## Next Steps

- [ ] EDA notebook
- [ ] Feature engineering pipeline
- [ ] Baseline model
- [ ] XGBoost + hyperparameter tuning
- [ ] SHAP explainability
- [ ] FastAPI serving layer
- [ ] Monitoring / data drift detection (future)

---

## License

MIT — feel free to use, fork, and build on this.