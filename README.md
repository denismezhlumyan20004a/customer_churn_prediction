# 🔁 Customer Churn Prediction — Business Decisioning System

> *Translating ML predictions into profit-driven retention strategy*

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 Business Problem

Customer churn is one of the most costly challenges in subscription and e-commerce businesses. Retaining an existing customer is 5–7x cheaper than acquiring a new one.

This project answers two questions:

> 1. *Which customers are at risk of churning?*
> 2. *At what decision threshold does acting on those predictions generate maximum profit?*

---

## 💰 Cost Matrix

The model is not evaluated on accuracy alone — it is evaluated on **business impact**.

| Scenario | Prediction | Reality | Business Outcome | Impact |
|----------|-----------|---------|------------------|---------:|
| True Positive  | Churn | Churn | Customer retained via campaign | **+€280** |
| False Positive | Churn | No churn | Unnecessary outreach | **-€20** |
| False Negative | No churn | Churn | Customer lost, no action | **-€300** |
| True Negative  | No churn | No churn | No action needed | **€0** |

*Assumptions: avg. customer annual value = €300 · retention campaign cost = €20/customer*

---

## 📂 Project Structure

```
churn-decisioning-system/
│
├── 📓 notebooks/
│   └── Customer_Churn_Prediction_v2.ipynb   ← Main analysis & business case
│
├── 🐍 src/
│   ├── train.py          ← Model training pipeline
│   ├── evaluate.py       ← Standard ML evaluation
│   ├── profit.py         ← Cost matrix & profit calculation
│   └── threshold.py      ← Threshold optimization loop
│
├── 🤖 models/
│   ├── xgb_churn_model.pkl
│   └── normalizer.pkl
│
├── 📊 reports/
│   ├── eda_overview.png
│   ├── model_evaluation.png
│   ├── feature_importance.png
│   ├── profit_vs_threshold.png
│   ├── cumulative_profit_curve.png
│   └── business_summary.txt
│
├── 📁 data/
│   └── data.csv
│
├── requirements.txt
└── README.md
```

---

## 🗂️ Dataset

- **Source**: [Kaggle — Digital Marketing E-commerce Customer Behavior](https://www.kaggle.com/datasets/ermismbatuhan/digital-marketing-ecommerce-customer-behavior)
- **Size**: 3,333 customers · 20 features · binary target (`churn`)
- **Features**: session behavior, purchase patterns, customer service calls, engagement metrics

---

## 🤖 Model

**Algorithm**: XGBoost (Extreme Gradient Boosting)

Trained with `GridSearchCV` (3-fold CV, optimizing ROC-AUC):

| Metric | Baseline XGBoost | Tuned XGBoost |
|--------|-----------------|---------------|
| Accuracy | 91.5% | 92.7% |
| ROC-AUC | ~0.87 | ~0.88 |

---

## 📈 Profit Optimization

The key insight: **the default threshold of 0.5 is not the profit-maximizing threshold.**

By sweeping thresholds from 0.05 to 0.95 and calculating net profit at each point:

- Optimal threshold found at **~0.35** (example — actual value depends on run)
- Profit at default threshold (0.5): €X,XXX
- Profit at optimal threshold: **€Y,YYY** (+Z% improvement)

![Profit vs Threshold](reports/profit_vs_threshold.png)

---

## 💼 Retention Policy — Final Recommendation

> *Target the top ~18% highest-risk customers identified by the model.*

| Metric | Value |
|--------|-------|
| Decision threshold | 0.35 (profit-optimized) |
| Customers targeted | ~18% of base |
| Campaign investment | €X,XXX |
| **Net profit** | **€Y,YYY** |
| **ROI** | **ZZ%** |

---

## 🚀 Quickstart

```bash
# Clone
git clone https://github.com/your-username/churn-decisioning-system.git
cd churn-decisioning-system

# Install
pip install -r requirements.txt

# Run
jupyter notebook notebooks/Customer_Churn_Prediction_v2.ipynb
```

---

## 📦 Requirements

```
xgboost>=1.7
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
```

---

## 📁 Key Files

| File | Description |
|------|-------------|
| `notebooks/Customer_Churn_Prediction_v2.ipynb` | Full pipeline: EDA → model → business metrics |
| `src/profit.py` | Cost matrix & profit calculation functions |
| `src/threshold.py` | Threshold sweep & optimization |
| `reports/business_summary.txt` | Auto-generated executive summary |

---

*Built as a portfolio project demonstrating end-to-end ML + business impact analysis.*
