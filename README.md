# 🔁 Customer Churn Prediction — Business Decisioning System

> *From model selection to profit-driven retention strategy*

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 Business Problem

Customer churn is one of the most costly challenges in e-commerce. Retaining an existing customer is 5–7x cheaper than acquiring a new one — yet most churn models are evaluated on accuracy rather than financial impact.

This project builds a **retention decisioning system** that answers:

> 1. *Which model best identifies at-risk customers — and why?*
> 2. *At what decision threshold does acting on those predictions generate maximum profit?*

---

## 💰 Cost Matrix

All models are evaluated on **business impact**, not accuracy alone.

| Scenario | Prediction | Reality | Business Outcome | Impact |
|----------|-----------|---------|------------------|---------:|
| True Positive  | Churn    | Churn    | Customer retained via campaign | **+€280** |
| False Positive | Churn    | No churn | Unnecessary outreach           | **-€20**  |
| False Negative | No churn | Churn    | Customer lost, no intervention | **-€300** |
| True Negative  | No churn | No churn | No action needed               | **€0**    |

*Assumptions: avg. customer annual value = €300 · retention campaign cost = €20/customer*

---

## 🧭 Model Selection Strategy

Three models representing different interpretability–performance tradeoffs:

| Model | Role | Key Characteristic |
|-------|------|-------------------|
| **Logistic Regression** | Interpretable baseline | Linear, fast to explain to stakeholders |
| **Random Forest** | Non-linear ensemble | Robust, minimal tuning needed |
| **XGBoost** | Production model | Best AUC + calibrated probabilities |

**Selection criterion: maximum net profit after threshold optimisation — not AUC.**

---

## 📊 Model Comparison Results

| Model | ROC-AUC | Opt. Threshold | Net Profit | ROI |
|-------|---------|----------------|------------|-----|
| Logistic Regression | ~0.81 | — | €X,XXX | XX% |
| Random Forest | ~0.86 | — | €X,XXX | XX% |
| **XGBoost (tuned)** | **~0.88** | **0.3X** | **€X,XXX** | **XX%** |

*Exact values generated at runtime*

### Why XGBoost?
- Best ROC-AUC and highest net profit
- `scale_pos_weight` handles class imbalance directly
- Well-calibrated probabilities → reliable threshold optimisation
- Industry standard for tabular data in production

---

## 📂 Project Structure

```
churn-decisioning-system/
│
├── 📓 notebooks/
│   └── Customer_Churn_Prediction_v3.ipynb
│
├── 🐍 src/
│   ├── profit.py        ← Cost matrix & profit calculation
│   └── threshold.py     ← Threshold optimisation loop
│
├── 🤖 models/
│   ├── xgb_churn_model.pkl
│   ├── normalizer.pkl
│   └── policy_config.pkl
│
├── 📊 reports/
│   ├── eda_overview.png
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   ├── profit_vs_threshold.png
│   ├── cumulative_profit_curve.png
│   ├── feature_importance.png
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
- **Features**: session behaviour, purchase patterns, customer service calls, engagement metrics

---

## 📈 Key Visualisations

| Chart | What it shows |
|-------|--------------|
| ROC Curves (all models) | AUC comparison across 3 candidates |
| Net Profit Bar Chart | Business impact per model |
| Confusion Matrices | TP/FP/FN at profit-optimised threshold |
| Profit vs Threshold | Why 0.5 is not the optimal threshold |
| Cumulative Profit Curve | Exact customer segment to target |
| Feature Importance | Top drivers of churn |

---

## 💼 Retention Policy — Final Recommendation

> *Target the top ~18% highest-risk customers identified by XGBoost.*

| Metric | Value |
|--------|-------|
| Decision threshold | 0.3X (profit-optimised) |
| Campaign investment | €X,XXX |
| **Net profit** | **€X,XXX** |
| **ROI** | **XX%** |

---

## 🚀 Quickstart

```bash
git clone https://github.com/your-username/churn-decisioning-system.git
cd churn-decisioning-system
pip install -r requirements.txt
jupyter notebook notebooks/Customer_Churn_Prediction_v3.ipynb
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
jupyter>=1.0
```

---

*Portfolio project — end-to-end ML pipeline with business impact analysis.*
