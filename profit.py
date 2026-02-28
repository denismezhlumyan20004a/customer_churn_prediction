"""
profit.py
─────────
Business profit calculation using a Cost Matrix.

Cost Matrix (default assumptions):
  TP: +€280  — churner retained (€300 revenue - €20 campaign cost)
  FP:  -€20  — unnecessary contact
  FN: -€300  — churner missed, customer lost
  TN:   €0   — correct non-action
"""

import numpy as np
import pandas as pd


# ── Default cost parameters ───────────────────────────────────────────────────
DEFAULT_REVENUE   = 300   # € annual value per customer
DEFAULT_CAMP_COST = 20    # € retention campaign cost per customer


def compute_profit(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    revenue_per_customer: float = DEFAULT_REVENUE,
    retention_cost: float = DEFAULT_CAMP_COST,
) -> dict:
    """
    Calculate business profit for a given decision threshold.

    Parameters
    ----------
    y_true  : array of true binary labels
    y_proba : array of predicted churn probabilities
    threshold : decision threshold (predict churn if proba >= threshold)
    revenue_per_customer : annual revenue value per retained customer
    retention_cost : cost of retention campaign per targeted customer

    Returns
    -------
    dict with profit, roi, TP, FP, FN, TN, n_targeted, campaign_cost
    """
    benefit_tp = revenue_per_customer - retention_cost
    cost_fp    = -retention_cost
    cost_fn    = -revenue_per_customer

    y_pred = (y_proba >= threshold).astype(int)

    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())

    profit        = TP * benefit_tp + FP * cost_fp + FN * cost_fn
    campaign_cost = (TP + FP) * retention_cost
    roi           = (profit / campaign_cost * 100) if campaign_cost > 0 else 0.0

    return {
        "threshold": threshold,
        "profit": profit,
        "roi": roi,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "n_targeted": TP + FP,
        "campaign_cost": campaign_cost,
    }


def cost_matrix_table(
    revenue_per_customer: float = DEFAULT_REVENUE,
    retention_cost: float = DEFAULT_CAMP_COST,
) -> pd.DataFrame:
    """Return a human-readable Cost Matrix DataFrame."""
    rows = [
        ("True Positive",  "Churn",    "Churn",    "Churner retained",        f"+€{revenue_per_customer - retention_cost}"),
        ("False Positive", "Churn",    "No churn", "Unnecessary campaign",    f"-€{retention_cost}"),
        ("False Negative", "No churn", "Churn",    "Churner missed",          f"-€{revenue_per_customer}"),
        ("True Negative",  "No churn", "No churn", "No action needed",        "€0"),
    ]
    return pd.DataFrame(rows, columns=["Case", "Predicted", "Actual", "Business Outcome", "Impact"])
