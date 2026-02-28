"""
threshold.py
────────────
Threshold optimization: find the decision threshold that maximizes
business profit, not just classification accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt
from profit import compute_profit


def sweep_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_steps: int = 200,
    t_min: float = 0.05,
    t_max: float = 0.95,
    **profit_kwargs,
) -> dict:
    """
    Sweep thresholds and compute profit at each point.

    Returns
    -------
    dict with arrays: thresholds, profits, rois, n_targeted
    and scalars: best_threshold, best_profit, best_roi
    """
    thresholds  = np.linspace(t_min, t_max, n_steps)
    profits     = []
    rois        = []
    n_targeted  = []

    for t in thresholds:
        m = compute_profit(y_true, y_proba, t, **profit_kwargs)
        profits.append(m["profit"])
        rois.append(m["roi"])
        n_targeted.append(m["n_targeted"])

    best_idx       = int(np.argmax(profits))
    best_threshold = thresholds[best_idx]
    best_metrics   = compute_profit(y_true, y_proba, best_threshold, **profit_kwargs)

    return {
        "thresholds":      thresholds,
        "profits":         np.array(profits),
        "rois":            np.array(rois),
        "n_targeted":      np.array(n_targeted),
        "best_threshold":  best_threshold,
        "best_profit":     best_metrics["profit"],
        "best_roi":        best_metrics["roi"],
        "best_metrics":    best_metrics,
    }


def plot_profit_curve(sweep_result: dict, save_path: str = None) -> None:
    """Plot Profit & ROI vs Decision Threshold."""
    r = sweep_result
    fig, ax1 = plt.subplots(figsize=(11, 5))

    ax1.fill_between(r["thresholds"], r["profits"], alpha=0.12, color="steelblue")
    ax1.plot(r["thresholds"], r["profits"], color="steelblue", lw=2.5, label="Net Profit (€)")
    ax1.axvline(r["best_threshold"], color="#e74c3c", linestyle="--", lw=2,
                label=f'Optimal threshold = {r["best_threshold"]:.2f}')
    ax1.axvline(0.5, color="gray", linestyle=":", lw=1.5, label="Default = 0.50")
    ax1.axhline(0, color="black", lw=0.8)
    ax1.scatter([r["best_threshold"]], [r["best_profit"]], color="#e74c3c", s=100, zorder=5)
    ax1.set_xlabel("Decision Threshold", fontsize=12)
    ax1.set_ylabel("Net Profit (€)", color="steelblue", fontsize=12)

    ax2 = ax1.twinx()
    ax2.plot(r["thresholds"], r["rois"], color="#f39c12", lw=1.5, linestyle=":", alpha=0.7, label="ROI (%)")
    ax2.set_ylabel("ROI (%)", color="#f39c12", fontsize=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Profit & ROI Optimization by Decision Threshold", fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
