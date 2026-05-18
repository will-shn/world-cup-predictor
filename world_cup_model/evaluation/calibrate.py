"""
Model calibration and backtesting utilities.

Brier and log-loss are the headline metrics. The reliability diagram shows
*where* the model is over- or under-confident across the probability spectrum.
backtest() refits on pre-cutoff data only and evaluates against held-out
matches, so we can sanity-check that the workflow generalizes across
tournaments rather than overfitting to past results.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")              # safe default for headless / CI use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from world_cup_model.config import OUTPUT_DIR
from world_cup_model.model.dixon_coles import (
    DixonColesParams,
    fit_model,
    predict_match_probabilities,
)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Multi-class Brier score.

    y_true is one-hot encoded across K classes; y_pred is a probability vector
    over the same K classes. Returns the mean squared error per row.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        return float(np.mean((y_pred - y_true) ** 2))
    return float(np.mean(np.sum((y_pred - y_true) ** 2, axis=1)))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute Brier and log-loss; print + return a small report dict."""
    bs = brier_score(y_true, y_pred)
    eps = 1e-12
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    if y_true.ndim == 1:
        ll = log_loss(y_true, y_pred_clipped, labels=[0, 1])
    else:
        ll = log_loss(
            np.argmax(y_true, axis=1),
            y_pred_clipped,
            labels=list(range(y_true.shape[1])),
        )
    print(f"Brier Score: {bs:.4f}")
    print(f"Log-Loss:    {ll:.4f}")
    return {"brier": bs, "log_loss": float(ll), "n": int(len(y_pred))}


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------
def reliability_diagram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None,
) -> str:
    """Plot a reliability diagram. Returns the figure path."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    bins = np.linspace(0, 1, n_bins + 1)
    bin_means: list[float] = []
    bin_actuals: list[float] = []
    bin_counts: list[int] = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_pred >= lo) & (y_pred < hi if i < n_bins - 1 else y_pred <= hi)
        if mask.sum() > 0:
            bin_means.append(float(y_pred[mask].mean()))
            bin_actuals.append(float(y_true[mask].mean()))
            bin_counts.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(bin_means, bin_actuals, "o-", label="Model")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    for x, y, n in zip(bin_means, bin_actuals, bin_counts):
        ax.annotate(f"n={n}", (x, y), fontsize=8, alpha=0.7, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "reliability_diagram.png")
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
def _predictions_for_matches(
    matches: pd.DataFrame, params: DixonColesParams
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (n_matches, 3) probability matrix and one-hot ground truth.

    Columns = [home_win, draw, away_win].
    """
    probs = np.zeros((len(matches), 3))
    actual = np.zeros((len(matches), 3))
    for i, (_, row) in enumerate(matches.iterrows()):
        neutral = bool(row.get("neutral", False))
        res = predict_match_probabilities(
            row["home_team"], row["away_team"], params, neutral=neutral
        )
        probs[i] = [res["p_home"], res["p_draw"], res["p_away"]]

        hg = int(row["home_score"])
        ag = int(row["away_score"])
        if hg > ag:
            actual[i, 0] = 1.0
        elif hg == ag:
            actual[i, 1] = 1.0
        else:
            actual[i, 2] = 1.0
    return probs, actual


def backtest(
    full_df: pd.DataFrame,
    cutoff_date: str,
    test_window_end: Optional[str] = None,
    plot: bool = True,
) -> dict:
    """Refit DC on `date < cutoff_date`, evaluate on the following window."""
    cutoff = pd.Timestamp(cutoff_date)
    train = full_df[full_df["date"] < cutoff].copy()
    test = full_df[full_df["date"] >= cutoff].copy()
    if test_window_end is not None:
        test = test[test["date"] <= pd.Timestamp(test_window_end)]

    if train.empty:
        raise ValueError("No training data before cutoff.")
    if test.empty:
        raise ValueError("No test data on or after cutoff.")

    print(f"Backtest: train={len(train)} matches, test={len(test)} matches "
          f"(cutoff={cutoff_date})")

    params = fit_model(train)
    probs, actual = _predictions_for_matches(test, params)
    metrics = evaluate_predictions(actual, probs)

    if plot:
        # Use the predicted probability for the actually-realized outcome to
        # build a reliability diagram for the model "as called".
        realized_idx = np.argmax(actual, axis=1)
        realized_prob = probs[np.arange(len(probs)), realized_idx]
        # ground truth is 1 by construction here, so reliability bins compare
        # realized_prob against 1: that is trivially 1.0. Instead build a flat
        # representation across all three classes.
        y_true_flat = actual.ravel()
        y_pred_flat = probs.ravel()
        path = reliability_diagram(
            y_true_flat,
            y_pred_flat,
            title=f"Reliability (test from {cutoff_date})",
            save_path=os.path.join(OUTPUT_DIR, f"reliability_{cutoff_date}.png"),
        )
        metrics["reliability_plot"] = path

    metrics["params"] = params
    metrics["n_train"] = len(train)
    metrics["n_test"] = len(test)
    return metrics
