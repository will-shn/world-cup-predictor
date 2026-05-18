"""
Elo rating utilities.

Elo acts as a regularizer for teams with sparse competitive history (Andorra,
San Marino, etc.). We support two ways to use it:

    1. Feature column: attach the Elo of each side at match time and the diff.
       Useful for second-stage models or diagnostic plots.
    2. Bayesian prior: turn current Elo into prior means for Dixon-Coles attack
       and defense parameters via `elo_to_prior`. The DC fitter will then pull
       sparse teams toward those priors.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


ELO_DEFAULT = 1500.0
# Scale that maps an Elo gap into a log-goal-rate gap.
# Empirically, ~600 Elo points ~= one e-fold in match-level goal expectations,
# which keeps Spain-vs-San-Marino style mismatches in a believable range.
ELO_TO_LOG_GOAL = 1.0 / 600.0


def load_elo_data(filepath: str) -> pd.DataFrame:
    """Load an Elo history CSV with columns: team, date, elo."""
    df = pd.read_csv(filepath, parse_dates=["date"])
    required = {"team", "date", "elo"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Elo CSV missing columns: {missing}")
    return df.sort_values(["team", "date"]).reset_index(drop=True)


def get_elo_at_date(
    elo_df: pd.DataFrame,
    team: str,
    match_date: pd.Timestamp,
    default: float = ELO_DEFAULT,
) -> float:
    """Return the most recent Elo for `team` on or before `match_date`."""
    past = elo_df[(elo_df["team"] == team) & (elo_df["date"] <= match_date)]
    if past.empty:
        return default
    return float(past.iloc[-1]["elo"])


def build_feature_matrix(
    df: pd.DataFrame,
    elo_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach home_elo, away_elo, elo_diff features in place (vectorized)."""
    # Sort once, then asof-merge each side. asof merge gives us the latest Elo
    # at or before each match date in O((m + n) log n) instead of O(m * n).
    elo_sorted = elo_df.sort_values("date").reset_index(drop=True)

    df = df.sort_values("date").reset_index(drop=True).copy()
    df["__order"] = np.arange(len(df))

    def _merge_side(side: str) -> pd.Series:
        team_col = f"{side}_team"
        merged = pd.merge_asof(
            df[["date", team_col, "__order"]].sort_values("date"),
            elo_sorted.rename(columns={"team": team_col}),
            on="date",
            by=team_col,
            direction="backward",
        )
        merged = merged.sort_values("__order")
        return merged["elo"].fillna(ELO_DEFAULT).to_numpy()

    df["home_elo"] = _merge_side("home")
    df["away_elo"] = _merge_side("away")
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df = df.drop(columns="__order")
    return df


def current_team_elos(
    elo_df: pd.DataFrame,
    teams: list[str],
    as_of: Optional[pd.Timestamp] = None,
) -> dict[str, float]:
    """Snapshot the latest Elo for each team in `teams` (or default 1500)."""
    if as_of is None:
        as_of = elo_df["date"].max()
    latest = (
        elo_df[elo_df["date"] <= as_of]
        .sort_values("date")
        .groupby("team")
        .tail(1)
        .set_index("team")["elo"]
    )
    return {team: float(latest.get(team, ELO_DEFAULT)) for team in teams}


def elo_to_prior(elos: dict[str, float]) -> dict[str, dict[str, float]]:
    """Convert Elo snapshot into prior attack/defense means.

    Symmetrically split the team's Elo edge between attack and defense parameters
    so that an average team (Elo == mean_elo) has prior 0 attack / 0 defense.
    """
    mean_elo = float(np.mean(list(elos.values()))) if elos else ELO_DEFAULT
    priors = {}
    for team, elo in elos.items():
        delta = (elo - mean_elo) * ELO_TO_LOG_GOAL
        priors[team] = {"attack": +delta / 2.0, "defense": -delta / 2.0}
    return priors
