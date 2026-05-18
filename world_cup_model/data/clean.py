"""
Data cleaning pipeline.

Three responsibilities:
    1. Filter to competitive matches (drop friendlies).
    2. Standardize team names so they join cleanly against Elo / draw configs.
    3. Apply exponential time decay weights for the Dixon-Coles likelihood.

The single entry point is build_clean_dataset(filepath, decay_rate).
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from world_cup_model.config import (
    COMPETITIVE_TOURNAMENTS,
    DEFAULT_TOURNAMENT_WEIGHT,
    MIN_DATE,
    NAME_MAP,
    TOURNAMENT_WEIGHTS,
)
from world_cup_model.data.fetch import load_raw_data


def filter_competitive(df: pd.DataFrame) -> pd.DataFrame:
    """Drop friendlies and unsupported competitions."""
    return df[df["tournament"].isin(COMPETITIVE_TOURNAMENTS)].copy()


def filter_min_date(df: pd.DataFrame, min_date: str = MIN_DATE) -> pd.DataFrame:
    """Restrict to matches on or after `min_date`."""
    cutoff = pd.Timestamp(min_date)
    return df[df["date"] >= cutoff].copy()


def standardize_names(df: pd.DataFrame, extra_map: Optional[dict] = None) -> pd.DataFrame:
    """Apply NAME_MAP (plus optional overrides) to both team columns."""
    mapping = dict(NAME_MAP)
    if extra_map:
        mapping.update(extra_map)
    df = df.copy()
    df["home_team"] = df["home_team"].replace(mapping)
    df["away_team"] = df["away_team"].replace(mapping)
    return df


def apply_decay(
    df: pd.DataFrame,
    decay_rate: float,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Add `days_ago` and exponential `weight` columns."""
    if reference_date is None:
        reference_date = pd.Timestamp(date.today())
    df = df.copy()
    df["days_ago"] = (reference_date - df["date"]).dt.days.clip(lower=0)
    df["weight"] = np.exp(-decay_rate * df["days_ago"].astype(float))
    return df


def apply_tournament_weights(
    df: pd.DataFrame,
    weights_map: Optional[dict] = None,
    default_weight: float = DEFAULT_TOURNAMENT_WEIGHT,
) -> pd.DataFrame:
    """Multiply per-match `weight` by a competition-importance factor.

    World Cup finals -> high weight; minor qualifiers -> low weight. This is
    what stops a team from gaming the rating by farming weak qualifiers.
    """
    weights_map = weights_map if weights_map is not None else TOURNAMENT_WEIGHTS
    df = df.copy()
    df["tournament_weight"] = (
        df["tournament"].map(weights_map).fillna(default_weight).astype(float)
    )
    df["weight"] = df["weight"].astype(float) * df["tournament_weight"]
    return df


def build_clean_dataset(
    filepath: str,
    decay_rate: float,
    min_date: str = MIN_DATE,
    competitive_only: bool = True,
    extra_name_map: Optional[dict] = None,
    use_tournament_weights: bool = True,
) -> pd.DataFrame:
    """End-to-end clean: load, filter, standardize, weight."""
    df = load_raw_data(filepath)
    if competitive_only:
        df = filter_competitive(df)
    df = filter_min_date(df, min_date)
    df = standardize_names(df, extra_name_map)
    df = apply_decay(df, decay_rate)
    if use_tournament_weights:
        df = apply_tournament_weights(df)
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def clean_in_memory(
    df: pd.DataFrame,
    decay_rate: float,
    min_date: str = MIN_DATE,
    competitive_only: bool = True,
    extra_name_map: Optional[dict] = None,
    use_tournament_weights: bool = True,
) -> pd.DataFrame:
    """Same as build_clean_dataset but starting from an in-memory DataFrame."""
    if competitive_only:
        df = filter_competitive(df)
    df = filter_min_date(df, min_date)
    df = standardize_names(df, extra_name_map)
    df = apply_decay(df, decay_rate)
    if use_tournament_weights:
        df = apply_tournament_weights(df)
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df.sort_values("date").reset_index(drop=True)
