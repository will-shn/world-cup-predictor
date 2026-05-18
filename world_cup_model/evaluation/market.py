"""
Market comparison and edge identification.

Compare the model's match-level probabilities to bookmaker-implied
probabilities and flag positive-expected-value bets. Kelly sizing converts
those edges into capital allocations.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests

from world_cup_model.config import (
    EDGE_THRESHOLD,
    KELLY_FRACTION_DIVISOR,
    ODDS_API_KEY,
    ODDS_SPORT_KEY,
)


# ---------------------------------------------------------------------------
# Odds API client
# ---------------------------------------------------------------------------
def fetch_odds(
    api_key: Optional[str] = None,
    sport: str = ODDS_SPORT_KEY,
    region: str = "eu",
    market: str = "h2h",
) -> list[dict]:
    """Fetch current odds. Returns the raw JSON list from the-odds-api.com."""
    key = api_key or ODDS_API_KEY or os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise RuntimeError(
            "Set ODDS_API_KEY in config.py or your environment to call the Odds API."
        )

    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
    params = {
        "apiKey": key,
        "regions": region,
        "markets": market,
        "oddsFormat": "decimal",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Odds <-> probability conversions
# ---------------------------------------------------------------------------
def odds_to_prob(
    home_odds: float, draw_odds: float, away_odds: float
) -> tuple[float, float, float]:
    """Decimal odds -> vig-removed probability triple summing to 1."""
    raw = np.array([1.0 / home_odds, 1.0 / draw_odds, 1.0 / away_odds])
    raw = raw / raw.sum()
    return float(raw[0]), float(raw[1]), float(raw[2])


def market_response_to_probs(odds_response: list[dict]) -> dict[str, dict[str, float]]:
    """Convert an Odds API JSON response into match_id -> {home/draw/away}."""
    out: dict[str, dict[str, float]] = {}
    for event in odds_response:
        match_id = event.get("id") or f"{event.get('home_team')}_vs_{event.get('away_team')}"
        home_team = event.get("home_team")
        away_team = event.get("away_team")

        # Use the median across bookmakers per outcome for stability.
        home_o, draw_o, away_o = [], [], []
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                if home_team in outcomes:
                    home_o.append(outcomes[home_team])
                if away_team in outcomes:
                    away_o.append(outcomes[away_team])
                if "Draw" in outcomes:
                    draw_o.append(outcomes["Draw"])
        if not (home_o and away_o and draw_o):
            continue
        ph, pd_, pa = odds_to_prob(
            float(np.median(home_o)), float(np.median(draw_o)), float(np.median(away_o))
        )
        out[match_id] = {
            "home_team": home_team,
            "away_team": away_team,
            "p_home": ph,
            "p_draw": pd_,
            "p_away": pa,
            "decimal_home": float(np.median(home_o)),
            "decimal_draw": float(np.median(draw_o)),
            "decimal_away": float(np.median(away_o)),
        }
    return out


# ---------------------------------------------------------------------------
# Edge identification + Kelly
# ---------------------------------------------------------------------------
def find_edges(
    model_probs: dict[str, dict[str, float]],
    market_probs: dict[str, dict[str, float]],
    threshold: float = EDGE_THRESHOLD,
) -> pd.DataFrame:
    """Flag matches where the model and market disagree by more than `threshold`."""
    rows = []
    for match_id, model in model_probs.items():
        if match_id not in market_probs:
            continue
        market = market_probs[match_id]
        for outcome in ("p_home", "p_draw", "p_away"):
            model_p = float(model.get(outcome, 0.0))
            market_p = float(market.get(outcome, 0.0))
            diff = model_p - market_p
            if abs(diff) > threshold:
                rows.append(
                    {
                        "match_id": match_id,
                        "home_team": market.get("home_team", model.get("home_team")),
                        "away_team": market.get("away_team", model.get("away_team")),
                        "outcome": outcome.replace("p_", ""),
                        "model_prob": model_p,
                        "market_prob": market_p,
                        "edge": diff,
                        "decimal_odds": market.get(
                            f"decimal_{outcome.replace('p_', '')}", float("nan")
                        ),
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("edge", ascending=False).reset_index(drop=True)
    return df


def kelly_fraction(model_prob: float, decimal_odds: float) -> float:
    """Optimal Kelly stake as a fraction of bankroll. Negative means no bet."""
    if decimal_odds <= 1.0:
        return 0.0
    b = decimal_odds - 1.0
    p = float(model_prob)
    q = 1.0 - p
    return (b * p - q) / b


def add_kelly_column(
    edges_df: pd.DataFrame,
    fractional_divisor: float = KELLY_FRACTION_DIVISOR,
) -> pd.DataFrame:
    """Append Kelly fraction and fractional-Kelly stake columns."""
    if edges_df.empty:
        return edges_df
    df = edges_df.copy()
    df["kelly"] = df.apply(
        lambda r: kelly_fraction(r["model_prob"], r["decimal_odds"]), axis=1
    )
    df["fractional_kelly"] = df["kelly"] / fractional_divisor
    df["fractional_kelly"] = df["fractional_kelly"].clip(lower=0.0)
    return df


# ---------------------------------------------------------------------------
# Tournament-level outright market comparison
# ---------------------------------------------------------------------------
def compare_outright_probs(
    model_win_probs: dict[str, float],
    market_outright_odds: dict[str, float],
    threshold: float = EDGE_THRESHOLD,
) -> pd.DataFrame:
    """Compare a team -> P(win cup) model with bookmaker outright decimal odds."""
    raw = {team: 1.0 / o for team, o in market_outright_odds.items() if o > 1.0}
    total = sum(raw.values())
    market = {team: p / total for team, p in raw.items()} if total > 0 else raw

    rows = []
    for team, mp in model_win_probs.items():
        market_p = market.get(team, np.nan)
        diff = mp - (market_p if not np.isnan(market_p) else 0.0)
        rows.append(
            {
                "team": team,
                "model_prob": mp,
                "market_prob": market_p,
                "edge": diff,
                "decimal_odds": market_outright_odds.get(team, np.nan),
                "kelly": (
                    kelly_fraction(mp, market_outright_odds[team])
                    if team in market_outright_odds
                    else np.nan
                ),
            }
        )
    df = pd.DataFrame(rows).sort_values("edge", ascending=False).reset_index(drop=True)
    df["flagged"] = df["edge"].abs() > threshold
    return df
