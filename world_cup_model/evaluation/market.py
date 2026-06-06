"""
Market comparison and edge identification.

Compare the model's match-level probabilities to bookmaker-implied
probabilities and flag positive-expected-value bets. Kelly sizing converts
those edges into capital allocations.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

from world_cup_model.config import (
    EDGE_THRESHOLD,
    HOST_COUNTRIES,
    KELLY_FRACTION_DIVISOR,
    NAME_MAP,
    ODDS_API_KEY,
    ODDS_SPORT_KEY,
    OUTRIGHT_SPORT_KEY,
    SAMPLE_OUTRIGHT_ODDS,
)
from world_cup_model.model.dixon_coles import predict_match_probabilities


# ---------------------------------------------------------------------------
# Team name alignment
# ---------------------------------------------------------------------------
def standardize_market_team(name: str, extra_map: Optional[dict[str, str]] = None) -> str:
    """Map bookmaker team labels onto model/config naming conventions."""
    mapping = dict(NAME_MAP)
    if extra_map:
        mapping.update(extra_map)
    return mapping.get(name.strip(), name.strip())


def has_odds_api_key(api_key: Optional[str] = None) -> bool:
    key = api_key or ODDS_API_KEY or os.environ.get("ODDS_API_KEY", "")
    return bool(key.strip())


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


def fetch_outright_odds(
    api_key: Optional[str] = None,
    sport: str = OUTRIGHT_SPORT_KEY,
    region: str = "eu",
) -> list[dict]:
    """Fetch tournament-winner outright odds from the-odds-api.com."""
    return fetch_odds(api_key=api_key, sport=sport, region=region, market="outrights")


def load_outright_odds_from_file(path: str) -> dict[str, float]:
    """Load team -> decimal odds from a JSON file."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {
        standardize_market_team(team): float(odds)
        for team, odds in raw.items()
        if not str(team).startswith("_") and float(odds) > 1.0
    }


def outright_response_to_odds(
    odds_response: list[dict],
    extra_map: Optional[dict[str, str]] = None,
) -> dict[str, float]:
    """Parse an Odds API outright response into team -> median decimal odds."""
    collected: dict[str, list[float]] = {}
    for event in odds_response:
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != "outrights":
                    continue
                for outcome in market.get("outcomes", []):
                    team = standardize_market_team(outcome["name"], extra_map=extra_map)
                    price = float(outcome["price"])
                    if price > 1.0:
                        collected.setdefault(team, []).append(price)
    return {team: float(np.median(prices)) for team, prices in collected.items()}


def resolve_market_outright_odds(
    api_key: Optional[str] = None,
    odds_path: Optional[str] = None,
    use_live: bool = True,
) -> tuple[dict[str, float], str]:
    """
    Resolve outright odds from live API, a JSON file, or the bundled sample.

    Returns (team -> decimal odds, source label).
    """
    key = api_key or ODDS_API_KEY or os.environ.get("ODDS_API_KEY", "")
    if use_live and key:
        response = fetch_outright_odds(api_key=key)
        odds = outright_response_to_odds(response)
        if odds:
            return odds, "live (The Odds API)"
        # Fall through if the API returned no outright rows (off-season, etc.).

    if odds_path and os.path.exists(odds_path):
        return load_outright_odds_from_file(odds_path), odds_path

    if os.path.exists(SAMPLE_OUTRIGHT_ODDS):
        return (
            load_outright_odds_from_file(SAMPLE_OUTRIGHT_ODDS),
            f"sample ({SAMPLE_OUTRIGHT_ODDS})",
        )

    raise FileNotFoundError(
        "No market odds available. Set ODDS_API_KEY, pass --market-odds, "
        f"or add {SAMPLE_OUTRIGHT_ODDS}."
    )


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
def _vig_removed_outright_probs(market_outright_odds: dict[str, float]) -> dict[str, float]:
    raw = {team: 1.0 / o for team, o in market_outright_odds.items() if o > 1.0}
    total = sum(raw.values())
    return {team: p / total for team, p in raw.items()} if total > 0 else raw


def build_outright_comparison(
    model_win_probs: dict[str, float],
    market_outright_odds: dict[str, float],
    threshold: float = EDGE_THRESHOLD,
    fractional_divisor: float = KELLY_FRACTION_DIVISOR,
) -> pd.DataFrame:
    """Full model-vs-market outright table with edge and Kelly sizing."""
    market = _vig_removed_outright_probs(market_outright_odds)

    rows = []
    for team, mp in model_win_probs.items():
        market_p = market.get(team, np.nan)
        diff = mp - (market_p if not np.isnan(market_p) else 0.0)
        decimal_odds = market_outright_odds.get(team, np.nan)
        kelly = (
            kelly_fraction(mp, decimal_odds)
            if team in market_outright_odds and not np.isnan(decimal_odds)
            else np.nan
        )
        rows.append(
            {
                "team": team,
                "model_prob": mp,
                "market_prob": market_p,
                "edge": diff if not np.isnan(market_p) else np.nan,
                "decimal_odds": decimal_odds,
                "kelly": kelly,
                "fractional_kelly": (
                    max(kelly / fractional_divisor, 0.0) if not np.isnan(kelly) else np.nan
                ),
            }
        )
    df = pd.DataFrame(rows).sort_values("model_prob", ascending=False).reset_index(drop=True)
    df["flagged"] = df["edge"].abs() > threshold
    return df


def compare_outright_probs(
    model_win_probs: dict[str, float],
    market_outright_odds: dict[str, float],
    threshold: float = EDGE_THRESHOLD,
) -> pd.DataFrame:
    """Compare a team -> P(win cup) model with bookmaker outright decimal odds."""
    return build_outright_comparison(model_win_probs, market_outright_odds, threshold=threshold)


def compute_benchmark_metrics(comparison_df: pd.DataFrame) -> dict[str, Any]:
    """Summary statistics for matched model vs market outright probabilities."""
    matched = comparison_df.dropna(subset=["market_prob"]).copy()
    if matched.empty:
        return {
            "n_model_teams": int(len(comparison_df)),
            "n_matched": 0,
            "n_flagged": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "correlation": np.nan,
            "mean_edge": np.nan,
            "max_positive_edge": np.nan,
            "max_negative_edge": np.nan,
        }

    model = matched["model_prob"].to_numpy()
    market = matched["market_prob"].to_numpy()
    edges = matched["edge"].to_numpy()
    corr = float(np.corrcoef(model, market)[0, 1]) if len(matched) > 1 else np.nan

    return {
        "n_model_teams": int(len(comparison_df)),
        "n_matched": int(len(matched)),
        "n_flagged": int(matched["flagged"].sum()),
        "mae": float(np.mean(np.abs(model - market))),
        "rmse": float(np.sqrt(np.mean((model - market) ** 2))),
        "correlation": corr,
        "mean_edge": float(np.mean(edges)),
        "max_positive_edge": float(np.max(edges)),
        "max_negative_edge": float(np.min(edges)),
    }


def build_model_probs_for_market(
    params,
    market_probs: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Price each bookmaker H2H fixture with the Dixon-Coles model."""
    model_probs: dict[str, dict[str, float]] = {}
    for match_id, market in market_probs.items():
        home = market.get("home_team")
        away = market.get("away_team")
        if not home or not away:
            continue
        if home not in params.attack or away not in params.attack:
            continue
        neutral = (home not in HOST_COUNTRIES) and (away not in HOST_COUNTRIES)
        res = predict_match_probabilities(home, away, params, neutral=neutral)
        model_probs[match_id] = {
            "home_team": home,
            "away_team": away,
            "p_home": res["p_home"],
            "p_draw": res["p_draw"],
            "p_away": res["p_away"],
        }
    return model_probs


def run_h2h_benchmark(
    params,
    api_key: Optional[str] = None,
    threshold: float = EDGE_THRESHOLD,
) -> pd.DataFrame:
    """Compare model H2H prices to live bookmaker lines when fixtures exist."""
    if not has_odds_api_key(api_key):
        return pd.DataFrame()
    response = fetch_odds(api_key=api_key)
    market_probs = market_response_to_probs(response)
    if not market_probs:
        return pd.DataFrame()
    model_probs = build_model_probs_for_market(params, market_probs)
    edges = find_edges(model_probs, market_probs, threshold=threshold)
    return add_kelly_column(edges)


def run_outright_benchmark(
    model_win_probs: dict[str, float],
    api_key: Optional[str] = None,
    odds_path: Optional[str] = None,
    use_live: bool = True,
    threshold: float = EDGE_THRESHOLD,
) -> dict[str, Any]:
    """
    End-to-end outright benchmark: load odds, compare to model, compute metrics.
    """
    market_odds, source = resolve_market_outright_odds(
        api_key=api_key, odds_path=odds_path, use_live=use_live
    )
    comparison = build_outright_comparison(
        model_win_probs, market_odds, threshold=threshold
    )
    metrics = compute_benchmark_metrics(comparison)
    flagged = comparison[comparison["flagged"]].sort_values("edge", ascending=False)
    return {
        "source": source,
        "comparison": comparison,
        "metrics": metrics,
        "flagged_edges": flagged,
        "market_odds": market_odds,
    }


def benchmark_to_records(comparison_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize a comparison DataFrame for JSON export."""
    records = comparison_df.copy()
    for col in ("model_prob", "market_prob", "edge", "decimal_odds", "kelly", "fractional_kelly"):
        if col in records.columns:
            records[col] = records[col].astype(float).round(6)
    return records.to_dict(orient="records")


def save_benchmark_results(
    benchmark: dict[str, Any],
    output_path: str,
) -> str:
    """Persist benchmark comparison + metrics to JSON."""
    payload = {
        "source": benchmark["source"],
        "metrics": benchmark["metrics"],
        "comparison": benchmark_to_records(benchmark["comparison"]),
        "flagged_edges": benchmark_to_records(benchmark["flagged_edges"]),
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


def print_benchmark_summary(benchmark: dict[str, Any], top_k: int = 10) -> None:
    """Human-readable CLI summary of model vs market."""
    metrics = benchmark["metrics"]
    print(f"      Odds source: {benchmark['source']}")
    print(
        f"      Matched teams: {metrics['n_matched']}/{metrics['n_model_teams']} | "
        f"Flagged edges: {metrics['n_flagged']} | "
        f"MAE: {metrics['mae']:.4f} | "
        f"RMSE: {metrics['rmse']:.4f} | "
        f"Corr: {metrics['correlation']:.3f}"
    )

    flagged = benchmark["flagged_edges"]
    if flagged.empty:
        print("      No edges above threshold.")
        return

    print(f"\n      Top {min(top_k, len(flagged))} flagged edges (model vs market):")
    print(
        f"      {'Team':<24s} {'Model':>7s} {'Market':>7s} {'Edge':>7s} "
        f"{'Kelly':>7s} {'Frac.K':>7s}"
    )
    for _, row in flagged.head(top_k).iterrows():
        print(
            f"      {row['team']:<24s} "
            f"{row['model_prob']*100:6.1f}% "
            f"{row['market_prob']*100:6.1f}% "
            f"{row['edge']*100:+6.1f}% "
            f"{row['kelly']*100:6.1f}% "
            f"{row['fractional_kelly']*100:6.1f}%"
        )
