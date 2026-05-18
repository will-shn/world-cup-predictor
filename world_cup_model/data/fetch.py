"""
Data acquisition utilities.

Primary path: load a pre-built CSV of historical international results
(the Mart Jürisoo "international football results" dataset on Kaggle is the
recommended source - it ships columns `date, home_team, away_team,
home_score, away_score, tournament, city, country, neutral`).

Secondary path: fetch fresh results from football-data.org for a single
competition. Requires a free API token in the FOOTBALL_DATA_API_KEY env var.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import requests


REQUIRED_COLUMNS = (
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the Kaggle CSV of historical international results.

    Raises a descriptive error if the file is missing or columns are off.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Could not find {filepath}. Download the Kaggle "
            "'international football results' dataset and save it there, "
            "or call generate_synthetic_results() for a smoke test."
        )

    df = pd.read_csv(filepath, parse_dates=["date"])
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}. "
            f"Got columns: {list(df.columns)}"
        )
    return df


def fetch_football_data_org(
    competition: str = "WC",
    season: Optional[int] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch recent matches for one competition from football-data.org.

    Useful for topping up your CSV with the most recent qualifiers. Not used
    by the default pipeline because the free tier has tight rate limits.
    """
    api_key = api_key or os.environ.get("FOOTBALL_DATA_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "Set FOOTBALL_DATA_API_KEY in your environment to call this API."
        )

    url = f"https://api.football-data.org/v4/competitions/{competition}/matches"
    headers = {"X-Auth-Token": api_key}
    params = {}
    if season is not None:
        params["season"] = season
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    matches = response.json().get("matches", [])

    rows = []
    for m in matches:
        if m.get("status") != "FINISHED":
            continue
        score = m.get("score", {}).get("fullTime", {})
        rows.append(
            {
                "date": pd.to_datetime(m["utcDate"]).normalize(),
                "home_team": m["homeTeam"]["name"],
                "away_team": m["awayTeam"]["name"],
                "home_score": score.get("home"),
                "away_score": score.get("away"),
                "tournament": m.get("competition", {}).get("name", competition),
                "neutral": False,
            }
        )
    return pd.DataFrame(rows)


def generate_synthetic_results(
    n_teams: int = 48,
    n_matches: int = 1500,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a synthetic dataset for smoke tests and CI.

    Real Dixon-Coles fits should use the actual historical CSV.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    teams = [f"Team_{i:02d}" for i in range(n_teams)]
    true_strength = rng.normal(0, 0.4, size=n_teams)

    dates = pd.date_range("2018-01-01", "2025-12-31", freq="D")
    rows = []
    for _ in range(n_matches):
        h_idx, a_idx = rng.choice(n_teams, size=2, replace=False)
        lam_h = float(np.exp(1.0 + true_strength[h_idx] - true_strength[a_idx] + 0.25))
        lam_a = float(np.exp(1.0 + true_strength[a_idx] - true_strength[h_idx]))
        rows.append(
            {
                "date": rng.choice(dates),
                "home_team": teams[h_idx],
                "away_team": teams[a_idx],
                "home_score": int(rng.poisson(lam_h)),
                "away_score": int(rng.poisson(lam_a)),
                "tournament": "FIFA World Cup qualification",
                "neutral": False,
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
