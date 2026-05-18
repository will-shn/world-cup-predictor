"""
Download Elo ratings from eloratings.net.

The site does not offer a CSV export button, but it serves the same data the
web UI uses as plain TSV files. No browser scraping is required.

Key files on https://www.eloratings.net/
    en.teams.tsv   - 3-letter code -> English display name (+ URL slug)
    World.tsv      - current ratings for all teams
    {Team_Slug}.tsv - match-by-match rating history for one national team

Run from the project root:

    python -m world_cup_model.data.fetch_elo
    python -m world_cup_model.data.fetch_elo --current-only   # fast snapshot
    python -m world_cup_model.data.fetch_elo --max-teams 20  # smoke test

Output: world_cup_model/data_files/elo_ratings.csv
        columns: team, date, elo
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
import unicodedata
from typing import Iterable, Optional

import pandas as pd
import requests

from world_cup_model.config import DATA_DIR, ELO_CSV, NAME_MAP

BASE_URL = "https://www.eloratings.net"
REQUEST_TIMEOUT = 30
DEFAULT_DELAY = 0.12  # be polite; ~30s for all ~240 teams


def page_name(display_name: str) -> str:
    """Match eloratings.net `pageName()` (spaces -> underscores, strip accents)."""
    if not display_name:
        return ""
    text = display_name
    text = text.replace(" ", "_")
    for old, new in (
        ("àáâãäå", "a"),
        ("ç", "c"),
        ("èéêë", "e"),
        ("ìíîï", "i"),
        ("òóôõö", "o"),
        ("ùúûü", "u"),
        ("ñ", "n"),
    ):
        for ch in old:
            text = text.replace(ch, new)
    # Fallback for any remaining non-ASCII (e.g. Côte d'Ivoire)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if ord(c) < 128)
    return text


def _normalize_number(value: str) -> Optional[float]:
    if value is None:
        return None
    value = value.strip().replace("\u2212", "-").replace("−", "-")
    if not value or value in ("-", "–"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def download_text(path: str, session: Optional[requests.Session] = None) -> str:
    """GET a path relative to BASE_URL and return decoded text."""
    session = session or requests.Session()
    url = f"{BASE_URL}/{path.lstrip('/')}"
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return response.text


def load_en_teams(session: Optional[requests.Session] = None) -> pd.DataFrame:
    """Load code -> display name -> URL slug from en.teams.tsv."""
    text = download_text("en.teams.tsv", session=session)
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        code = parts[0].strip()
        if code.endswith("_loc") or not code:
            continue
        display = parts[1].strip() if len(parts) > 1 else code
        rows.append(
            {
                "code": code,
                "name": display,
                "page": page_name(display),
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["code"]).reset_index(drop=True)


def model_team_name(elo_display_name: str) -> str:
    """Align eloratings names with Mart Jürisoo / config NAME_MAP conventions."""
    return NAME_MAP.get(elo_display_name, elo_display_name)


def parse_team_history_tsv(
    text: str,
    team_name: str,
    team_code: str,
) -> pd.DataFrame:
    """Parse one national-team history file into (team, date, elo) rows.

    Each match row stores *two* post-match ratings:
        parts[3]  home code   -> parts[10] home Elo after
        parts[4]  away code   -> parts[11] away Elo after

    The old parser always read parts[10], so whenever the focal team played
    away it recorded the opponent's rating (e.g. Egypt 2165 = Spain's Elo).
    """
    records = []
    team_code = team_code.strip().upper()
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 12:
            continue
        try:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
        except ValueError:
            continue

        try:
            match_date = pd.Timestamp(dt.date(year, month, day))
        except ValueError:
            continue

        home_code = parts[3].strip().upper()
        away_code = parts[4].strip().upper()
        if home_code == team_code:
            elo_after = _normalize_number(parts[10])
        elif away_code == team_code:
            elo_after = _normalize_number(parts[11])
        else:
            # Row is not about this team (should not happen in team files).
            continue
        if elo_after is None:
            continue

        records.append(
            {
                "team": team_name,
                "date": match_date,
                "elo": elo_after,
            }
        )

    if not records:
        return pd.DataFrame(columns=["team", "date", "elo"])
    df = pd.DataFrame(records)
    return df.sort_values("date").reset_index(drop=True)


def fetch_team_history(
    page: str,
    team_name: str,
    team_code: str,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Download and parse {page}.tsv for one team."""
    session = session or requests.Session()
    try:
        text = download_text(f"{page}.tsv", session=session)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return pd.DataFrame(columns=["team", "date", "elo"])
        raise
    return parse_team_history_tsv(text, team_name, team_code)


def reconcile_with_world_snapshot(
    history: pd.DataFrame,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Overwrite each team's latest Elo with the authoritative World.tsv value."""
    if history.empty:
        return history
    snap = fetch_current_snapshot(session=session)
    out = history.copy()
    for _, row in snap.iterrows():
        team = row["team"]
        mask = out["team"] == team
        if not mask.any():
            out = pd.concat(
                [out, pd.DataFrame([{"team": team, "date": row["date"], "elo": row["elo"]}])],
                ignore_index=True,
            )
            continue
        latest_idx = out.loc[mask, "date"].idxmax()
        out.loc[latest_idx, "elo"] = row["elo"]
        out.loc[latest_idx, "date"] = row["date"]
    return out.sort_values(["team", "date"]).reset_index(drop=True)


def fetch_current_snapshot(session: Optional[requests.Session] = None) -> pd.DataFrame:
    """Current rating for every team from World.tsv (single date = today)."""
    session = session or requests.Session()
    teams = load_en_teams(session=session)
    code_to_name = dict(zip(teams["code"], teams["name"]))

    text = download_text("World.tsv", session=session)
    today = pd.Timestamp.today().normalize()
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        code = parts[2].strip()
        rating = _normalize_number(parts[3])
        if code not in code_to_name or rating is None:
            continue
        rows.append(
            {
                "team": model_team_name(code_to_name[code]),
                "date": today,
                "elo": rating,
            }
        )
    return pd.DataFrame(rows)


def fetch_full_history(
    teams_df: Optional[pd.DataFrame] = None,
    max_teams: Optional[int] = None,
    delay: float = DEFAULT_DELAY,
    session: Optional[requests.Session] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Download match-by-match history for every team in en.teams.tsv."""
    session = session or requests.Session()
    teams_df = teams_df if teams_df is not None else load_en_teams(session=session)
    if max_teams is not None:
        teams_df = teams_df.head(max_teams)

    frames: list[pd.DataFrame] = []
    n = len(teams_df)
    for i, row in teams_df.iterrows():
        name = model_team_name(row["name"])
        page = row["page"]
        if verbose:
            print(f"  [{i + 1}/{n}] {name} ({page}.tsv)")
        hist = fetch_team_history(page, name, row["code"], session=session)
        if not hist.empty:
            frames.append(hist)
        if delay > 0:
            time.sleep(delay)

    if not frames:
        return pd.DataFrame(columns=["team", "date", "elo"])

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["team", "date"]).reset_index(drop=True)
    # One row per team per day (keep last match that day if duplicates).
    out = out.groupby(["team", "date"], as_index=False).tail(1)
    out = reconcile_with_world_snapshot(out, session=session)
    return out.sort_values(["team", "date"]).reset_index(drop=True)


def build_elo_csv(
    output_path: str = ELO_CSV,
    current_only: bool = False,
    max_teams: Optional[int] = None,
    delay: float = DEFAULT_DELAY,
    verbose: bool = True,
) -> pd.DataFrame:
    """Download Elo data and write elo_ratings.csv."""
    if verbose:
        mode = "current snapshot" if current_only else "full history"
        print(f"Downloading Elo ratings from {BASE_URL} ({mode})...")

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "WorldCupSimulator/0.1 (educational; contact: local)"}
    )

    if current_only:
        df = fetch_current_snapshot(session=session)
    else:
        df = fetch_full_history(
            max_teams=max_teams,
            delay=delay,
            session=session,
            verbose=verbose,
        )

    df.to_csv(output_path, index=False)
    if verbose:
        print(f"Saved {len(df):,} rows -> {output_path}")
        print(f"Teams: {df['team'].nunique()}, "
              f"dates {df['date'].min().date()} .. {df['date'].max().date()}")
    return df


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download Elo ratings from eloratings.net TSV files."
    )
    parser.add_argument(
        "--output",
        default=ELO_CSV,
        help=f"Output CSV path (default: {ELO_CSV})",
    )
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="Only download today's ratings from World.tsv (fast, no history).",
    )
    parser.add_argument(
        "--max-teams",
        type=int,
        default=None,
        help="Limit number of teams (for testing).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="Seconds between team requests when downloading full history.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    build_elo_csv(
        output_path=args.output,
        current_only=args.current_only,
        max_teams=args.max_teams,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
