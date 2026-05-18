"""
Vectorized Monte Carlo simulator for the 2026 FIFA World Cup.

Tournament structure
--------------------
    * 48 teams, 12 groups of 4 (A..L)
    * 6 matches per group, round-robin
    * Top 2 from each group advance plus the best 8 of the 12 third-placed
      teams -> 32-team knockout bracket
    * Single-elimination knockouts: R32 -> R16 -> QF -> SF -> F
    * Knockout draws use extra time (scaled lambdas) then penalty shootouts

Performance strategy
--------------------
Rather than looping `n_sims` times in Python, we generate every match in every
simulated tournament as one giant np.random.poisson draw. Standings, knockouts,
and bracket bookkeeping all run on (n_sims, ...) arrays.

This gets 100,000 full tournaments to finish in well under a minute on a
modern laptop.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from world_cup_model.config import (
    EXTRA_TIME_LAMBDA_SCALE,
    GROUPS_2026,
    HOST_COUNTRIES,
    PENALTY_DEFAULT_WIN_RATE,
    PENALTY_WIN_RATE,
    ROUND_OF_32,
)
from world_cup_model.model.dixon_coles import (
    DixonColesParams,
    predict_lambdas_batch,
)


# ---------------------------------------------------------------------------
# Group stage
# ---------------------------------------------------------------------------
def _group_match_schedule(groups: dict[str, list[str]]) -> pd.DataFrame:
    """Return DataFrame of all group-stage matches (6 per group * 12 groups)."""
    rows = []
    for g, teams in groups.items():
        for h, a in combinations(teams, 2):
            host = h in HOST_COUNTRIES
            rows.append(
                {
                    "group": g,
                    "home_team": h,
                    "away_team": a,
                    # Group-stage matches in host countries are not neutral for
                    # the host. Everything else is effectively neutral.
                    "neutral": not host,
                }
            )
    return pd.DataFrame(rows)


def _simulate_group_stage(
    groups: dict[str, list[str]],
    params: DixonColesParams,
    n_sims: int,
    rng: np.random.Generator,
) -> dict:
    """Run all group matches for all n_sims tournaments at once.

    Returns a dict of arrays keyed by group letter, each of shape (n_sims, 4)
    holding points/gd/gf for the four teams in draw order.
    """
    schedule = _group_match_schedule(groups)
    lam_h, lam_a = predict_lambdas_batch(
        schedule["home_team"].tolist(),
        schedule["away_team"].tolist(),
        params,
        neutral=schedule["neutral"].to_numpy(),
    )

    n_matches = len(schedule)
    home_goals = rng.poisson(lam=lam_h[None, :], size=(n_sims, n_matches))
    away_goals = rng.poisson(lam=lam_a[None, :], size=(n_sims, n_matches))

    by_group: dict[str, dict[str, np.ndarray]] = {}
    for g, teams in groups.items():
        n_teams = len(teams)
        points = np.zeros((n_sims, n_teams), dtype=np.int32)
        gf = np.zeros((n_sims, n_teams), dtype=np.int32)
        ga = np.zeros((n_sims, n_teams), dtype=np.int32)

        team_to_pos = {t: i for i, t in enumerate(teams)}
        mask = schedule["group"].to_numpy() == g
        sub = schedule[mask]
        hg = home_goals[:, mask]
        ag = away_goals[:, mask]

        for col, (_, row) in enumerate(sub.iterrows()):
            hpos = team_to_pos[row["home_team"]]
            apos = team_to_pos[row["away_team"]]
            hh = hg[:, col]
            aa = ag[:, col]

            home_win = hh > aa
            away_win = aa > hh
            draw = hh == aa

            points[:, hpos] += np.where(home_win, 3, np.where(draw, 1, 0))
            points[:, apos] += np.where(away_win, 3, np.where(draw, 1, 0))
            gf[:, hpos] += hh
            gf[:, apos] += aa
            ga[:, hpos] += aa
            ga[:, apos] += hh

        gd = gf - ga
        by_group[g] = {
            "teams": np.array(teams),
            "points": points,
            "gd": gd,
            "gf": gf,
            "ga": ga,
        }
    return by_group


def _rank_within_group(group_stats: dict, rng: np.random.Generator) -> np.ndarray:
    """Rank the four teams in a group for every simulation.

    Tiebreakers: points -> goal difference -> goals for -> random.
    Returns an (n_sims, 4) array of team indices sorted best-to-worst.
    """
    pts = group_stats["points"]
    gd = group_stats["gd"]
    gf = group_stats["gf"]
    n_sims, n_teams = pts.shape

    jitter = rng.random(size=(n_sims, n_teams)) * 1e-6
    composite = (
        pts.astype(np.float64) * 1e9
        + gd.astype(np.float64) * 1e5
        + gf.astype(np.float64) * 1e1
        + jitter
    )
    order = np.argsort(-composite, axis=1)
    return order


def _rank_third_place_teams(
    group_results: dict,
    ranks: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the best 8 of the 12 third-placed teams in each simulation.

    Returns
    -------
    third_team_names : (n_sims, 8) array of team name strings (best->worst)
    third_groups     : (n_sims, 8) array of group letters in the same order
    """
    groups_list = list(group_results.keys())
    n_groups = len(groups_list)
    n_sims = next(iter(ranks.values())).shape[0]

    third_pts = np.zeros((n_sims, n_groups), dtype=np.int64)
    third_gd = np.zeros((n_sims, n_groups), dtype=np.int64)
    third_gf = np.zeros((n_sims, n_groups), dtype=np.int64)
    third_team_idx = np.zeros((n_sims, n_groups), dtype=np.int32)

    for gi, g in enumerate(groups_list):
        third_pos = ranks[g][:, 2]                           # third-placed slot
        sims = np.arange(n_sims)
        third_pts[:, gi] = group_results[g]["points"][sims, third_pos]
        third_gd[:, gi] = group_results[g]["gd"][sims, third_pos]
        third_gf[:, gi] = group_results[g]["gf"][sims, third_pos]
        third_team_idx[:, gi] = third_pos

    jitter = rng.random(size=(n_sims, n_groups)) * 1e-6
    composite = (
        third_pts.astype(np.float64) * 1e9
        + third_gd.astype(np.float64) * 1e5
        + third_gf.astype(np.float64) * 1e1
        + jitter
    )
    order = np.argsort(-composite, axis=1)                   # group indices best->worst

    third_names = np.empty((n_sims, 8), dtype=object)
    third_groups_top = np.empty((n_sims, 8), dtype=object)
    for slot in range(8):
        gi_col = order[:, slot]                              # which group is slot-th best
        for gi, g in enumerate(groups_list):
            sel = gi_col == gi
            if not np.any(sel):
                continue
            team_pos = third_team_idx[sel, gi]
            names = group_results[g]["teams"][team_pos]
            third_names[sel, slot] = names
            third_groups_top[sel, slot] = g
    return third_names, third_groups_top


# ---------------------------------------------------------------------------
# Knockout stage
# ---------------------------------------------------------------------------
def _simulate_knockout_match(
    home_team: np.ndarray,
    away_team: np.ndarray,
    params: DixonColesParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate one knockout round across all sims. Returns winner names array."""
    home_list = home_team.tolist()
    away_list = away_team.tolist()
    lam_h, lam_a = predict_lambdas_batch(home_list, away_list, params, neutral=True)

    hg = rng.poisson(lam=lam_h)
    ag = rng.poisson(lam=lam_a)

    winner = np.where(hg > ag, home_team, np.where(ag > hg, away_team, ""))
    drawn = winner == ""

    if np.any(drawn):
        # Extra time: rescale lambdas (30 minutes ~= 1/3 of 90).
        lh_et = lam_h[drawn] * EXTRA_TIME_LAMBDA_SCALE
        la_et = lam_a[drawn] * EXTRA_TIME_LAMBDA_SCALE
        hg_et = rng.poisson(lam=lh_et)
        ag_et = rng.poisson(lam=la_et)

        et_home = hg_et > ag_et
        et_away = ag_et > hg_et
        et_draw = ~et_home & ~et_away

        et_winner = np.where(
            et_home, home_team[drawn], np.where(et_away, away_team[drawn], "")
        )

        # Penalty shootouts for ties after ET.
        if np.any(et_draw):
            home_rates = np.array(
                [
                    PENALTY_WIN_RATE.get(t, PENALTY_DEFAULT_WIN_RATE)
                    for t in home_team[drawn][et_draw]
                ]
            )
            away_rates = np.array(
                [
                    PENALTY_WIN_RATE.get(t, PENALTY_DEFAULT_WIN_RATE)
                    for t in away_team[drawn][et_draw]
                ]
            )
            # Normalize the two competing rates to a single probability of home win.
            denom = home_rates + (1.0 - away_rates)
            denom = np.where(denom <= 0, 1.0, denom)
            p_home_pens = home_rates / denom
            roll = rng.random(size=et_draw.sum())
            et_winner_pens = np.where(
                roll < p_home_pens,
                home_team[drawn][et_draw],
                away_team[drawn][et_draw],
            )
            et_winner[et_draw] = et_winner_pens

        winner[drawn] = et_winner
    return winner


def _resolve_bracket_slot(
    slot: tuple,
    sims_idx: np.ndarray,
    group_results: dict,
    ranks: dict[str, np.ndarray],
    third_names: np.ndarray,
) -> np.ndarray:
    """Map a bracket slot spec to an array of team names per simulation."""
    kind, key = slot
    if kind == "group_1st":
        positions = ranks[key][:, 0]
        return group_results[key]["teams"][positions]
    if kind == "group_2nd":
        positions = ranks[key][:, 1]
        return group_results[key]["teams"][positions]
    if kind == "group_3rd":
        rank = int(key) - 1            # 1..8 -> 0..7
        return third_names[:, rank].astype(str)
    raise ValueError(f"Unknown bracket slot kind: {kind!r}")


def _simulate_knockouts(
    group_results: dict,
    ranks: dict[str, np.ndarray],
    third_names: np.ndarray,
    params: DixonColesParams,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Run R32 -> Final and return reached-stage arrays for every team-sim cell."""
    n_sims = next(iter(ranks.values())).shape[0]
    sims_idx = np.arange(n_sims)

    # Resolve the 32 R32 entrants.
    r32_home = np.empty((n_sims, len(ROUND_OF_32)), dtype=object)
    r32_away = np.empty((n_sims, len(ROUND_OF_32)), dtype=object)
    for ti, (slot_h, slot_a) in enumerate(ROUND_OF_32):
        r32_home[:, ti] = _resolve_bracket_slot(slot_h, sims_idx, group_results, ranks, third_names)
        r32_away[:, ti] = _resolve_bracket_slot(slot_a, sims_idx, group_results, ranks, third_names)

    stage_reached = {
        "r32_home": r32_home,
        "r32_away": r32_away,
    }

    def _round(home: np.ndarray, away: np.ndarray) -> np.ndarray:
        n_ties = home.shape[1]
        flat_home = home.reshape(-1)
        flat_away = away.reshape(-1)
        winners = _simulate_knockout_match(flat_home, flat_away, params, rng)
        return winners.reshape(n_sims, n_ties)

    # R32 -> R16 (16 ties).
    r32_winners = _round(r32_home, r32_away)                 # (n_sims, 16)
    r16_home = r32_winners[:, 0::2]
    r16_away = r32_winners[:, 1::2]

    r16_winners = _round(r16_home, r16_away)                 # (n_sims, 8)
    qf_home = r16_winners[:, 0::2]
    qf_away = r16_winners[:, 1::2]

    qf_winners = _round(qf_home, qf_away)                    # (n_sims, 4)
    sf_home = qf_winners[:, 0::2]
    sf_away = qf_winners[:, 1::2]

    sf_winners = _round(sf_home, sf_away)                    # (n_sims, 2)
    f_winner = _round(sf_winners[:, :1], sf_winners[:, 1:])  # (n_sims, 1)

    stage_reached.update(
        {
            "r16": r32_winners,
            "qf": r16_winners,
            "sf": qf_winners,
            "f": sf_winners,
            "champion": f_winner.ravel(),
        }
    )
    return stage_reached


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_tournament(
    params: DixonColesParams,
    groups: Optional[dict[str, list[str]]] = None,
    n_sims: int = 100_000,
    seed: int = 42,
    return_full: bool = False,
) -> dict:
    """Run `n_sims` World Cup tournaments in parallel and aggregate outcomes.

    Returns a dict with:
        win_probs    : team -> P(team wins the cup)
        finalist_probs : team -> P(reaches final)
        semi_probs   : team -> P(reaches semis)
        qf_probs     : team -> P(reaches QF)
        r16_probs    : team -> P(reaches R16)
        group_advance: team -> P(advances from group)
        n_sims       : int
        raw          : optional dict of underlying arrays (when return_full=True)
    """
    if groups is None:
        groups = GROUPS_2026
    rng = np.random.default_rng(seed)

    group_results = _simulate_group_stage(groups, params, n_sims, rng)
    ranks = {g: _rank_within_group(group_results[g], rng) for g in group_results}
    third_names, _ = _rank_third_place_teams(group_results, ranks, rng)
    stages = _simulate_knockouts(group_results, ranks, third_names, params, rng)

    all_teams = sorted({t for ts in groups.values() for t in ts})

    def freq(arr: np.ndarray) -> dict[str, float]:
        flat = arr.ravel()
        flat = flat[flat != None]                            # noqa: E711
        unique, counts = np.unique(flat.astype(str), return_counts=True)
        total = n_sims
        return {t: float(counts[np.where(unique == t)[0][0]] / total)
                if t in unique else 0.0
                for t in all_teams}

    # Group advancement (top 2 plus 8 best thirds).
    advance_counts = {t: 0 for t in all_teams}
    for g in groups:
        positions = ranks[g]
        teams_arr = group_results[g]["teams"]
        top2_teams = teams_arr[positions[:, :2]]            # (n_sims, 2)
        for col in range(2):
            unique, counts = np.unique(top2_teams[:, col], return_counts=True)
            for t, c in zip(unique, counts):
                advance_counts[str(t)] = advance_counts.get(str(t), 0) + int(c)

    # 8 best thirds contribute too.
    unique, counts = np.unique(third_names.astype(str), return_counts=True)
    for t, c in zip(unique, counts):
        if t == "None" or t == "":
            continue
        advance_counts[t] = advance_counts.get(t, 0) + int(c)

    win_probs = freq(stages["champion"])
    finalist_probs = freq(stages["f"])
    semi_probs = freq(stages["sf"])
    qf_probs = freq(stages["qf"])
    r16_probs = freq(stages["r16"])

    out = {
        "win_probs": win_probs,
        "finalist_probs": finalist_probs,
        "semi_probs": semi_probs,
        "qf_probs": qf_probs,
        "r16_probs": r16_probs,
        "group_advance": {t: advance_counts.get(t, 0) / n_sims for t in all_teams},
        "n_sims": n_sims,
    }
    if return_full:
        out["raw"] = {
            "group_results": group_results,
            "ranks": ranks,
            "third_names": third_names,
            "stages": stages,
        }
    return out


def simulate_match(lambda_h: float, lambda_a: float, rng: Optional[np.random.Generator] = None) -> tuple:
    """One-off scalar match sim (kept for parity with the brief)."""
    rng = rng or np.random.default_rng()
    return int(rng.poisson(lambda_h)), int(rng.poisson(lambda_a))


def simulate_penalty_shootout(home_team: str, away_team: str, rng: Optional[np.random.Generator] = None) -> str:
    """Scalar penalty-shootout helper used by smoke tests."""
    rng = rng or np.random.default_rng()
    home_rate = PENALTY_WIN_RATE.get(home_team, PENALTY_DEFAULT_WIN_RATE)
    away_rate = PENALTY_WIN_RATE.get(away_team, PENALTY_DEFAULT_WIN_RATE)
    denom = home_rate + (1.0 - away_rate)
    p_home = home_rate / denom if denom > 0 else 0.5
    return home_team if rng.random() < p_home else away_team
