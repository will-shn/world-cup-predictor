"""Diagnose why certain teams have high/low World Cup win probabilities."""
import json
from pathlib import Path

import pandas as pd

from world_cup_model.config import DECAY_RATE, GROUPS_2026, MIN_DATE
from world_cup_model.data.clean import build_clean_dataset
from world_cup_model.features.ratings import (
    current_team_elos,
    elo_to_prior,
    load_elo_data,
)
from world_cup_model.model.dixon_coles import fit_model, predict_lambdas

FOCUS = [
    "Egypt", "Colombia", "Brazil", "Germany",
    "Spain", "Argentina", "France", "Belgium", "Morocco", "Japan",
]

def main():
    df = build_clean_dataset(
        "world_cup_model/data_files/results.csv", DECAY_RATE, min_date=MIN_DATE
    )
    elo_df = load_elo_data("world_cup_model/data_files/elo_ratings.csv")
    all_teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    elo_snap = current_team_elos(elo_df, all_teams)
    priors = elo_to_prior(elo_snap)
    params = fit_model(df, elo_priors=priors)

    print("=" * 72)
    print("1. FITTED STRENGTHS vs ELO PRIOR")
    print("=" * 72)
    rows = []
    for t in FOCUS:
        rows.append({
            "team": t,
            "attack": params.attack.get(t, 0),
            "defense": params.defense.get(t, 0),
            "net": params.attack.get(t, 0) - params.defense.get(t, 0),
            "elo": elo_snap.get(t, 1500),
            "prior_atk": priors.get(t, {}).get("attack", 0),
            "prior_def": priors.get(t, {}).get("defense", 0),
        })
    tab = pd.DataFrame(rows).sort_values("net", ascending=False)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:+.3f}" if abs(x) < 10 else f"{x:.0f}"))

    print("\n" + "=" * 72)
    print("2. RECENT MATCH RECORD (2018+, weighted)")
    print("=" * 72)
    for t in FOCUS:
        tm = df[(df["home_team"] == t) | (df["away_team"] == t)]
        gf = int(((tm["home_team"] == t) * tm["home_score"]
                  + (tm["away_team"] == t) * tm["away_score"]).sum())
        ga = int(((tm["home_team"] == t) * tm["away_score"]
                  + (tm["away_team"] == t) * tm["home_score"]).sum())
        print(f"  {t:12s}  n={len(tm):4d}  w_sum={tm['weight'].sum():6.1f}  "
              f"GF={gf:3d} GA={ga:3d} GD={gf-ga:+3d}")

    print("\n" + "=" * 72)
    print("3. OPPONENT QUALITY IN TRAINING DATA")
    print("=" * 72)
    for t in ["Egypt", "Colombia", "Brazil", "Germany"]:
        tm = df[(df["home_team"] == t) | (df["away_team"] == t)].copy()
        opp = tm.apply(
            lambda r: r["away_team"] if r["home_team"] == t else r["home_team"], axis=1
        )
        tm["opponent"] = opp
        tm["opp_elo"] = tm["opponent"].map(elo_snap).fillna(1500)
        tm["opp_net"] = tm["opponent"].map(
            lambda o: params.attack.get(o, 0) - params.defense.get(o, 0)
        )
        print(f"\n  {t}: median opponent Elo = {tm['opp_elo'].median():.0f}, "
              f"mean weighted opp Elo = {(tm['opp_elo']*tm['weight']).sum()/tm['weight'].sum():.0f}")
        weak = (tm["opp_elo"] < 1550).sum()
        strong = (tm["opp_elo"] > 1750).sum()
        print(f"       vs Elo<1550: {weak} matches  |  vs Elo>1750: {strong} matches")

    print("\n" + "=" * 72)
    print("4. TOP TOURNAMENTS BY WEIGHTED VOLUME")
    print("=" * 72)
    for t in ["Egypt", "Colombia", "Brazil", "Germany"]:
        tm = df[(df["home_team"] == t) | (df["away_team"] == t)]
        by = (
            tm.groupby("tournament")
            .agg(n=("date", "count"), w=("weight", "sum"))
            .sort_values("w", ascending=False)
            .head(6)
        )
        print(f"\n  {t}:")
        print(by.to_string())

    print("\n" + "=" * 72)
    print("5. 2026 GROUP DRAW (simulation path)")
    print("=" * 72)
    for g, teams in GROUPS_2026.items():
        if any(t in teams for t in FOCUS):
            nets = {t: params.attack.get(t, 0) - params.defense.get(t, 0) for t in teams}
            ranked = sorted(nets.items(), key=lambda x: -x[1])
            print(f"  Group {g}: {teams}")
            print(f"    strength rank: " + ", ".join(f"{t}({v:+.2f})" for t, v in ranked))

    print("\n" + "=" * 72)
    print("6. NEUTRAL LAMBDAS vs GROUP OPPONENTS")
    print("=" * 72)
    matchups = [
        ("Egypt", "Belgium"), ("Egypt", "Iran"), ("Egypt", "New Zealand"),
        ("Colombia", "Portugal"), ("Colombia", "Uzbekistan"),
        ("Brazil", "Morocco"), ("Brazil", "Scotland"),
        ("Germany", "Curacao"), ("Germany", "Ivory Coast"),
        ("Spain", "England"), ("Argentina", "Austria"),
    ]
    for h, a in matchups:
        lh, la = predict_lambdas(h, a, params, neutral=True)
        print(f"  {h:12s} vs {a:18s}  λ_h={lh:.2f}  λ_a={la:.2f}  "
              f"(expected goals)")

    wp = json.loads(Path("world_cup_model/outputs/win_probs.json").read_text())
    print("\n" + "=" * 72)
    print("7. SIMULATION OUTPUT (from win_probs.json)")
    print("=" * 72)
    for t in FOCUS:
        print(f"  {t:12s}  win={wp['win_probs'].get(t,0)*100:5.2f}%  "
              f"final={wp['finalist_probs'].get(t,0)*100:5.2f}%  "
              f"advance={wp['group_advance'].get(t,0)*100:5.2f}%")

if __name__ == "__main__":
    main()
