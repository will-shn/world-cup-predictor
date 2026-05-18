"""
End-to-end driver for the World Cup 2026 prediction model.

Run with:
    python -m world_cup_model.main
or, from the project root:
    python world_cup_model/main.py

CLI flags
---------
    --synthetic       use generated synthetic data (no CSV required, smoke test)
    --n-sims N        override config.N_SIMULATIONS
    --skip-backtest   do not run the historical backtest
    --backtest-only   skip the simulation
    --results PATH    override the results CSV path
    --elo PATH        path to an Elo CSV (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import pandas as pd

from world_cup_model.config import (
    DECAY_RATE,
    ELO_CSV,
    ELO_PRIOR_STRENGTH,
    GROUPS_2026,
    HOST_COUNTRIES,
    MIN_DATE,
    N_SIMULATIONS,
    OUTPUT_DIR,
    RESULTS_CSV,
)
from world_cup_model.data.clean import build_clean_dataset, clean_in_memory
from world_cup_model.data.fetch import generate_synthetic_results
from world_cup_model.evaluation.calibrate import backtest
from world_cup_model.features.ratings import (
    build_feature_matrix,
    current_team_elos,
    elo_to_prior,
    load_elo_data,
)
from world_cup_model.model.dixon_coles import (
    fit_model,
    predict_match_probabilities,
)
from world_cup_model.simulation.tournament import run_tournament


# ---------------------------------------------------------------------------
# Utility printing
# ---------------------------------------------------------------------------
def _print_top(rank_dict: dict, label: str, k: int = 12) -> None:
    print(f"\n{label}")
    for team, prob in sorted(rank_dict.items(), key=lambda x: -x[1])[:k]:
        print(f"  {team:<22s} {prob*100:6.2f}%")


def _print_match_card(matchups, params) -> None:
    print("\nSample group-stage probabilities (host venues use home advantage):")
    for h, a in matchups:
        neutral = (h not in HOST_COUNTRIES) and (a not in HOST_COUNTRIES)
        res = predict_match_probabilities(h, a, params, neutral=neutral)
        print(
            f"  {h} vs {a:<18s} "
            f"H {res['p_home']*100:5.1f}%  D {res['p_draw']*100:5.1f}%  A {res['p_away']*100:5.1f}%"
            f"   (lambdas {res['lambda_home']:.2f} / {res['lambda_away']:.2f})"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def load_data(
    use_synthetic: bool,
    results_path: str,
    elo_path: Optional[str] = None,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if use_synthetic:
        print("[1/4] Generating synthetic results (smoke test).")
        raw = generate_synthetic_results()
        df = clean_in_memory(raw, DECAY_RATE, min_date="2000-01-01",
                             competitive_only=False)
    else:
        print(f"[1/4] Loading historical results from {results_path}")
        df = build_clean_dataset(results_path, DECAY_RATE, min_date=MIN_DATE)

    elo_df = None
    if elo_path and os.path.exists(elo_path):
        print(f"      Loading Elo ratings from {elo_path}")
        elo_df = load_elo_data(elo_path)
        df = build_feature_matrix(df, elo_df)
    return df, elo_df


def _build_elo_priors(
    df: pd.DataFrame,
    elo_df: Optional[pd.DataFrame],
) -> Optional[dict]:
    """Snapshot Elo for every team in the training data, convert to priors."""
    if elo_df is None or elo_df.empty:
        return None
    teams = sorted(set(df["home_team"]).union(df["away_team"]))
    snapshot = current_team_elos(elo_df, teams)
    return elo_to_prior(snapshot)


def fit_and_simulate(
    df: pd.DataFrame,
    n_sims: int,
    elo_df: Optional[pd.DataFrame] = None,
    elo_strength: float = ELO_PRIOR_STRENGTH,
) -> dict:
    print(f"[2/4] Fitting Dixon-Coles on {len(df):,} matches "
          f"with weight sum {df['weight'].sum():.1f}.")
    elo_priors = _build_elo_priors(df, elo_df)
    if elo_priors:
        n_covered = sum(1 for t in elo_priors if t in df["home_team"].values
                        or t in df["away_team"].values)
        print(f"      Using Elo priors for {n_covered} teams "
              f"(strength={elo_strength}).")
    else:
        print("      No Elo priors available; using plain L2 ridge.")
    t0 = time.time()
    params = fit_model(df, elo_priors=elo_priors, elo_prior_strength=elo_strength)
    print(f"      Fit complete in {time.time()-t0:.1f}s. "
          f"Converged={params.converged} | home_adv={params.home_adv:.3f} | "
          f"rho={params.rho:.3f} | nll={params.fun:.1f}")

    # Optional: show top/bottom attack/defense strengths.
    top_attack = sorted(params.attack.items(), key=lambda kv: -kv[1])[:8]
    print("      Top 8 attack strengths:")
    for t, v in top_attack:
        print(f"        {t:<22s} {v:+.2f}")

    print(f"[3/4] Running {n_sims:,} Monte Carlo tournaments.")
    t0 = time.time()
    sim_out = run_tournament(params, GROUPS_2026, n_sims=n_sims)
    print(f"      Simulation done in {time.time()-t0:.1f}s.")

    _print_top(sim_out["win_probs"], "Top win probabilities:")
    _print_top(sim_out["finalist_probs"], "Top finalist probabilities:")
    _print_top(sim_out["semi_probs"], "Top semifinal probabilities:")
    _print_top(sim_out["group_advance"], "Top group-advancement probabilities:", k=16)

    # Highlight a few sample matches.
    sample_matches = [
        ("Argentina", "Brazil"),
        ("France", "Germany"),
        ("USA", "Mexico"),
        ("Spain", "England"),
    ]
    # Drop pairs containing teams not in fit (synthetic mode).
    sample_matches = [
        (h, a) for h, a in sample_matches
        if h in params.attack and a in params.attack
    ]
    if sample_matches:
        _print_match_card(sample_matches, params)

    # Persist a JSON of the win probabilities for downstream use.
    out_path = os.path.join(OUTPUT_DIR, "win_probs.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: sim_out[k] for k in ("win_probs", "finalist_probs",
                                     "semi_probs", "qf_probs", "r16_probs",
                                     "group_advance")},
            f, indent=2, sort_keys=True,
        )
    print(f"      Saved win-probability JSON to {out_path}")

    return {"params": params, "sim_out": sim_out}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="World Cup 2026 prediction model")
    p.add_argument("--synthetic", action="store_true",
                   help="Run with generated synthetic results instead of the CSV.")
    p.add_argument("--n-sims", type=int, default=N_SIMULATIONS,
                   help="Number of Monte Carlo tournaments.")
    p.add_argument("--results", default=RESULTS_CSV,
                   help="Path to historical results CSV.")
    p.add_argument("--elo", default=ELO_CSV if os.path.exists(ELO_CSV) else None,
                   help="Optional path to Elo CSV.")
    p.add_argument("--elo-strength", type=float, default=ELO_PRIOR_STRENGTH,
                   help="Strength of Elo prior pull. 0 = ignore Elo.")
    p.add_argument("--skip-backtest", action="store_true",
                   help="Skip the historical backtest step.")
    p.add_argument("--backtest-only", action="store_true",
                   help="Run only the historical backtest, no simulation.")
    p.add_argument("--cutoff", default="2022-11-01",
                   help="Backtest cutoff date (default: 2022 World Cup).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df, elo_df = load_data(args.synthetic, args.results, args.elo)

    if not args.backtest_only:
        fit_and_simulate(df, args.n_sims, elo_df=elo_df,
                         elo_strength=args.elo_strength)

    if not args.skip_backtest and not args.synthetic:
        print(f"\n[4/4] Running backtest with cutoff {args.cutoff}.")
        try:
            metrics = backtest(df, cutoff_date=args.cutoff)
            print(f"      Backtest Brier: {metrics['brier']:.4f}, "
                  f"log-loss: {metrics['log_loss']:.4f} "
                  f"(train={metrics['n_train']}, test={metrics['n_test']})")
            if "reliability_plot" in metrics:
                print(f"      Reliability diagram: {metrics['reliability_plot']}")
        except ValueError as e:
            print(f"      Backtest skipped: {e}")


if __name__ == "__main__":
    main()
