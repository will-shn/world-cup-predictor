"""Internal smoke checks - exercise each module in isolation."""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from world_cup_model.config import DECAY_RATE, GROUPS_2026
from world_cup_model.data.clean import clean_in_memory
from world_cup_model.data.fetch import generate_synthetic_results
from world_cup_model.evaluation.calibrate import (
    brier_score,
    evaluate_predictions,
    reliability_diagram,
)
from world_cup_model.evaluation.market import (
    add_kelly_column,
    compare_outright_probs,
    find_edges,
    kelly_fraction,
    odds_to_prob,
)
from world_cup_model.model.dixon_coles import (
    fit_model,
    predict_lambdas,
    predict_lambdas_batch,
    predict_match_probabilities,
)
from world_cup_model.simulation.tournament import run_tournament

print("=" * 70)
print("INTERNAL SMOKE TEST")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Data pipeline
# ---------------------------------------------------------------------------
print("\n[1] Data pipeline")
raw = generate_synthetic_results(n_teams=20, n_matches=600, seed=1)
df = clean_in_memory(raw, DECAY_RATE, min_date="2000-01-01", competitive_only=False)
print(f"   matches: {len(df)}, weight sum: {df['weight'].sum():.1f}")
assert {"home_team", "away_team", "weight"} <= set(df.columns)

# ---------------------------------------------------------------------------
# 2. Dixon-Coles fit
# ---------------------------------------------------------------------------
print("\n[2] Dixon-Coles fit")
t0 = time.time()
params = fit_model(df)
print(f"   fit time: {time.time() - t0:.2f}s, converged={params.converged}")
print(f"   home_adv={params.home_adv:.3f}, rho={params.rho:.3f}")
assert params.converged
assert -0.5 < params.home_adv < 1.0
assert -0.2 <= params.rho <= 0.2

# ---------------------------------------------------------------------------
# 3. Prediction helpers
# ---------------------------------------------------------------------------
print("\n[3] Prediction helpers")
teams_present = list(params.teams)
h, a = teams_present[0], teams_present[1]
lam_h, lam_a = predict_lambdas(h, a, params)
print(f"   single lambdas {h} vs {a}: {lam_h:.2f}/{lam_a:.2f}")
res = predict_match_probabilities(h, a, params)
total = res["p_home"] + res["p_draw"] + res["p_away"]
print(f"   probs h/d/a: {res['p_home']:.3f}/{res['p_draw']:.3f}/{res['p_away']:.3f} "
      f"(sum={total:.4f})")
assert abs(total - 1.0) < 1e-6

# Batched lambdas
lam_h_b, lam_a_b = predict_lambdas_batch(
    [h, teams_present[2]], [a, teams_present[3]], params
)
print(f"   batched lambdas: {lam_h_b}, {lam_a_b}")
assert np.isclose(lam_h_b[0], lam_h, rtol=1e-6)

# Unknown team fallback (synthetic teams aren't in 2026 group config)
res2 = predict_match_probabilities("XX-Unknown1", "XX-Unknown2", params, neutral=True)
print(f"   unknown teams h/d/a: {res2['p_home']:.3f}/{res2['p_draw']:.3f}/{res2['p_away']:.3f}")
assert abs(res2["p_home"] - res2["p_away"]) < 0.05  # symmetric

# ---------------------------------------------------------------------------
# 4. Calibration metrics
# ---------------------------------------------------------------------------
print("\n[4] Calibration metrics")
y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
y_pred = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6], [0.5, 0.3, 0.2]])
metrics = evaluate_predictions(y_true, y_pred)
print(f"   brier={metrics['brier']:.4f}, log_loss={metrics['log_loss']:.4f}")
assert 0 < metrics["brier"] < 1

# Reliability diagram (will save to outputs/)
rng = np.random.default_rng(0)
p = rng.random(1000)
y = (rng.random(1000) < p).astype(int)
path = reliability_diagram(y, p, save_path=os.path.join(
    os.path.dirname(__file__), "..", "world_cup_model", "outputs", "smoke_reliability.png"
))
print(f"   reliability diagram saved to {path}")
assert os.path.exists(path)

# ---------------------------------------------------------------------------
# 5. Market module
# ---------------------------------------------------------------------------
print("\n[5] Market module")
ph, pd_, pa = odds_to_prob(2.0, 3.5, 4.0)
print(f"   odds_to_prob: h={ph:.3f}, d={pd_:.3f}, a={pa:.3f}")
assert abs(ph + pd_ + pa - 1.0) < 1e-9

kf = kelly_fraction(0.6, 2.0)
print(f"   kelly(p=0.6, odds=2.0) = {kf:.3f}")
assert kf > 0
kf_neg = kelly_fraction(0.4, 2.0)
print(f"   kelly(p=0.4, odds=2.0) = {kf_neg:.3f}")
assert kf_neg < 0

model_probs = {
    "m1": {"home_team": "A", "away_team": "B",
           "p_home": 0.6, "p_draw": 0.25, "p_away": 0.15},
}
market_probs = {
    "m1": {"home_team": "A", "away_team": "B",
           "p_home": 0.5, "p_draw": 0.27, "p_away": 0.23,
           "decimal_home": 2.0, "decimal_draw": 3.5, "decimal_away": 4.0},
}
edges = find_edges(model_probs, market_probs, threshold=0.05)
edges = add_kelly_column(edges)
print(f"   edges: {len(edges)} rows")
print(edges.to_string(index=False))
assert len(edges) >= 1

outright = compare_outright_probs(
    {"A": 0.30, "B": 0.20, "C": 0.10},
    {"A": 3.0, "B": 5.0, "C": 10.0},
)
print("\n   outright comparison:")
print(outright.to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Full tournament (small)
# ---------------------------------------------------------------------------
print("\n[6] Full tournament (100 sims)")
t0 = time.time()
sim = run_tournament(params, GROUPS_2026, n_sims=100, seed=1)
print(f"   sim time: {time.time() - t0:.2f}s")
total_win = sum(sim["win_probs"].values())
print(f"   win-prob sum across 48 teams: {total_win:.3f}")
assert abs(total_win - 1.0) < 0.001

# ---------------------------------------------------------------------------
# 7. 100k sims benchmark
# ---------------------------------------------------------------------------
print("\n[7] Full tournament benchmark (100k sims)")
t0 = time.time()
sim = run_tournament(params, GROUPS_2026, n_sims=100_000, seed=1)
print(f"   100k sim time: {time.time() - t0:.2f}s")
total_win = sum(sim["win_probs"].values())
print(f"   win-prob sum: {total_win:.4f}")
assert abs(total_win - 1.0) < 0.01

print("\nALL INTERNAL CHECKS PASSED")
