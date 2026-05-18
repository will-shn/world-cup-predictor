"""
Vectorized Dixon-Coles bivariate Poisson model.

The model treats home and away goals as independent Poisson random variables
with team-specific attack/defense parameters and a global home-advantage term,
plus the Dixon-Coles multiplicative correction for low-score outcomes:

    lambda_h = exp(attack[h] + defense[a] + home_adv)
    lambda_a = exp(attack[a] + defense[h])

    P(H = i, A = j) = tau(i, j; lambda_h, lambda_a, rho)
                      * Pois(i; lambda_h) * Pois(j; lambda_a)

We fit by maximum (weighted) likelihood with an L2 ridge penalty on attack and
defense for numerical stability when teams have very little data. The whole
likelihood is computed with NumPy arrays rather than iterrows() so optimization
finishes in seconds on a normal laptop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from world_cup_model.config import (
    ATTACK_BOUNDS,
    DEFENSE_BOUNDS,
    ELO_PRIOR_STRENGTH,
    HOME_ADV_INIT,
    L2_REG,
    RHO_INIT,
)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------
@dataclass
class DixonColesParams:
    teams: list[str]
    attack: dict[str, float]
    defense: dict[str, float]
    home_adv: float
    rho: float
    converged: bool
    fun: float
    n_matches: int

    def as_dict(self) -> dict:
        return {
            "teams": self.teams,
            "attack": self.attack,
            "defense": self.defense,
            "home_adv": self.home_adv,
            "rho": self.rho,
            "converged": self.converged,
            "fun": self.fun,
            "n_matches": self.n_matches,
        }


# ---------------------------------------------------------------------------
# Likelihood (fully vectorized)
# ---------------------------------------------------------------------------
def _poisson_logpmf(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Stable log Poisson PMF using gammaln. k is integer-valued."""
    lam = np.clip(lam, 1e-10, None)
    return k * np.log(lam) - lam - gammaln(k + 1.0)


def _dc_correction_vec(
    hg: np.ndarray, ag: np.ndarray, lam_h: np.ndarray, lam_a: np.ndarray, rho: float
) -> np.ndarray:
    """Dixon-Coles multiplicative low-score correction, vectorized."""
    out = np.ones_like(lam_h, dtype=float)
    mask_00 = (hg == 0) & (ag == 0)
    mask_01 = (hg == 0) & (ag == 1)
    mask_10 = (hg == 1) & (ag == 0)
    mask_11 = (hg == 1) & (ag == 1)

    out[mask_00] = 1.0 - lam_h[mask_00] * lam_a[mask_00] * rho
    out[mask_01] = 1.0 + lam_h[mask_01] * rho
    out[mask_10] = 1.0 + lam_a[mask_10] * rho
    out[mask_11] = 1.0 - rho
    return out


def _unpack_params(params: np.ndarray, n_teams: int) -> tuple:
    attack = params[:n_teams]
    defense = params[n_teams : 2 * n_teams]
    home_adv = params[2 * n_teams]
    rho = params[2 * n_teams + 1]
    return attack, defense, home_adv, rho


def negative_log_likelihood(
    params: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    l2_reg: float = L2_REG,
) -> float:
    """Vectorized negative log-likelihood with L2 regularization."""
    attack, defense, home_adv, rho = _unpack_params(params, n_teams)

    lam_h = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
    lam_a = np.exp(attack[away_idx] + defense[home_idx])

    correction = _dc_correction_vec(home_goals, away_goals, lam_h, lam_a, rho)
    if np.any(correction <= 0):
        # Hand the optimizer a large finite value rather than NaN.
        return 1e10

    log_lik_per_match = (
        _poisson_logpmf(home_goals, lam_h)
        + _poisson_logpmf(away_goals, lam_a)
        + np.log(correction)
    )

    nll = -float(np.sum(weights * log_lik_per_match))
    nll += l2_reg * float(np.sum(attack ** 2) + np.sum(defense ** 2))
    return nll


def _gradient_check_safe(value: float) -> float:
    if not np.isfinite(value):
        return 1e10
    return value


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit_model(
    df: pd.DataFrame,
    l2_reg: float = L2_REG,
    rho_init: float = RHO_INIT,
    home_adv_init: float = HOME_ADV_INIT,
    rho_bounds: tuple = (-0.2, 0.2),
    verbose: bool = False,
    elo_priors: Optional[dict] = None,
    elo_prior_strength: float = ELO_PRIOR_STRENGTH,
) -> DixonColesParams:
    """Fit Dixon-Coles by maximum penalized weighted likelihood.

    Expects a DataFrame with columns:
        home_team, away_team, home_score, away_score, weight, neutral (optional)

    Elo priors (team -> {"attack": ..., "defense": ...}) are pulled in via a
    quadratic penalty: stronger `elo_prior_strength` keeps sparse teams near
    their Elo-implied strength rather than overfitting a handful of results.
    If no priors are supplied, the optimizer falls back to L2 ridge toward 0.
    """
    needed = {"home_team", "away_team", "home_score", "away_score", "weight"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"fit_model missing columns: {missing}")

    teams = sorted(set(df["home_team"]).union(df["away_team"]))
    team_to_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    # Neutral-venue matches do not use home advantage.
    neutral = df["neutral"].to_numpy() if "neutral" in df.columns else np.zeros(len(df), bool)

    home_idx = df["home_team"].map(team_to_idx).to_numpy()
    away_idx = df["away_team"].map(team_to_idx).to_numpy()
    home_goals = df["home_score"].astype(int).to_numpy()
    away_goals = df["away_score"].astype(int).to_numpy()
    weights = df["weight"].astype(float).to_numpy()
    home_adv_active = (~neutral).astype(float)

    # Build Elo prior arrays in the same ordering as the parameter vector.
    prior_attack = np.zeros(n_teams)
    prior_defense = np.zeros(n_teams)
    prior_active = np.zeros(n_teams)
    if elo_priors and elo_prior_strength > 0:
        for i, team in enumerate(teams):
            if team in elo_priors:
                prior_attack[i] = float(elo_priors[team].get("attack", 0.0))
                prior_defense[i] = float(elo_priors[team].get("defense", 0.0))
                prior_active[i] = 1.0
    use_elo_prior = bool(elo_priors) and elo_prior_strength > 0 and prior_active.any()

    def nll(params: np.ndarray) -> float:
        attack, defense, home_adv, rho = _unpack_params(params, n_teams)
        lam_h = np.exp(attack[home_idx] + defense[away_idx] + home_adv * home_adv_active)
        lam_a = np.exp(attack[away_idx] + defense[home_idx])

        correction = _dc_correction_vec(home_goals, away_goals, lam_h, lam_a, rho)
        if np.any(correction <= 0):
            return 1e10

        log_lik_per_match = (
            _poisson_logpmf(home_goals, lam_h)
            + _poisson_logpmf(away_goals, lam_a)
            + np.log(correction)
        )
        value = -float(np.sum(weights * log_lik_per_match))

        # Regularization: prefer Elo priors when available, fall back to L2.
        if use_elo_prior:
            atk_dev = (attack - prior_attack) * prior_active
            def_dev = (defense - prior_defense) * prior_active
            value += elo_prior_strength * float(
                np.sum(atk_dev ** 2) + np.sum(def_dev ** 2)
            )
            # Light L2 on teams with no Elo prior so they don't run away.
            no_prior = 1.0 - prior_active
            value += l2_reg * float(
                np.sum((attack * no_prior) ** 2)
                + np.sum((defense * no_prior) ** 2)
            )
        else:
            value += l2_reg * float(np.sum(attack ** 2) + np.sum(defense ** 2))
        return _gradient_check_safe(value)

    # Start the team parameters at their priors when available - speeds up fits.
    x0 = np.zeros(2 * n_teams + 2)
    if use_elo_prior:
        x0[:n_teams] = prior_attack
        x0[n_teams : 2 * n_teams] = prior_defense
    x0[2 * n_teams] = home_adv_init
    x0[2 * n_teams + 1] = rho_init

    bounds = (
        [ATTACK_BOUNDS] * n_teams
        + [DEFENSE_BOUNDS] * n_teams
        + [(-0.5, 1.0)]            # home advantage
        + [rho_bounds]
    )

    result = minimize(
        nll,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-9, "gtol": 1e-7, "disp": verbose},
    )

    attack_vec, defense_vec, home_adv, rho = _unpack_params(result.x, n_teams)
    return DixonColesParams(
        teams=teams,
        attack={t: float(attack_vec[i]) for i, t in enumerate(teams)},
        defense={t: float(defense_vec[i]) for i, t in enumerate(teams)},
        home_adv=float(home_adv),
        rho=float(rho),
        converged=bool(result.success),
        fun=float(result.fun),
        n_matches=len(df),
    )


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def _lookup(params: DixonColesParams | dict, side: str, team: str) -> float:
    table = params.__dict__[side] if isinstance(params, DixonColesParams) else params[side]
    if team not in table:
        # Unknown team falls back to league-average (parameter = 0).
        return 0.0
    return float(table[team])


def predict_lambdas(
    home_team: str,
    away_team: str,
    params: DixonColesParams | dict,
    neutral: bool = False,
) -> tuple[float, float]:
    """Predict (lambda_home, lambda_away) for a single matchup."""
    home_adv = 0.0 if neutral else (
        params.home_adv if isinstance(params, DixonColesParams) else params["home_adv"]
    )
    lam_h = float(np.exp(
        _lookup(params, "attack", home_team)
        + _lookup(params, "defense", away_team)
        + home_adv
    ))
    lam_a = float(np.exp(
        _lookup(params, "attack", away_team)
        + _lookup(params, "defense", home_team)
    ))
    return lam_h, lam_a


def predict_match_probabilities(
    home_team: str,
    away_team: str,
    params: DixonColesParams | dict,
    neutral: bool = False,
    max_goals: int = 10,
) -> dict:
    """Return P(home win), P(draw), P(away win), and the full scoreline grid."""
    lam_h, lam_a = predict_lambdas(home_team, away_team, params, neutral)
    rho = params.rho if isinstance(params, DixonColesParams) else params["rho"]

    grid_h = np.arange(max_goals + 1)
    grid_a = np.arange(max_goals + 1)
    H, A = np.meshgrid(grid_h, grid_a, indexing="ij")

    pmf_h = np.exp(_poisson_logpmf(H.astype(float), np.full_like(H, lam_h, dtype=float)))
    pmf_a = np.exp(_poisson_logpmf(A.astype(float), np.full_like(A, lam_a, dtype=float)))
    grid = pmf_h * pmf_a

    correction = _dc_correction_vec(
        H.ravel(),
        A.ravel(),
        np.full(H.size, lam_h),
        np.full(H.size, lam_a),
        float(rho),
    ).reshape(H.shape)
    grid = grid * correction

    # Normalize against the truncation tail mass for cleaner probabilities.
    grid = grid / grid.sum()

    p_home = float(np.tril(grid, k=-1).sum())   # home goals > away goals
    p_away = float(np.triu(grid, k=1).sum())
    p_draw = float(np.trace(grid))

    return {
        "lambda_home": lam_h,
        "lambda_away": lam_a,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "scoreline_grid": grid,
    }


def predict_lambdas_batch(
    home_teams: list[str],
    away_teams: list[str],
    params: DixonColesParams,
    neutral: np.ndarray | bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized lambda lookup for many matchups at once."""
    attack = np.array([params.attack.get(t, 0.0) for t in params.teams])
    defense = np.array([params.defense.get(t, 0.0) for t in params.teams])
    idx = {t: i for i, t in enumerate(params.teams)}

    def to_idx(team: str) -> int:
        return idx.get(team, -1)

    h_idx = np.array([to_idx(t) for t in home_teams])
    a_idx = np.array([to_idx(t) for t in away_teams])

    h_attack = np.where(h_idx >= 0, attack[np.maximum(h_idx, 0)], 0.0)
    h_defense = np.where(h_idx >= 0, defense[np.maximum(h_idx, 0)], 0.0)
    a_attack = np.where(a_idx >= 0, attack[np.maximum(a_idx, 0)], 0.0)
    a_defense = np.where(a_idx >= 0, defense[np.maximum(a_idx, 0)], 0.0)

    if np.isscalar(neutral):
        neutral_arr = np.full(len(home_teams), bool(neutral))
    else:
        neutral_arr = np.asarray(neutral, dtype=bool)
    ha = np.where(neutral_arr, 0.0, params.home_adv)

    lam_h = np.exp(h_attack + a_defense + ha)
    lam_a = np.exp(a_attack + h_defense)
    return lam_h, lam_a
