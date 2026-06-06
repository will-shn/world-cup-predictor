"""
World Cup 2026 Prediction Dashboard

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Share online (free): push to GitHub, deploy at https://share.streamlit.io
See README.md section "Web dashboard".
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from world_cup_model.config import (
    ELO_CSV,
    ELO_PRIOR_STRENGTH,
    GROUPS_2026,
    HOST_COUNTRIES,
    KELLY_FRACTION_DIVISOR,
    N_SIMULATIONS,
    OUTPUT_DIR,
    RESULTS_CSV,
    SAMPLE_OUTRIGHT_ODDS,
)
from world_cup_model.data.clean import build_clean_dataset
from world_cup_model.evaluation.market import (
    has_odds_api_key,
    run_outright_benchmark,
)
from world_cup_model.features.ratings import (
    build_feature_matrix,
    current_team_elos,
    elo_to_prior,
    load_elo_data,
)
from world_cup_model.model.dixon_coles import fit_model, predict_match_probabilities
from world_cup_model.simulation.tournament import run_tournament

ROOT = Path(__file__).resolve().parent
WIN_PROBS_PATH = ROOT / "world_cup_model" / "outputs" / "win_probs.json"


# ---------------------------------------------------------------------------
# Cached model run
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_saved_results() -> dict | None:
    if WIN_PROBS_PATH.exists():
        return json.loads(WIN_PROBS_PATH.read_text(encoding="utf-8"))
    return None


@st.cache_data(show_spinner="Loading match data and Elo ratings…")
def load_training_data(results_path: str, elo_path: str | None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    df = build_clean_dataset(results_path, decay_rate=0.002, min_date="2018-01-01")
    elo_df = None
    if elo_path and os.path.exists(elo_path):
        elo_df = load_elo_data(elo_path)
        df = build_feature_matrix(df, elo_df)
    return df, elo_df


@st.cache_data(show_spinner="Fitting Dixon-Coles model…")
def fit_ratings(
    elo_strength: float,
    results_path: str,
    elo_path: str | None,
):
    """Fit only (cached) — used for match predictor without re-simulating."""
    df, elo_df = load_training_data(results_path, elo_path)
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    elo_priors = elo_to_prior(current_team_elos(elo_df, teams)) if elo_df is not None else None
    return fit_model(df, elo_priors=elo_priors, elo_prior_strength=elo_strength)


@st.cache_data(show_spinner="Fitting Dixon-Coles model and simulating tournaments…")
def run_pipeline(
    n_sims: int,
    elo_strength: float,
    results_path: str,
    elo_path: str | None,
) -> dict:
    params = fit_ratings(elo_strength, results_path, elo_path)
    df, _ = load_training_data(results_path, elo_path)
    sim_out = run_tournament(params, GROUPS_2026, n_sims=n_sims)
    return {
        "params": params,
        "sim_out": sim_out,
        "n_matches": len(df),
        "weight_sum": float(df["weight"].sum()),
        "converged": params.converged,
        "home_adv": params.home_adv,
        "rho": params.rho,
    }


def sim_to_dataframe(sim_out: dict, metric: str = "win_probs") -> pd.DataFrame:
    data = sim_out[metric]
    df = pd.DataFrame({"team": list(data.keys()), "probability": list(data.values())})
    return df.sort_values("probability", ascending=False).reset_index(drop=True)


def prob_table_to_metrics(df: pd.DataFrame, label: str) -> None:
    df = df.copy()
    df["probability"] = (df["probability"] * 100).round(2)
    df.columns = ["Team", f"{label} (%)"]
    st.dataframe(df, width="stretch", hide_index=True)


@st.cache_data(show_spinner="Comparing model to market odds…", ttl=300)
def load_market_benchmark(
    win_probs: dict[str, float],
    use_live: bool,
    odds_path: str,
) -> dict:
    return run_outright_benchmark(
        win_probs,
        odds_path=odds_path,
        use_live=use_live,
    )


def model_vs_market_chart_data(comparison: pd.DataFrame) -> pd.DataFrame:
    """Wide-format data for grouped model vs market win-probability bars."""
    chart_df = comparison.set_index("team")[["model_prob", "market_prob"]].copy()
    chart_df.columns = ["Model", "Market"]
    return chart_df * 100


def format_benchmark_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("model_prob", "market_prob", "edge", "kelly", "fractional_kelly"):
        if col in out.columns:
            out[col] = (out[col] * 100).round(2)
    out = out.rename(
        columns={
            "team": "Team",
            "model_prob": "Model (%)",
            "market_prob": "Market (%)",
            "edge": "Edge (pp)",
            "decimal_odds": "Decimal odds",
            "kelly": "Kelly (%)",
            "fractional_kelly": f"Frac. Kelly (÷{KELLY_FRACTION_DIVISOR:.0f}) (%)",
            "flagged": "Flagged",
        }
    )
    display_cols = [
        "Team", "Model (%)", "Market (%)", "Edge (pp)",
        "Decimal odds", "Kelly (%)", f"Frac. Kelly (÷{KELLY_FRACTION_DIVISOR:.0f}) (%)",
        "Flagged",
    ]
    return out[[c for c in display_cols if c in out.columns]]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="World Cup 2026 Simulator",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("World Cup 2026 Prediction Model")
st.caption(
    "Dixon-Coles Poisson ratings · tournament-weighted history · Elo priors · "
    "Monte Carlo bracket simulation"
)

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    data_ok = os.path.exists(RESULTS_CSV)
    elo_ok = os.path.exists(ELO_CSV)
    st.markdown(
        f"**Data:** {'✅' if data_ok else '❌'} results.csv  \n"
        f"**Elo:** {'✅' if elo_ok else '❌'} elo_ratings.csv"
    )
    if not data_ok:
        st.warning(
            "Place `results.csv` in `world_cup_model/data_files/` to run the model. "
            "You can still view saved results below."
        )

    n_sims = st.slider("Monte Carlo simulations", 1_000, 100_000, min(N_SIMULATIONS, 20_000), 1_000)
    elo_strength = st.slider("Elo prior strength", 0.0, 50.0, float(ELO_PRIOR_STRENGTH), 5.0)
    use_saved = st.checkbox("Use last saved results (instant)", value=True)
    run_clicked = st.button("Run model", type="primary", disabled=not data_ok)

    st.divider()
    st.markdown(
        "**Method (short)**  \n"
        "1. Fit attack/defense from weighted international results  \n"
        "2. Pull ratings toward Elo (eloratings.net)  \n"
        "3. Simulate the 2026 draw 100k× for win odds"
    )

# --- Load results ---
sim_out: dict | None = None
meta: dict = {}

if run_clicked and data_ok:
    t0 = time.time()
    with st.spinner(f"Running {n_sims:,} tournament simulations…"):
        out = run_pipeline(n_sims, elo_strength, RESULTS_CSV, ELO_CSV if elo_ok else None)
    params = out["params"]
    sim_out = out["sim_out"]
    meta = {
        "elapsed": time.time() - t0,
        "converged": out["converged"],
        "home_adv": out["home_adv"],
        "rho": out["rho"],
        "n_matches": out["n_matches"],
    }
    # Persist for next visit
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(WIN_PROBS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {k: sim_out[k] for k in (
                "win_probs", "finalist_probs", "semi_probs",
                "qf_probs", "r16_probs", "group_advance",
            )},
            f,
            indent=2,
        )
    st.success(f"Done in {meta['elapsed']:.1f}s — results saved.")

elif use_saved:
    saved = load_saved_results()
    if saved:
        sim_out = saved
        meta["source"] = "saved"
    elif not data_ok:
        st.error("No saved results found. Add data files and click **Run model**.")
        st.stop()

if sim_out is None:
    if data_ok:
        st.info("Click **Run model** in the sidebar, or enable **Use last saved results**.")
    st.stop()

# --- Top metrics ---
top_win = sim_to_dataframe(sim_out, "win_probs").head(1)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Favorite to win", top_win.iloc[0]["team"], f"{top_win.iloc[0]['probability']*100:.1f}%")
c2.metric(
    "Top 4 share of title",
    f"{sim_to_dataframe(sim_out, 'win_probs').head(4)['probability'].sum()*100:.1f}%",
)
if meta.get("converged") is not None:
    c3.metric("Model converged", "Yes" if meta["converged"] else "No")
    c4.metric("Home advantage", f"{meta.get('home_adv', 0):.3f}")
else:
    c3.metric("Source", meta.get("source", "live run"))

# --- Tabs ---
tab_outlook, tab_market, tab_match, tab_groups, tab_about = st.tabs(
    ["Tournament outlook", "Market benchmark", "Match predictor", "Group draw", "About"]
)

with tab_outlook:
    col_chart, col_table = st.columns([3, 2])
    win_df = sim_to_dataframe(sim_out, "win_probs")

    with col_chart:
        st.subheader("Win the tournament")
        chart_df = win_df.head(16).set_index("team")["probability"]
        st.bar_chart(chart_df * 100, height=420)

    with col_table:
        st.subheader("Top 16")
        prob_table_to_metrics(win_df.head(16), "Win")

    st.divider()
    m1, m2, m3 = st.columns(3)
    with m1:
        st.subheader("Reach the final")
        prob_table_to_metrics(sim_to_dataframe(sim_out, "finalist_probs").head(10), "Final")
    with m2:
        st.subheader("Reach the semis")
        prob_table_to_metrics(sim_to_dataframe(sim_out, "semi_probs").head(10), "Semis")
    with m3:
        st.subheader("Advance from group")
        prob_table_to_metrics(sim_to_dataframe(sim_out, "group_advance").head(10), "Advance")

with tab_market:
    st.subheader("Model vs bookmaker outrights")
    st.caption(
        "Compares tournament win probabilities from the Monte Carlo sim to "
        "vig-removed implied probabilities from outright decimal odds. "
        "Kelly sizing uses the model probability against the offered price."
    )

    api_available = has_odds_api_key()
    odds_source = st.radio(
        "Odds source",
        options=(
            ["Live (The Odds API)"] if api_available else []
        ) + ["Sample file (demo)"],
        horizontal=True,
    )
    use_live = odds_source.startswith("Live")
    if not api_available:
        st.info(
            "Set `ODDS_API_KEY` in your environment for live bookmaker odds. "
            "Using bundled sample odds for now."
        )

    try:
        benchmark = load_market_benchmark(
            sim_out["win_probs"],
            use_live=use_live,
            odds_path=SAMPLE_OUTRIGHT_ODDS,
        )
    except Exception as exc:
        st.error(f"Could not load market benchmark: {exc}")
        benchmark = None

    if benchmark:
        metrics = benchmark["metrics"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Matched teams", f"{metrics['n_matched']}/{metrics['n_model_teams']}")
        m2.metric("MAE", f"{metrics['mae']:.3f}" if metrics["n_matched"] else "—")
        m3.metric("Correlation", f"{metrics['correlation']:.3f}" if metrics["n_matched"] else "—")
        m4.metric("Flagged edges", metrics["n_flagged"])
        m5.metric("Odds source", benchmark["source"].split("/")[-1][:20])

        comparison = benchmark["comparison"].dropna(subset=["market_prob"]).head(16).copy()
        if not comparison.empty:
            st.markdown("**Top 16 — model vs market win probability (%)**")
            st.bar_chart(model_vs_market_chart_data(comparison), stack=False, height=380)

        st.divider()
        col_all, col_edges = st.columns(2)
        with col_all:
            st.markdown("**Full comparison**")
            st.dataframe(
                format_benchmark_table(benchmark["comparison"]),
                width="stretch",
                hide_index=True,
            )
        with col_edges:
            st.markdown("**Flagged edges + Kelly**")
            flagged = benchmark["flagged_edges"]
            if flagged.empty:
                st.write("No edges above the configured threshold.")
            else:
                st.dataframe(
                    format_benchmark_table(flagged),
                    width="stretch",
                    hide_index=True,
                )

with tab_match:
    st.subheader("Head-to-head probabilities")
    all_teams = sorted({t for teams in GROUPS_2026.values() for t in teams})

    if not data_ok:
        st.warning("Match predictor needs `results.csv` in `world_cup_model/data_files/`.")
    else:
        _params = fit_ratings(elo_strength, RESULTS_CSV, ELO_CSV if elo_ok else None)
        c1, c2, c3 = st.columns(3)
        with c1:
            home = st.selectbox(
                "Home team",
                all_teams,
                index=all_teams.index("Argentina") if "Argentina" in all_teams else 0,
            )
        with c2:
            away = st.selectbox(
                "Away team",
                all_teams,
                index=all_teams.index("Brazil") if "Brazil" in all_teams else 1,
            )
        with c3:
            neutral = st.checkbox("Neutral venue", value=True)

        if home == away:
            st.error("Pick two different teams.")
        else:
            res = predict_match_probabilities(home, away, _params, neutral=neutral)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"{home} win", f"{res['p_home']*100:.1f}%")
            m2.metric("Draw", f"{res['p_draw']*100:.1f}%")
            m3.metric(f"{away} win", f"{res['p_away']*100:.1f}%")
            m4.metric("Expected goals", f"{res['lambda_home']:.2f} – {res['lambda_away']:.2f}")

with tab_groups:
    st.subheader("2026 group stage draw")
    rows = []
    for letter, teams in GROUPS_2026.items():
        adv = [sim_out["group_advance"].get(t, 0) * 100 for t in teams]
        rows.append({
            "Group": letter,
            "Team 1": f"{teams[0]} ({adv[0]:.0f}%)",
            "Team 2": f"{teams[1]} ({adv[1]:.0f}%)",
            "Team 3": f"{teams[2]} ({adv[2]:.0f}%)",
            "Team 4": f"{teams[3]} ({adv[3]:.0f}%)",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("Percentages = model probability to advance from the group (top 2 + best 3rd places).")

with tab_about:
    st.markdown(
        """
        ### What this project does

        This is a **probabilistic forecasting pipeline** for the 2026 FIFA World Cup:

        1. **Data** — Historical international results (Mart Jürisoo dataset), filtered to
           competitive matches since 2018.
        2. **Dixon-Coles model** — Estimates each nation's attack and defense strength from
           goal scoring, with a correction for low-score draws.
        3. **Tournament weights** — World Cup and continental finals count more than
           regional qualifiers (reduces "farming weak opponents").
        4. **Elo priors** — Ratings from [eloratings.net](https://eloratings.net) anchor
           teams with sparse data.
        5. **Monte Carlo simulation** — The full 48-team group + knockout bracket is simulated
           thousands of times to produce win / finalist / advancement probabilities.
        6. **Market benchmark** — Model win probabilities are compared to bookmaker outright
           odds (live via The Odds API or bundled sample data), with edge detection and
           fractional Kelly stake sizing.

        ### Tech stack

        Python · pandas · scipy · scikit-learn · Streamlit

        ### Repo layout

        `world_cup_model/data/` · `model/` · `simulation/` · `evaluation/` · `config.py`

        ### Disclaimer

        Outputs are **model estimates**, not betting advice. The 2026 draw in `config.py`
        should be updated when the official bracket is finalized.
        """
    )
