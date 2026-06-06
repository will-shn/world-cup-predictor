# World Cup 2026 Probabilistic Forecasting Engine

A production-style Python pipeline that estimates **full probability distributions** over 2026 FIFA World Cup outcomes — not single picks. Built for quantitative rigor: statistical modeling, large-scale simulation, calibration against bookmaker markets, and an interactive dashboard for exploration.

**Relevant for:** quantitative research, sales & trading, data science, and software engineering roles where probabilistic thinking, Python, and market benchmarking matter.

---

## What it does

| Capability | Description |
|------------|-------------|
| **Match-level pricing** | Win / draw / loss probabilities and expected goals for any international fixture |
| **Tournament simulation** | 100,000 Monte Carlo paths through the full 48-team bracket (groups + knockouts) |
| **Team strength estimation** | Attack/defense parameters from 4,600+ weighted historical matches (Dixon-Coles MLE) |
| **Market benchmarking** | Compare model probabilities to bookmaker implied odds; edge detection and fractional Kelly sizing |
| **Model validation** | Brier score, log-loss, reliability diagrams, and chronological backtesting |

**Example output:** Spain ~14% to win the tournament, Argentina ~6%, with full tables for finalist, semifinal, and group-advancement probabilities — updated when you re-run the model.

---

## Why this project matters (quant / S&T angle)

Bookmakers and trading desks price **events as probabilities**, not narratives. This project mirrors that workflow:

1. **Ingest** structured historical data with recency and competition-importance weighting  
2. **Fit** a peer-reviewed Poisson model (Dixon & Coles, 1997) with Elo regularization  
3. **Simulate** path-dependent tournament risk at scale (vectorized NumPy)  
4. **Calibrate** forecasts against realized outcomes and live market prices  
5. **Ship** results through a Streamlit UI for non-technical stakeholders  

It is a **research prototype**, not a live trading system — but the architecture (modular pipeline, config-driven parameters, evaluation layer) reflects how real forecasting stacks are organized.

---

## Technical highlights

- **Dixon-Coles bivariate Poisson** — MLE via L-BFGS-B; low-score correlation (ρ); neutral-venue and home-advantage handling  
- **Weighted likelihood** — exponential time decay + tournament multipliers (World Cup ×4 vs regional qualifiers ×0.4)  
- **Elo priors** — attack/defense anchored to [eloratings.net](https://eloratings.net); fixes sparse-team overfitting  
- **Vectorized Monte Carlo** — ~100,000 full tournaments in **~5–10 seconds** on a laptop  
- **48-team 2026 format** — 12 groups, 8 best third-place qualifiers, 32-team knockout bracket with extra time and penalties  
- **Market module** — The Odds API integration, vig removal, edge flags, Kelly criterion  

---

## Tech stack

`Python 3.11+` · `pandas` · `NumPy` · `SciPy` · `scikit-learn` · `matplotlib` · `Streamlit` · `requests` · `reportlab`

---

## Architecture

```
Historical results CSV          Elo ratings (eloratings.net)
         │                                  │
         ▼                                  ▼
    ┌─────────┐                      ┌──────────┐
    │  Data   │  filter · decay ·    │ Features │  Elo at match date
    │ pipeline│  tournament weights  │ (priors) │
    └────┬────┘                      └────┬─────┘
         │                                │
         └────────────┬───────────────────┘
                      ▼
               ┌─────────────┐
               │ Dixon-Coles │  attack / defense / home_adv / ρ
               │    (MLE)    │
               └──────┬──────┘
                      ▼
               ┌─────────────┐
               │ Monte Carlo │  100k × full 2026 bracket
               │ simulation  │
               └──────┬──────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
   Calibration   Market compare   Streamlit
   Brier · backtest  edges · Kelly   dashboard
```

Each stage is a **separate module** with clean interfaces (DataFrames in, parameter dicts out), so components can be tested and swapped independently.

---

## Repository structure

```
world_cup_model/
├── config.py              # All hyperparameters, 2026 draw, tournament weights
├── main.py                # CLI: load → fit → simulate → backtest
├── data/                  # Ingestion, cleaning, Elo download
├── features/              # Elo lookups and prior conversion
├── model/                 # Dixon-Coles fit and match prediction
├── simulation/            # Vectorized World Cup Monte Carlo
└── evaluation/            # Calibration metrics and market comparison

app.py                     # Interactive Streamlit dashboard
docs/World_Cup_Engine_Guide.pdf   # Full technical reference (concepts + glossary)
```

---

## Skills demonstrated

- **Probability & statistics** — Poisson models, maximum likelihood, regularization, calibration metrics  
- **Simulation** — Monte Carlo at scale, vectorization for performance  
- **Data engineering** — ETL from CSV/API, name standardization, time-weighted features  
- **Software design** — modular packages, centralized config, CLI + web UI  
- **Markets literacy** — implied probability, vig, edge, Kelly sizing (evaluation layer)  
- **Communication** — technical PDF guide, interactive dashboard, structured documentation  

---

## Documentation

| Resource | Audience |
|----------|----------|
| **[docs/World_Cup_Engine_Guide.pdf](docs/World_Cup_Engine_Guide.pdf)** | Deep dive: concepts, glossary, formulas, every file explained |
| **[docs/ENGINE_GUIDE.md](docs/ENGINE_GUIDE.md)** | Same content in Markdown (editable source) |

---

## Demo (interactive)

A **Streamlit** dashboard (`app.py`) provides:

- Tournament win-probability charts and tables  
- Head-to-head match predictor (win / draw / loss %)  
- 2026 group draw with advancement probabilities  
- Adjustable simulation count and Elo prior strength  

**Run locally:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Deploy online (free):** push to GitHub → [share.streamlit.io](https://share.streamlit.io) → main file `app.py` → share the public URL with reviewers.

---

## Quick run (for technical reviewers)

```bash
pip install -r requirements.txt

# Place historical results in world_cup_model/data_files/results.csv
# (Kaggle: "international football results" by Mart Jurisoo)

# Optional: download Elo ratings (~45 s)
python -m world_cup_model.data.fetch_elo

# Full pipeline: fit + 100,000 simulations + backtest
python -m world_cup_model.main --n-sims 100000
```

Outputs land in `world_cup_model/outputs/win_probs.json` and calibration plots in `world_cup_model/outputs/`.

---

## Data sources

- **Match results** — [International football results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) (Mart Jürisoo, Kaggle)  
- **Elo ratings** — [eloratings.net](https://eloratings.net) (built-in TSV fetcher)  
- **Bookmaker odds** — [The Odds API](https://the-odds-api.com) (optional; requires API key)  

---

## Disclaimer

This project is for **education and research**. Outputs are model estimates, not betting or investment advice. Tournament draw configuration should be updated when official FIFA bracket rules are finalized.

---

## Contact

William Shan - william.shan@mail.utoronto.ca - https://www.linkedin.com/in/william-shan/ 
