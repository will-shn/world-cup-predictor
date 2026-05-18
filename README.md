# World Cup 2026 Prediction Model

A Dixon-Coles bivariate Poisson model plus Monte Carlo tournament simulation,
calibrated against betting markets.

## Quick start

```bash
pip install -r requirements.txt

# Smoke test with synthetic data (no CSV needed)
python -m world_cup_model.main --synthetic --n-sims 5000 --skip-backtest

# Internal sanity test (exercises every module)
python scripts/smoke_internal.py

# Full run with real data (after placing results.csv)
python -m world_cup_model.main --n-sims 100000
```

Benchmark on a typical laptop: 100,000 Monte Carlo tournaments finishes in
~5 seconds; full Dixon-Coles fit on ~5,000 matches finishes in 1-2 seconds.

## Web dashboard (share with a mentor)

An interactive **Streamlit** app lives at `app.py` in the project root.

### Run on your laptop

```bash
pip install -r requirements.txt
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. Use the sidebar to run the model or
load the last saved `world_cup_model/outputs/win_probs.json` instantly.

### Share a public link (free)

1. Push this folder to a **GitHub** repository (private or public).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repo → main file: `app.py`.
4. Deploy. You get a URL like `https://your-app.streamlit.app` to send your mentor.

**For cloud deploy**, include in the repo:

- `app.py`, `requirements.txt`, all of `world_cup_model/`
- `world_cup_model/data_files/results.csv` and `elo_ratings.csv` (or the app
  only shows saved results until data is added)
- `world_cup_model/outputs/win_probs.json` (optional — enables instant load without
  re-running the model on first visit)

Optional secrets (Streamlit → Settings → Secrets): `ODDS_API_KEY` for future
market-odds features.

### What the mentor will see

- Bar chart of **tournament win probabilities**
- Tables for finalist / semifinal / group-advance odds
- **Match predictor** (pick any two teams, get win/draw/loss %)
- **2026 group draw** with advancement %
- **About** tab explaining Dixon-Coles, Elo priors, and Monte Carlo

## Project layout

```
world_cup_model/
├── config.py                 # All hyperparameters, paths, and the 2026 draw
├── main.py                   # End-to-end driver
├── data/
│   ├── fetch.py              # Load CSV / football-data.org / synthetic
│   ├── fetch_elo.py          # Download Elo from eloratings.net TSV files
│   └── clean.py              # Filter friendlies, decay, name standardization
├── features/
│   └── ratings.py            # Elo lookups + priors
├── model/
│   └── dixon_coles.py        # Vectorized DC log-likelihood + fit + prediction
├── simulation/
│   └── tournament.py         # Vectorized Monte Carlo over 48 teams, 64+ matches
└── evaluation/
    ├── calibrate.py          # Brier, log-loss, reliability diagram, backtest
    └── market.py             # Odds API, vig removal, edge detection, Kelly
```

## Data sources

1. **Historical results** - download the Kaggle
   "[International football results from 1872 to 2017](https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017)"
   dataset (continually updated by Mart Jürisoo) and save as
   `world_cup_model/data_files/results.csv`. Required columns: `date,
   home_team, away_team, home_score, away_score, tournament, neutral`.

2. **Elo ratings (optional)** - the site has no download button, but it serves
   the same data as public TSV files. Use the built-in fetcher (no scraping
   needed):

   ```bash
   # Full match-by-match history (~240 teams, ~30 seconds)
   python -m world_cup_model.data.fetch_elo

   # Or today's snapshot only (fast)
   python -m world_cup_model.data.fetch_elo --current-only
   ```

   Writes `world_cup_model/data_files/elo_ratings.csv` (`team`, `date`, `elo`).
   Then run the model with Elo features:

   ```bash
   python -m world_cup_model.main --elo world_cup_model/data_files/elo_ratings.csv
   ```

   The model runs without Elo, but sparse teams will have noisier estimates.

3. **Betting odds** - sign up at [the-odds-api.com](https://the-odds-api.com)
   for a free key and set `ODDS_API_KEY` in `config.py` or as an environment
   variable.

## Updating the 2026 draw

Edit `GROUPS_2026` and `ROUND_OF_32` in `config.py`. The placeholders inside
`config.py` are a reasonable approximation but should be replaced with the
official draw before going live.

## What's vectorized vs. what isn't

* **Dixon-Coles likelihood** - 100% vectorized in NumPy. The optimizer touches
  `negative_log_likelihood` only a few hundred times, but each call is one
  pass of array math rather than a Python loop over rows.
* **Monte Carlo** - all 12 groups and all knockout rounds are simulated as
  `(n_sims, n_matches)` arrays. 100,000 tournaments finish in well under a
  minute.
* **Backtest predictions** - currently iterate per match for readability. If
  you need speed, swap `_predictions_for_matches` in `evaluation/calibrate.py`
  for a batched version using `predict_lambdas_batch`.
