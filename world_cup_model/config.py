"""
Global configuration for the World Cup 2026 prediction model.

All hyperparameters, file paths, and tournament structure constants live here.
Nothing inside the modeling, simulation, or evaluation modules should hardcode
values that belong in this file.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PACKAGE_DIR, "data_files")
OUTPUT_DIR = os.path.join(PACKAGE_DIR, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(DATA_DIR, "results.csv")          # Mart Jürisoo dataset
ELO_CSV = os.path.join(DATA_DIR, "elo_ratings.csv")          # eloratings.net export

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
DECAY_RATE = 0.002          # Exponential time decay per day. 0.002 ~= half-life of ~1 year
N_SIMULATIONS = 100_000     # Monte Carlo iterations for full-tournament sim
MIN_DATE = "2018-01-01"     # Earliest match date to include
COMPETITIVE_ONLY = True     # Filter out friendlies
HOME_ADVANTAGE = True       # Include home advantage parameter in Dixon-Coles
RHO_INIT = 0.0              # Initial value for DC low-score correction
HOME_ADV_INIT = 0.25        # Initial value for log home advantage
L2_REG = 0.05               # Fallback ridge penalty (used if no Elo priors)

# Strength of the pull toward Elo-derived priors for attack/defense parameters.
# 0 = ignore Elo entirely. 20 = moderate. 50+ = priors dominate sparse-team fits.
ELO_PRIOR_STRENGTH = 25.0

# Hard bounds on per-team attack/defense in log-goal-rate space.
# Tighter bounds prevent qualifier-farming teams from drifting to extreme values.
ATTACK_BOUNDS = (-1.5, 1.5)
DEFENSE_BOUNDS = (-1.5, 1.5)

# ---------------------------------------------------------------------------
# Extra-time / penalties tuning
# ---------------------------------------------------------------------------
EXTRA_TIME_LAMBDA_SCALE = 1.0 / 3.0   # Extra time = 30 of 90 minutes, so scale lambdas
PENALTY_DEFAULT_WIN_RATE = 0.5

# Historical World Cup / Euro penalty shootout success rates
PENALTY_WIN_RATE = {
    "Germany": 0.71,
    "Argentina": 0.67,
    "Brazil": 0.64,
    "Croatia": 0.75,
    "France": 0.57,
    "England": 0.36,
    "Italy": 0.50,
    "Spain": 0.50,
    "Netherlands": 0.40,
    "Portugal": 0.60,
    "Uruguay": 0.50,
    "Belgium": 0.50,
    "Switzerland": 0.50,
    "Russia": 0.60,
    "Sweden": 0.50,
    "Mexico": 0.40,
}

# ---------------------------------------------------------------------------
# Market / edge configuration
# ---------------------------------------------------------------------------
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_SPORT_KEY = "soccer_fifa_world_cup"              # H2H match odds
OUTRIGHT_SPORT_KEY = "soccer_fifa_world_cup_winner"   # Tournament winner futures
SAMPLE_OUTRIGHT_ODDS = os.path.join(DATA_DIR, "sample_market_outrights.json")
EDGE_THRESHOLD = 0.05          # Min absolute probability gap to flag
KELLY_FRACTION_DIVISOR = 4.0   # Fractional Kelly multiplier

# ---------------------------------------------------------------------------
# Tournament structure - 2026 FIFA World Cup
# ---------------------------------------------------------------------------
# 48 teams, 12 groups of 4. Top 2 from each group plus 8 best 3rd-placed teams
# advance to a 32-team knockout round.
#
# IMPORTANT: replace these groups with the actual final draw if you have it.
# The structure of the dictionary must stay {"A": [t1, t2, t3, t4], ...}.
# Team names must match the standardized names produced by data/clean.py.
GROUPS_2026: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Democratic Republic of the Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Host countries get their group-stage matches treated as home games (not neutral)
HOST_COUNTRIES = ("USA", "Canada", "Mexico")

# ---------------------------------------------------------------------------
# Knockout bracket structure for 2026
# ---------------------------------------------------------------------------
# Each Round of 32 slot is a label that resolves to either:
#   ("group_1st", "<letter>")  -> winner of group
#   ("group_2nd", "<letter>")  -> runner-up of group
#   ("group_3rd", "<rank>")    -> Nth-ranked third-place qualifier (1..8)
#
# The bracket is a list of 16 R32 ties (slot_a vs slot_b). The winners feed the
# Round of 16, then quarters, semis, and final. The exact mapping below is a
# best-effort approximation of the FIFA 2026 bracket; adjust as needed once the
# final R32 pairings are published.
ROUND_OF_32 = [
    # Top half
    (("group_1st", "A"), ("group_3rd", 1)),
    (("group_2nd", "B"), ("group_2nd", "F")),
    (("group_1st", "C"), ("group_3rd", 2)),
    (("group_2nd", "E"), ("group_2nd", "G")),
    (("group_1st", "B"), ("group_3rd", 3)),
    (("group_2nd", "A"), ("group_2nd", "D")),
    (("group_1st", "F"), ("group_2nd", "C")),
    (("group_1st", "D"), ("group_3rd", 4)),
    # Bottom half
    (("group_1st", "E"), ("group_3rd", 5)),
    (("group_2nd", "H"), ("group_2nd", "L")),
    (("group_1st", "G"), ("group_3rd", 6)),
    (("group_2nd", "I"), ("group_2nd", "K")),
    (("group_1st", "H"), ("group_3rd", 7)),
    (("group_2nd", "J"), ("group_1st", "L")),
    (("group_1st", "I"), ("group_3rd", 8)),
    (("group_1st", "J"), ("group_1st", "K")),
]

# ---------------------------------------------------------------------------
# Competitive tournaments used to filter friendlies
# ---------------------------------------------------------------------------
COMPETITIVE_TOURNAMENTS = [
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "UEFA Euro",
    "UEFA Euro qualification",
    "Copa América",
    "African Cup of Nations",
    "African Cup of Nations qualification",
    "AFC Asian Cup",
    "AFC Asian Cup qualification",
    "UEFA Nations League",
    "CONCACAF Nations League",
    "CONCACAF Gold Cup",
    "Confederations Cup",
]

# ---------------------------------------------------------------------------
# Tournament importance weights
# ---------------------------------------------------------------------------
# Multiplied INTO the time-decay weight so that a recent World Cup match counts
# more than a recent Asian Cup qualifier. Tweak these to taste; the relative
# ratios matter more than the absolute numbers.
TOURNAMENT_WEIGHTS = {
    # Top tier - real-stakes knockout football
    "FIFA World Cup": 4.0,
    "UEFA Euro": 2.5,
    "Copa América": 2.5,
    "African Cup of Nations": 2.2,
    "AFC Asian Cup": 2.0,
    "Confederations Cup": 1.5,

    # Strong continental league play
    "UEFA Nations League": 1.5,
    "CONCACAF Nations League": 1.0,
    "CONCACAF Gold Cup": 1.0,

    # Qualifiers - relevant but full of mismatches; downweight
    "FIFA World Cup qualification": 0.7,
    "UEFA Euro qualification": 0.6,
    "African Cup of Nations qualification": 0.5,
    "AFC Asian Cup qualification": 0.4,
}

# Fallback when a tournament name slips past the filter; should be small.
DEFAULT_TOURNAMENT_WEIGHT = 0.3

# ---------------------------------------------------------------------------
# Team name standardization map (extend as needed when joining datasets)
# ---------------------------------------------------------------------------
NAME_MAP = {
    "United States": "USA",
    "United States of America": "USA",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Iran (Islamic Republic of)": "Iran",
    "Czechia": "Czech Republic",
    "Cape Verde Islands": "Cape Verde",
    "Cabo Verde": "Cape Verde",
    "Côte d'Ivoire": "Ivory Coast",
    "Republic of Ireland": "Ireland",
    "FYR Macedonia": "North Macedonia",
    "Macedonia": "North Macedonia",
    "Curaçao": "Curacao",
}
