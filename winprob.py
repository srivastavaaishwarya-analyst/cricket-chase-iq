"""Shared win-probability logic used by both the training script and the app."""
import numpy as np
import pandas as pd

# Situation-only model
FEATURES = ["runs_needed", "balls_left", "wickets_left", "req_rate"]
MONOTONIC = [-1, 1, 1, -1]  # more runs needed -> worse; more balls/wickets -> better; higher RRR -> worse

# Player-aware model: the situation + who is batting and who is bowling
PLAYER_COLS = ["bat_sr", "bat_out_rate", "bowl_econ", "bowl_wkt_rate"]
PLAYER_FEATURES = FEATURES + PLAYER_COLS
# better batter strike rate -> better; batter gets out more often -> worse;
# bowler more expensive -> better for chasers; bowler takes more wickets -> worse
PLAYER_MONOTONIC = MONOTONIC + [1, -1, 1, -1]


def make_features(runs_needed, balls_left, wickets_left) -> pd.DataFrame:
    runs_needed = np.asarray(runs_needed, dtype=float)
    balls_left = np.asarray(balls_left, dtype=float)
    wickets_left = np.asarray(wickets_left, dtype=float)
    req_rate = np.clip(runs_needed * 6 / np.maximum(balls_left, 1), 0, 36)
    return pd.DataFrame({"runs_needed": runs_needed, "balls_left": balls_left,
                         "wickets_left": wickets_left, "req_rate": req_rate})


def make_player_features(runs_needed, balls_left, wickets_left, bat_sr, bat_out_rate, bowl_econ, bowl_wkt_rate):
    X = make_features(runs_needed, balls_left, wickets_left)
    n = len(X)
    for col, val in zip(PLAYER_COLS, [bat_sr, bat_out_rate, bowl_econ, bowl_wkt_rate]):
        X[col] = np.broadcast_to(np.asarray(val, dtype=float), (n,))
    return X


def _apply_rules(p, X):
    """Enforce the rules of cricket for finished states."""
    rn, bl, wl = X["runs_needed"].values, X["balls_left"].values, X["wickets_left"].values
    p = np.where(rn <= 0, 1.0, p)
    return np.where((rn > 0) & ((wl <= 0) | (bl <= 0)), 0.0, p)


def predict_win_prob(model, runs_needed, balls_left, wickets_left) -> np.ndarray:
    X = make_features(runs_needed, balls_left, wickets_left)
    return _apply_rules(model.predict_proba(X[FEATURES])[:, 1], X)


def predict_player_win_prob(model, runs_needed, balls_left, wickets_left,
                            bat_sr, bat_out_rate, bowl_econ, bowl_wkt_rate) -> np.ndarray:
    X = make_player_features(runs_needed, balls_left, wickets_left, bat_sr, bat_out_rate, bowl_econ, bowl_wkt_rate)
    return _apply_rules(model.predict_proba(X[PLAYER_FEATURES])[:, 1], X)


def phase_of_over(over: pd.Series) -> pd.Series:
    """Cricsheet overs are 0-indexed: 0-5 powerplay, 6-14 middle, 15+ death."""
    return pd.cut(over, bins=[-1, 5, 14, 99], labels=["Powerplay (1-6)", "Middle (7-15)", "Death (16-20)"])
