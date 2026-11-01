"""Train the win-probability models and score every ball of every chase.

Usage: python train_model.py
Inputs : data/deliveries.parquet
Outputs: models/winprob.joblib          (situation-only model, used for WPA and match replays)
         models/winprob_players.joblib  (player-aware model, used in the chase calculator)
         models/metrics.json
         data/chase_states.parquet, data/batter_ratings.parquet, data/bowler_ratings.parquet
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from winprob import (FEATURES, MONOTONIC, PLAYER_COLS, PLAYER_FEATURES, PLAYER_MONOTONIC,
                     make_features, predict_win_prob)

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
SHRINK = 120  # balls of "league average" mixed into every player's record (stops tiny samples looking extreme)
NOT_BOWLER_WICKETS = ["run out", "retired hurt", "retired out", "obstructing the field"]


# ----------------------------------------------------------------------------- player ratings
def player_ratings(hist: pd.DataFrame):
    """Batting and bowling ratings from a set of deliveries, shrunk toward the league average."""
    bat = hist.groupby("batter").agg(runs=("runs_batter", "sum"), balls=("is_legal", "sum"),
                                     team=("batting_team", "last"), last_season=("season", "max"))
    outs = hist.dropna(subset=["player_out"]).groupby("player_out").size()
    bat["outs"] = outs.reindex(bat.index).fillna(0)
    lg_rpb = bat.runs.sum() / bat.balls.sum()
    lg_opb = bat.outs.sum() / bat.balls.sum()
    bat["bat_sr"] = 100 * (bat.runs + SHRINK * lg_rpb) / (bat.balls + SHRINK)
    bat["bat_out_rate"] = 100 * (bat.outs + SHRINK * lg_opb) / (bat.balls + SHRINK)  # dismissals per 100 balls

    h = hist.assign(bowler_wkt=(hist.wicket_kind.notna() & ~hist.wicket_kind.isin(NOT_BOWLER_WICKETS)).astype(int))
    bowl = h.groupby("bowler").agg(runs=("runs_total", "sum"), balls=("is_legal", "sum"), wkts=("bowler_wkt", "sum"),
                                   team=("bowling_team", "last"), last_season=("season", "max"))
    lg_bowl_rpb = bowl.runs.sum() / bowl.balls.sum()
    lg_wpb = bowl.wkts.sum() / bowl.balls.sum()
    bowl["bowl_econ"] = 6 * (bowl.runs + SHRINK * lg_bowl_rpb) / (bowl.balls + SHRINK)
    bowl["bowl_wkt_rate"] = 100 * (bowl.wkts + SHRINK * lg_wpb) / (bowl.balls + SHRINK)  # wickets per 100 balls

    league = {"bat_sr": 100 * lg_rpb, "bat_out_rate": 100 * lg_opb,
              "bowl_econ": 6 * lg_bowl_rpb, "bowl_wkt_rate": 100 * lg_wpb}
    return bat, bowl, league


def prior_season_ratings(df: pd.DataFrame, league_fallback: dict):
    """For every season, each player's ratings using ONLY earlier seasons (no peeking at the future)."""
    bat_rows, bowl_rows = [], []
    for s in sorted(df.season.unique()):
        hist = df[df.season < s]
        if hist.empty:
            continue
        bat, bowl, _ = player_ratings(hist)
        bat_rows.append(bat[["bat_sr", "bat_out_rate"]].assign(season=s).reset_index())
        bowl_rows.append(bowl[["bowl_econ", "bowl_wkt_rate"]].assign(season=s).reset_index())
    return pd.concat(bat_rows), pd.concat(bowl_rows)


# ----------------------------------------------------------------------------- chase states
def build_chase_states(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ball of each second innings (state AFTER the ball), plus a 'ball 0' start row."""
    ch = df[(df.inning == 2) & (df.result == "normal") & df.winner.notna() & df.target_runs.notna()].copy()
    ch = ch.reset_index(drop=True)
    ch["ball_seq"] = ch.groupby("match_id").cumcount() + 1
    ch["runs_so_far"] = ch.groupby("match_id")["runs_total"].cumsum()
    ch["wkts_so_far"] = ch.groupby("match_id")["wickets_on_ball"].cumsum()
    ch["runs_needed"] = ch.target_runs - ch.runs_so_far
    ch["balls_left"] = ch.target_overs * 6 - ch.legal_balls
    ch["wickets_left"] = 10 - ch.wkts_so_far
    ch["chaser_won"] = (ch.winner == ch.batting_team).astype(int)

    start = ch.groupby("match_id").first().reset_index()
    start = start.assign(
        ball_seq=0, runs_so_far=0, wkts_so_far=0, runs_needed=start.target_runs,
        balls_left=start.target_overs * 6, wickets_left=10, over=0, legal_balls=0,
        batter=None, bowler=None, runs_batter=0, runs_total=0, wickets_on_ball=0,
        player_out=None, wicket_kind=None, is_legal=False,
    )
    states = pd.concat([start[ch.columns], ch], ignore_index=True)
    return states.sort_values(["match_id", "ball_seq"]).reset_index(drop=True)


def training_rows(states, bat_prior, bowl_prior, league):
    """The state BEFORE each ball, with the batter and bowler about to contest it."""
    r = states[states.ball_seq > 0].copy()
    r["runs_needed"] = r.runs_needed + r.runs_total
    r["balls_left"] = r.balls_left + r.is_legal.astype(int)
    r["wickets_left"] = r.wickets_left + r.wickets_on_ball
    r = r.merge(bat_prior, on=["batter", "season"], how="left").merge(bowl_prior, on=["bowler", "season"], how="left")
    for c in PLAYER_COLS:
        r[c] = r[c].fillna(league[c])  # debutants = league average
    r = r[(r.runs_needed > 0) & (r.balls_left > 0) & (r.wickets_left > 0)]
    X = make_features(r.runs_needed, r.balls_left, r.wickets_left)
    for c in PLAYER_COLS:
        X[c] = r[c].values
    return r, X


def evaluate(y, p, base, late):
    return {"brier": round(brier_score_loss(y, p), 4), "brier_baseline": round(brier_score_loss(y, base), 4),
            "log_loss": round(log_loss(y, p), 4), "accuracy": round(accuracy_score(y, p > 0.5), 4),
            "accuracy_last_5_overs": round(accuracy_score(y[late], p[late] > 0.5), 4)}


def new_model(monotonic):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=100, l2_regularization=1.0,
                                          monotonic_cst=monotonic, random_state=0)


def main():
    df = pd.read_parquet(DATA / "deliveries.parquet")
    # Chronological order (stable, so balls inside each innings keep their order).
    df = df.sort_values(["date", "match_id", "inning"], kind="stable").reset_index(drop=True)
    states = build_chase_states(df)

    _, _, league_all = player_ratings(df)
    bat_prior, bowl_prior = prior_season_ratings(df, league_all)
    rows, X = training_rows(states, bat_prior, bowl_prior, league_all)
    y = rows.chaser_won.values

    # Honest evaluation: train on older seasons, test on the two most recent.
    seasons = sorted(rows.season.unique())
    test_seasons = seasons[-2:]
    is_test = rows.season.isin(test_seasons).values
    late = (rows.balls_left.values[is_test] <= 30)
    base = np.full(is_test.sum(), y[~is_test].mean())

    sit = new_model(MONOTONIC).fit(X.loc[~is_test, FEATURES], y[~is_test])
    ply = new_model(PLAYER_MONOTONIC).fit(X.loc[~is_test, PLAYER_FEATURES], y[~is_test])
    m_sit = evaluate(y[is_test], sit.predict_proba(X.loc[is_test, FEATURES])[:, 1], base, late)
    m_ply = evaluate(y[is_test], ply.predict_proba(X.loc[is_test, PLAYER_FEATURES])[:, 1], base, late)

    metrics = {"train_seasons": f"{seasons[0]}-{seasons[-3]}", "test_seasons": f"{test_seasons[0]}-{test_seasons[-1]}",
               "test_balls": int(is_test.sum()), "test_matches": int(rows[is_test].match_id.nunique()),
               **m_sit, "player_model": m_ply, "league": {k: round(v, 3) for k, v in league_all.items()}}

    # Final models use every season.
    sit.fit(X[FEATURES], y)
    ply.fit(X[PLAYER_FEATURES], y)

    states["win_prob"] = predict_win_prob(sit, states.runs_needed, states.balls_left, states.wickets_left)
    states["wp_before"] = states.groupby("match_id")["win_prob"].shift(1)
    states["wp_delta"] = states.win_prob - states.wp_before  # + helps the chasing side

    # Current ratings (all seasons) for the calculator.
    bat_now, bowl_now, _ = player_ratings(df)

    MODELS.mkdir(exist_ok=True)
    joblib.dump(sit, MODELS / "winprob.joblib")
    joblib.dump(ply, MODELS / "winprob_players.joblib")
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    states.to_parquet(DATA / "chase_states.parquet", index=False)
    bat_now.reset_index().to_parquet(DATA / "batter_ratings.parquet", index=False)
    bowl_now.reset_index().to_parquet(DATA / "bowler_ratings.parquet", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"Scored {len(states):,} chase states from {states.match_id.nunique():,} matches.")


if __name__ == "__main__":
    main()
