"""Download IPL ball-by-ball data from Cricsheet and flatten it into one table.

Usage:
    python prepare_data.py              # download from cricsheet.org (recommended)
    python prepare_data.py --zip FILE   # use an ipl_json.zip you already downloaded
    python prepare_data.py --demo       # synthetic data, only for testing the code offline

Output: data/deliveries.parquet (one row per ball)
"""
import argparse
import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

CRICSHEET_URL = "https://cricsheet.org/downloads/ipl_json.zip"
DATA_DIR = Path(__file__).parent / "data"


def parse_match(match_id: str, m: dict) -> list[dict]:
    info = m["info"]
    teams = info.get("teams", [])
    outcome = info.get("outcome", {})
    result = outcome.get("result", "normal")  # "normal", "tie", "no result"
    winner = outcome.get("winner") or outcome.get("eliminator")
    date = info.get("dates", [None])[0]
    rows = []
    first_innings_total = None
    for inn_idx, inn in enumerate(m.get("innings", [])):
        if inn.get("super_over") or inn_idx > 1:
            continue
        bat = inn["team"]
        bowl = next((t for t in teams if t != bat), None)
        target = inn.get("target", {})
        target_runs = target.get("runs")
        target_overs = target.get("overs", 20)
        if inn_idx == 1 and target_runs is None and first_innings_total is not None:
            target_runs = first_innings_total + 1
        legal = 0
        total = 0
        for ov in inn.get("overs", []):
            for d in ov["deliveries"]:
                extras = d.get("extras", {})
                is_wide = "wides" in extras
                is_noball = "noballs" in extras
                is_legal = not (is_wide or is_noball)
                legal += int(is_legal)
                total += d["runs"]["total"]
                wkts = d.get("wickets", [])
                rows.append(
                    {
                        "match_id": match_id,
                        "date": date,
                        "venue": info.get("venue"),
                        "inning": inn_idx + 1,
                        "batting_team": bat,
                        "bowling_team": bowl,
                        "over": ov["over"],
                        "legal_balls": legal,
                        "batter": d["batter"],
                        "non_striker": d.get("non_striker"),
                        "bowler": d["bowler"],
                        "runs_batter": d["runs"]["batter"],
                        "runs_extras": d["runs"]["extras"],
                        "runs_total": d["runs"]["total"],
                        "is_wide": is_wide,
                        "is_noball": is_noball,
                        "is_legal": is_legal,
                        "wickets_on_ball": len(wkts),
                        "player_out": wkts[0]["player_out"] if wkts else None,
                        "wicket_kind": wkts[0]["kind"] if wkts else None,
                        "target_runs": target_runs if inn_idx == 1 else None,
                        "target_overs": target_overs if inn_idx == 1 else None,
                        "winner": winner,
                        "result": result,
                    }
                )
        if inn_idx == 0:
            first_innings_total = total
    return rows


def load_zip(zip_bytes: bytes) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = [n for n in z.namelist() if n.endswith(".json")]
        for i, name in enumerate(names, 1):
            m = json.loads(z.read(name))
            rows.extend(parse_match(Path(name).stem, m))
            if i % 200 == 0:
                print(f"  parsed {i}/{len(names)} matches")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Synthetic data in Cricsheet's format, ONLY for testing without internet.
# Do not submit an analysis built on this.
# --------------------------------------------------------------------------
def _demo_matches(n_matches=400, seed=7):
    rng = random.Random(seed)
    teams = [f"Team {c}" for c in "ABCDEFGH"]
    squads = {t: [f"{t[-1]}. Batter{i}" for i in range(1, 12)] for t in teams}
    for idx in range(n_matches):
        a, b = rng.sample(teams, 2)
        year = 2015 + idx * 10 // n_matches
        innings, totals = [], []
        for inn_idx, (bat, bowl) in enumerate([(a, b), (b, a)]):
            bowlers = squads[bowl][6:]
            overs, total, wkts, legal = [], 0, 0, 0
            target = totals[0] + 1 if inn_idx == 1 else None
            for over in range(20):
                dels, balls = [], 0
                bowler = bowlers[over % 5]
                while balls < 6 and wkts < 10 and not (target and total >= target):
                    striker = squads[bat][min(wkts, 10)]
                    aggression = 1.0
                    if target:
                        rrr = (target - total) * 6 / max(120 - legal, 1)
                        aggression = min(max(rrr / 8, 0.7), 1.8)
                    if over >= 15:
                        aggression *= 1.25
                    d = {"batter": striker, "bowler": bowler, "non_striker": squads[bat][min(wkts + 1, 10)]}
                    if rng.random() < 0.04:
                        d["runs"] = {"batter": 0, "extras": 1, "total": 1}
                        d["extras"] = {"wides": 1}
                        total += 1
                    else:
                        balls += 1
                        legal += 1
                        if rng.random() < 0.045 * aggression:
                            wkts += 1
                            d["runs"] = {"batter": 0, "extras": 0, "total": 0}
                            d["wickets"] = [{"player_out": striker, "kind": rng.choice(["caught", "bowled", "lbw", "run out"])}]
                        else:
                            r = rng.choices([0, 1, 2, 4, 6], weights=[38 / aggression, 36, 8, 12 * aggression, 5 * aggression])[0]
                            d["runs"] = {"batter": r, "extras": 0, "total": r}
                            total += r
                    dels.append(d)
                if dels:
                    overs.append({"over": over, "deliveries": dels})
                if wkts >= 10 or (target and total >= target):
                    break
            inn = {"team": bat, "overs": overs}
            if target:
                inn["target"] = {"runs": target, "overs": 20}
            innings.append(inn)
            totals.append(total)
        winner = b if totals[1] > totals[0] else a if totals[0] > totals[1] else None
        outcome = {"winner": winner} if winner else {"result": "tie"}
        yield f"demo{idx}", {
            "info": {"dates": [f"{year}-04-{1 + idx % 28:02d}"], "teams": [a, b], "venue": "Demo Stadium", "outcome": outcome},
            "innings": innings,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", help="path to an already-downloaded ipl_json.zip")
    ap.add_argument("--demo", action="store_true", help="generate synthetic data for offline testing")
    args = ap.parse_args()
    DATA_DIR.mkdir(exist_ok=True)

    if args.demo:
        rows = [r for mid, m in _demo_matches() for r in parse_match(mid, m)]
        df = pd.DataFrame(rows)
    else:
        if args.zip:
            zip_bytes = Path(args.zip).read_bytes()
        else:
            print(f"Downloading {CRICSHEET_URL} ...")
            req = urllib.request.Request(CRICSHEET_URL, headers={"User-Agent": "Mozilla/5.0"})
            zip_bytes = urllib.request.urlopen(req, timeout=120).read()
        df = load_zip(zip_bytes)

    df["date"] = pd.to_datetime(df["date"])
    df["season"] = df["date"].dt.year
    out = DATA_DIR / "deliveries.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {len(df):,} deliveries from {df.match_id.nunique():,} matches -> {out}")


if __name__ == "__main__":
    main()
