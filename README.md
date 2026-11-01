# 🏏 Chase IQ: IPL Win Probability & Clutch-Player Analytics

**AQX Sports Analytics Data Bowl 4.0 submission**

## The problem
In T20 cricket, raw stats lie about who wins chases. A batter who scores 40 off 30 when the game is already won
looks the same on a scorecard as one who hits 40 off 20 with 60 needed from 30. Captains, analysts and fantasy
players need to know **how likely a chase is to succeed right now**, and **which players move that number**.

## The solution
Chase IQ is an interactive web app built on every ball of every IPL run chase (data from Cricsheet):

1. **Win-probability model.** A gradient-boosted classifier predicts the chasing team's chance of winning from
   runs needed, balls left, wickets in hand and the required rate. Monotonic constraints keep it consistent with
   cricket logic: more wickets in hand can never lower a team's chances. It's trained on older seasons and tested
   on the most recent two.
2. **🏏 Live chase predictor.** Enter any match situation **and pick the batter on strike and the bowler**
   (for example, 50 needed off 30 with Virat Kohli facing Jasprit Bumrah). A second, **player-aware model** adjusts the
   win probability using each player's IPL record. It also shows what the next ball is worth, and ranks a team's
   bowlers by who gives the fielding side the best chance (**"Who should bowl next?"**).
3. **📈 Match replay.** The win-probability curve for any IPL chase, with wickets marked and the biggest
   swing moments listed.
4. **🏆 Clutch leaderboard.** Players ranked by **Win Probability Added (WPA)**. Each ball's change in win
   probability is credited to the batter (+) and the bowler (−), which shows who actually wins chases rather
   than who pads stats.
5. **🧠 The model.** Side-by-side test results for the baseline, the situation model and the player-aware model.
6. **👤 Player profile.** Strike rate and economy by phase (powerplay, middle and death overs) and
   head-to-head matchups.

## Actionable impact
- **Team strategy:** the next-ball values show when to take a risk (a dot ball costs X%, a six gains Y%) and
  when protecting wickets matters more than scoring.
- **Selection & auctions:** WPA identifies undervalued finishers and death bowlers whose impact strike rate
  and economy don't show.
- **Fans & broadcasters:** live win probability and match replays make the swings of a chase easy to see.

## Run it locally
```bash
pip install -r requirements.txt
python prepare_data.py      # downloads IPL ball-by-ball data from cricsheet.org
python train_model.py       # trains both models and scores every chase ball
python -m streamlit run app.py
```
If the download is blocked, download `ipl_json.zip` from https://cricsheet.org/downloads/ in a browser and run
`python prepare_data.py --zip ipl_json.zip`.

## Tech
Python, pandas, scikit-learn (HistGradientBoostingClassifier), Plotly and Streamlit.

## Data & credit
Ball-by-ball data: [Cricsheet](https://cricsheet.org), under the Open Data Commons Attribution License.
