"""Chase IQ: IPL win probability & clutch-player analytics.  Run: python -m streamlit run app.py"""
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from winprob import phase_of_over, predict_player_win_prob, predict_win_prob

ROOT = Path(__file__).parent
BAT_BLUE, BOWL_ORANGE, GOLD, PITCH = "#3D8BFF", "#FF7A1A", "#FFC83D", "#2BB673"
GRID = "rgba(255,255,255,0.10)"
TEAM_COLORS = {  # one distinct colour per franchise (renamed franchises share a colour)
    "Chennai Super Kings": "#FDB913",          # yellow
    "Mumbai Indians": "#1E6FE0",               # blue
    "Royal Challengers Bangalore": "#E2231A",  # red
    "Royal Challengers Bengaluru": "#E2231A",
    "Kolkata Knight Riders": "#8E5CD9",        # purple
    "Sunrisers Hyderabad": "#FF822A",          # orange
    "Rajasthan Royals": "#EA1A85",             # pink
    "Delhi Capitals": "#4FC3F7",               # sky blue
    "Delhi Daredevils": "#4FC3F7",
    "Punjab Kings": "#C0C6D0",                 # silver
    "Kings XI Punjab": "#C0C6D0",
    "Gujarat Titans": "#C5A572",               # sand gold
    "Lucknow Super Giants": "#00C2A8",         # teal
    "Deccan Chargers": "#7D8CA3",              # slate
    "Rising Pune Supergiant": "#D16BD6", "Rising Pune Supergiants": "#D16BD6",  # magenta
    "Pune Warriors": "#9CCC65",                # lime
    "Gujarat Lions": "#F26B38",                # coral
    "Kochi Tuskers Kerala": "#A1887F",         # brown
}
CURRENT_TEAMS = ["Chennai Super Kings", "Delhi Capitals", "Gujarat Titans", "Kolkata Knight Riders",
                 "Lucknow Super Giants", "Mumbai Indians", "Punjab Kings", "Rajasthan Royals",
                 "Royal Challengers Bengaluru", "Sunrisers Hyderabad"]
# Cricsheet names players as initials + surname; show full names for well-known players so they're easy to find.
FULL_NAMES = {
    "V Kohli": "Virat Kohli", "RG Sharma": "Rohit Sharma", "MS Dhoni": "MS Dhoni", "JJ Bumrah": "Jasprit Bumrah",
    "SA Yadav": "Suryakumar Yadav", "HH Pandya": "Hardik Pandya", "RA Jadeja": "Ravindra Jadeja",
    "Shubman Gill": "Shubman Gill", "YBK Jaiswal": "Yashasvi Jaiswal", "RR Pant": "Rishabh Pant",
    "KL Rahul": "KL Rahul", "SV Samson": "Sanju Samson", "Rashid Khan": "Rashid Khan", "YS Chahal": "Yuzvendra Chahal",
    "Arshdeep Singh": "Arshdeep Singh", "Mohammed Siraj": "Mohammed Siraj", "B Kumar": "Bhuvneshwar Kumar",
    "AD Russell": "Andre Russell", "SP Narine": "Sunil Narine", "H Klaasen": "Heinrich Klaasen",
    "TM Head": "Travis Head", "Abhishek Sharma": "Abhishek Sharma", "RD Gaikwad": "Ruturaj Gaikwad",
    "SS Iyer": "Shreyas Iyer", "N Pooran": "Nicholas Pooran", "JC Buttler": "Jos Buttler", "PJ Cummins": "Pat Cummins",
    "Kuldeep Yadav": "Kuldeep Yadav", "AR Patel": "Axar Patel", "V Suryavanshi": "Vaibhav Sooryavanshi",
    "SL Malinga": "Lasith Malinga", "AB de Villiers": "AB de Villiers", "CH Gayle": "Chris Gayle",
    "DA Warner": "David Warner", "KA Pollard": "Kieron Pollard", "R Ashwin": "Ravichandran Ashwin",
    "TA Boult": "Trent Boult", "Mohammed Shami": "Mohammed Shami", "SN Thakur": "Shardul Thakur",
    "Tilak Varma": "Tilak Varma", "RK Singh": "Rinku Singh", "PD Salt": "Phil Salt", "MP Stoinis": "Marcus Stoinis",
    "GJ Maxwell": "Glenn Maxwell", "F du Plessis": "Faf du Plessis", "AK Markram": "Aiden Markram",
    "M Pathirana": "Matheesha Pathirana", "MA Starc": "Mitchell Starc", "Harshit Rana": "Harshit Rana",
    "Ravi Bishnoi": "Ravi Bishnoi", "PP Shaw": "Prithvi Shaw", "SC Kuggeleijn": "Scott Kuggeleijn",
}


def team_color(t):
    return TEAM_COLORS.get(t, GOLD)


st.set_page_config(page_title="Chase IQ: IPL Win Probability", page_icon="🏏", layout="wide")

# ---------------------------------------------------------------- styling
st.markdown(f"""
<style>
.block-container {{padding-top: 1.6rem;}}
.hero {{
  position: relative; overflow: hidden; border-radius: 22px; padding: 28px 34px 34px 34px; margin-bottom: 18px;
  background: linear-gradient(115deg, #1B2A80 0%, #4B1FA0 45%, #C2185B 78%, #FF7A1A 100%);
  box-shadow: 0 10px 30px rgba(0,0,0,.35);
}}
.hero:after {{ /* the pitch */
  content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 12px;
  background: repeating-linear-gradient(90deg, {PITCH} 0 40px, #239C61 40px 80px);
}}
.hero h1 {{ color: #fff; font-size: 3rem; margin: 0; letter-spacing: .5px; }}
.hero .tag {{ color: #FFE7C2; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; font-size: .85rem; }}
.hero p {{ color: #F4F1FF; font-size: 1.08rem; max-width: 760px; margin: 8px 0 0 0; }}
.hero svg {{ position: absolute; right: 30px; top: 14px; height: 150px; }}
[data-testid="stMetric"] {{
  background: linear-gradient(160deg, #18265A, #111B42); border: 1px solid rgba(255,255,255,.08);
  border-left: 5px solid {BOWL_ORANGE}; border-radius: 16px; padding: 14px 18px;
}}
[data-testid="stMetricValue"] {{ font-weight: 800; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
.stTabs [data-baseweb="tab"] {{
  background: #15214A; border-radius: 12px 12px 0 0; padding: 10px 18px; font-weight: 700;
}}
.stTabs [aria-selected="true"] {{ background: linear-gradient(180deg, #FF7A1A, #E0560B) !important; color: #fff !important; }}
.card {{ border-radius: 16px; padding: 14px 18px; margin-bottom: 10px; border: 1px solid rgba(255,255,255,.08); }}
.card .who {{ font-size: 1.15rem; font-weight: 800; color: #fff; }}
.card .stat {{ color: #DDE3FF; font-size: .95rem; }}
.bigprob {{ font-size: 4.2rem; font-weight: 900; line-height: 1; }}
.scoreline {{ font-size: 1.1rem; color: #DDE3FF; }}
</style>
""", unsafe_allow_html=True)

HERO_SVG = """
<svg viewBox="0 0 220 160" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <defs>
    <radialGradient id="ball" cx="35%" cy="35%" r="70%">
      <stop offset="0%" stop-color="#FF6B6B"/><stop offset="60%" stop-color="#C1121F"/><stop offset="100%" stop-color="#6A040F"/>
    </radialGradient>
    <linearGradient id="wood" x1="0" x2="1"><stop offset="0%" stop-color="#F6D7A0"/><stop offset="100%" stop-color="#D9A75F"/></linearGradient>
  </defs>
  <g transform="rotate(-35 110 80)">
    <rect x="98" y="4" width="14" height="46" rx="6" fill="#1E1E2E"/>
    <rect x="98" y="10" width="14" height="4" fill="#FF7A1A"/><rect x="98" y="20" width="14" height="4" fill="#FF7A1A"/>
    <rect x="98" y="30" width="14" height="4" fill="#FF7A1A"/>
    <path d="M90 50 h30 v86 q0 12 -15 12 q-15 0 -15 -12 z" fill="url(#wood)" stroke="#B07D3A" stroke-width="2"/>
    <line x1="105" y1="58" x2="105" y2="140" stroke="#C9934B" stroke-width="2"/>
  </g>
  <circle cx="170" cy="112" r="30" fill="url(#ball)"/>
  <path d="M152 88 q18 24 0 48" stroke="#FFF" stroke-width="2.2" fill="none" stroke-dasharray="3 3"/>
  <path d="M158 86 q18 26 0 52" stroke="#FFF" stroke-width="2.2" fill="none" stroke-dasharray="3 3"/>
</svg>
"""


def style(fig, height=380, ytitle=None, pct=False, hover="x unified"):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode=hover,
                      legend=dict(orientation="h", y=1.1, x=0), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E8ECFF"))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title=ytitle, tickformat=".0%" if pct else None)
    return fig


# ---------------------------------------------------------------- data
@st.cache_data
def load_data():
    deliveries = pd.read_parquet(ROOT / "data" / "deliveries.parquet")
    states = pd.read_parquet(ROOT / "data" / "chase_states.parquet")
    bat_r = pd.read_parquet(ROOT / "data" / "batter_ratings.parquet")
    bowl_r = pd.read_parquet(ROOT / "data" / "bowler_ratings.parquet")
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    return deliveries, states, bat_r, bowl_r, metrics


@st.cache_resource
def load_models():
    return joblib.load(ROOT / "models" / "winprob.joblib"), joblib.load(ROOT / "models" / "winprob_players.joblib")


try:
    deliveries, states, bat_r, bowl_r, metrics = load_data()
    model, player_model = load_models()
except FileNotFoundError:
    st.error("Data not found. Run `python prepare_data.py` and then `python train_model.py` first.")
    st.stop()
league = metrics["league"]

# ---------------------------------------------------------------- header
st.markdown(f"""
<div class="hero">{HERO_SVG}
  <div class="tag">🏏 IPL Analytics · {deliveries.season.min()}–{deliveries.season.max()}</div>
  <h1>Chase IQ</h1>
  <p><b>How likely is the chasing team to win right now, given who's batting and who's bowling?</b>
  A machine-learning model trained on every ball of IPL run chases, plus a clutch ranking of the players who
  actually swing matches.</p>
</div>""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("🏟️ Matches analysed", f"{deliveries.match_id.nunique():,}")
c2.metric("⚾ Deliveries", f"{len(deliveries):,}")
c3.metric(f"🎯 Test accuracy ({metrics['test_seasons']})", f"{metrics['accuracy']:.0%}",
          help="Share of balls in held-out seasons where the side the model favoured went on to win.")
c4.metric("📉 Brier score (lower = better)", f"{metrics['brier']:.3f}",
          delta=f"{metrics['brier'] - metrics['brier_baseline']:+.3f} vs. baseline", delta_color="inverse",
          help="Baseline always predicts the historical chase win rate.")

tab_calc, tab_replay, tab_clutch, tab_player, tab_model = st.tabs(
    ["🏏 Live chase predictor", "📈 Match replay", "🏆 Clutch leaderboard", "👤 Player profile", "🧠 The model"])

# ---------------------------------------------------------------- live chase predictor
with tab_calc:
    st.markdown("Set the match situation, then pick **who's on strike** and **who's bowling**. "
                "The player-aware model adjusts the win chance using each player's IPL record.")
    st.info("💡 **Example loaded:** Virat Kohli batting against Jasprit Bumrah, needing 50 off 30 balls. "
            "Change the score, overs, wickets or players on the left (type any name to search).")
    left, right = st.columns([1, 2], gap="large")
    latest = int(bowl_r.last_season.max())
    AVG = "— Average player —"
    ALL = "All teams (every IPL player)"

    def label(name, team=None):
        if name == AVG:
            return name
        full = FULL_NAMES.get(name)
        nm = f"{full} ({name})" if full and full != name else name
        return f"{nm} · {team}" if team else nm

    def default_index(options, needle):
        hits = [i for i, n in enumerate(options) if needle.lower() in str(n).lower()]
        return hits[0] if hits else 0

    with left:
        target = st.number_input("🎯 Target", 50, 300, 181)
        score = st.number_input("🏏 Current score", 0, 300, 131)
        wkts = st.slider("❌ Wickets lost", 0, 9, 4)
        overs = st.slider("⏱️ Overs completed", 0, 19, 15)
        balls = st.slider("Extra balls in current over", 0, 5, 0)
        team_opts = [ALL] + [t for t in CURRENT_TEAMS if t in set(bat_r.team)]
        bat_team = st.selectbox("Batting team", team_opts,
                                index=team_opts.index("Royal Challengers Bengaluru") if "Royal Challengers Bengaluru" in team_opts else 0)
        bat_pool = bat_r if bat_team == ALL else bat_r[(bat_r.team == bat_team) & (bat_r.last_season == latest)]
        bat_pool = bat_pool[bat_pool.balls >= 20].sort_values("balls", ascending=False)
        batter_names = [AVG] + bat_pool.batter.tolist()
        bat_team_of = dict(zip(bat_r.batter, bat_r.team))
        batter = st.selectbox("🏏 Batter on strike (type to search)", batter_names,
                              index=default_index(batter_names, "Kohli"),
                              format_func=lambda n: label(n, bat_team_of.get(n) if bat_team == ALL else None))

        bowl_team = st.selectbox("Bowling team", team_opts,
                                 index=team_opts.index("Mumbai Indians") if "Mumbai Indians" in team_opts else 0)
        bowl_pool = bowl_r if bowl_team == ALL else bowl_r[(bowl_r.team == bowl_team) & (bowl_r.last_season == latest)]
        bowl_pool = bowl_pool[bowl_pool.balls >= 20].sort_values("balls", ascending=False)
        bowler_names = [AVG] + bowl_pool.bowler.tolist()
        bowl_team_of = dict(zip(bowl_r.bowler, bowl_r.team))
        bowler = st.selectbox("🎳 Bowler (type to search)", bowler_names,
                              index=default_index(bowler_names, "Bumrah"),
                              format_func=lambda n: label(n, bowl_team_of.get(n) if bowl_team == ALL else None))
        st.caption(f"Team lists show each franchise's {latest} squad. Names follow Cricsheet style "
                   "(initials + surname, e.g. *V Suryavanshi*, *JJ Bumrah*); you can type either the full name or the surname.")

    def bat_stats(name):
        if name == AVG:
            return league["bat_sr"], league["bat_out_rate"], None
        r = bat_r[bat_r.batter == name].iloc[0]
        return r.bat_sr, r.bat_out_rate, r

    def bowl_stats(name):
        if name == AVG:
            return league["bowl_econ"], league["bowl_wkt_rate"], None
        r = bowl_r[bowl_r.bowler == name].iloc[0]
        return r.bowl_econ, r.bowl_wkt_rate, r

    sr, out_rate, brow = bat_stats(batter)
    econ, wkt_rate, wrow = bowl_stats(bowler)
    need = target - score
    left_balls = 120 - (overs * 6 + balls)
    wl = 10 - wkts

    with right:
        if need <= 0:
            st.success("🎉 Target already reached!")
        else:
            p_sit = predict_win_prob(model, [need], [left_balls], [wl])[0]
            p = predict_player_win_prob(player_model, [need], [left_balls], [wl], sr, out_rate, econ, wkt_rate)[0]
            col = PITCH if p >= 0.5 else BOWL_ORANGE
            a, b = st.columns([1.1, 1])
            with a:
                st.markdown(f"<div class='scoreline'>Chasing team win chance</div>"
                            f"<div class='bigprob' style='color:{col}'>{p:.0%}</div>"
                            f"<div class='scoreline'>Need <b>{need}</b> off <b>{left_balls}</b> balls · "
                            f"{wl} wickets left · RRR <b>{need * 6 / max(left_balls, 1):.2f}</b></div>",
                            unsafe_allow_html=True)
                st.caption(f"Situation only (average players): {p_sit:.0%} · players change it by "
                           f"{(p - p_sit) * 100:+.1f} points")
            with b:
                bp = (f"Strike rate <b>{brow.bat_sr:.0f}</b> · out every <b>{100 / brow.bat_out_rate:.0f}</b> balls"
                      f"<br>{int(brow.balls):,} IPL balls faced · {brow.team}" if brow is not None
                      else "League-average batter")
                wp = (f"Economy <b>{wrow.bowl_econ:.2f}</b> · wicket every <b>{100 / wrow.bowl_wkt_rate:.0f}</b> balls"
                      f"<br>{int(wrow.balls):,} IPL balls bowled · {wrow.team}" if wrow is not None
                      else "League-average bowler")
                bcol = team_color(brow.team) if brow is not None else BAT_BLUE
                wcol = team_color(wrow.team) if wrow is not None else BOWL_ORANGE
                st.markdown(f"<div class='card' style='background:linear-gradient(135deg,#1C3F95,#122A66);border-left:8px solid {bcol}'>"
                            f"<div class='who'>🏏 {label(batter)}</div><div class='stat'>{bp}</div></div>"
                            f"<div class='card' style='background:linear-gradient(135deg,#B34A0C,#6E2A06);border-left:8px solid {wcol}'>"
                            f"<div class='who'>🎳 {label(bowler)}</div><div class='stat'>{wp}</div></div>",
                            unsafe_allow_html=True)

            st.markdown("#### ⚡ What the next ball is worth")
            outs_ = {"Wicket": (need, left_balls - 1, wl - 1), "Dot": (need, left_balls - 1, wl),
                     "Single": (need - 1, left_balls - 1, wl), "Four": (need - 4, left_balls - 1, wl),
                     "Six": (need - 6, left_balls - 1, wl)}
            after = predict_player_win_prob(player_model, *zip(*outs_.values()), sr, out_rate, econ, wkt_rate)
            delta = after - p
            fig = go.Figure(go.Bar(x=list(outs_), y=delta,
                                   marker_color=[BOWL_ORANGE if d < 0 else PITCH for d in delta],
                                   text=[f"{d:+.1%}" for d in delta], textposition="outside",
                                   hovertemplate="%{x}: %{y:+.1%}<extra></extra>"))
            st.plotly_chart(style(fig, 280, "Change in win chance", pct=True, hover="closest"), width="stretch")

    st.divider()
    st.markdown("### 🧢 Captain's corner: best options from each squad")
    st.caption("Both lists use the exact match situation above. Bars are coloured by team. "
               "Players with similar records can show the same value because the model groups similar situations.")
    teams = [t for t in CURRENT_TEAMS if t in set(bowl_r.team)]

    def squad_chart(names, probs, color, xtitle):
        fig = go.Figure(go.Bar(x=probs, y=[label(n) for n in names], orientation="h", marker_color=color,
                               text=[f"{v:.0%}" for v in probs], textposition="outside",
                               hovertemplate="%{y}<br>Chasing team win chance %{x:.1%}<extra></extra>"))
        fig = style(fig, 70 + 36 * len(names), hover="closest")
        fig.update_xaxes(tickformat=".0%", title=xtitle, showgrid=True, gridcolor=GRID, range=[0, 1.12])
        return fig

    col_bat, col_bowl = st.columns(2, gap="large")
    with col_bat:
        st.markdown("#### 🏏 Who should bat next?")
        st.caption(f"For the **batting** captain: each batter's win chance against **{label(bowler)}**. Higher is better.")
        bat_default = bat_team if bat_team != ALL else (brow.team if brow is not None and brow.team in teams else teams[0])
        bpick = st.selectbox("Batting team (current squad)", teams,
                             index=teams.index(bat_default) if bat_default in teams else 0, key="bat_squad")
        bsquad = bat_r[(bat_r.team == bpick) & (bat_r.last_season == latest) & (bat_r.balls >= 20)]
        if need > 0 and len(bsquad):
            bp_ = predict_player_win_prob(player_model, [need] * len(bsquad), [left_balls] * len(bsquad),
                                          [wl] * len(bsquad), bsquad.bat_sr.values, bsquad.bat_out_rate.values,
                                          econ, wkt_rate)
            bsquad = bsquad.assign(win_chance=bp_).sort_values("win_chance", ascending=False).head(8).iloc[::-1]
            st.plotly_chart(squad_chart(bsquad.batter, bsquad.win_chance, team_color(bpick),
                                        "Chasing team win chance with this batter"), width="stretch")
        else:
            st.info("No batters with enough IPL balls for this team yet.")

    with col_bowl:
        st.markdown("#### 🎳 Who should bowl the next over?")
        st.caption(f"For the **fielding** captain: each bowler's effect against **{label(batter)}**. Lower is better.")
        bowl_default = bowl_team if bowl_team != ALL else (wrow.team if wrow is not None and wrow.team in teams else teams[0])
        pick = st.selectbox("Bowling team (current squad)", teams,
                            index=teams.index(bowl_default) if bowl_default in teams else 0, key="bowl_squad")
        pool = bowl_r[(bowl_r.team == pick) & (bowl_r.last_season == latest) & (bowl_r.balls >= 20)]
        if need > 0 and len(pool):
            probs = predict_player_win_prob(player_model, [need] * len(pool), [left_balls] * len(pool),
                                            [wl] * len(pool), sr, out_rate, pool.bowl_econ.values,
                                            pool.bowl_wkt_rate.values)
            pool = pool.assign(win_chance=probs).sort_values("win_chance").head(8).iloc[::-1]
            st.plotly_chart(squad_chart(pool.bowler, pool.win_chance, team_color(pick),
                                        "Chasing team win chance with this bowler"), width="stretch")
        else:
            st.info("No bowlers with enough IPL balls for this team yet.")

# ---------------------------------------------------------------- match replay
with tab_replay:
    meta = (states[states.ball_seq == 0][["match_id", "date", "season", "batting_team", "bowling_team",
                                          "target_runs", "winner", "venue"]].sort_values("date", ascending=False))
    a, b = st.columns([1, 3])
    s = a.selectbox("Season", sorted(meta.season.unique(), reverse=True))
    sm = meta[meta.season == s]
    labels = {r.match_id: f"{r.date:%b %d} · {r.batting_team} chasing {int(r.target_runs)} vs {r.bowling_team}"
              for r in sm.itertuples()}
    mid = b.selectbox("Match", list(labels), format_func=labels.get)
    g = states[states.match_id == mid]
    info = sm[sm.match_id == mid].iloc[0]
    st.markdown(f"🏟️ {info.venue} · 🏆 Winner: **{info.winner}**")

    chase_c, bowl_c = team_color(info.batting_team), team_color(info.bowling_team)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.ball_seq, y=[1] * len(g), mode="lines", line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g.ball_seq, y=g.win_prob, fill="tonexty", fillcolor="rgba(255,255,255,0.04)",
                             name=f"{info.batting_team} (chasing)", mode="lines", line=dict(color=chase_c, width=3),
                             customdata=g[["runs_needed", "balls_left", "wickets_left"]],
                             hovertemplate="Need %{customdata[0]:.0f} off %{customdata[1]:.0f}, "
                                           "%{customdata[2]:.0f} wkts left<br>Win chance %{y:.0%}<extra></extra>"))
    w = g[g.wickets_on_ball > 0]
    fig.add_trace(go.Scatter(x=w.ball_seq, y=w.win_prob, name=f"Wicket for {info.bowling_team}", mode="markers",
                             marker=dict(color=bowl_c, size=12, symbol="circle", line=dict(width=2, color="white")),
                             text=w.player_out, hovertemplate="🏏 OUT: %{text}<extra></extra>"))
    fig.add_hline(y=0.5, line=dict(color="rgba(255,255,255,.4)", width=1, dash="dot"))
    fig = style(fig, 440, "Chasing team win chance", pct=True)
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(title="Delivery in chase")
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### 🔥 Biggest swing moments")
    top = g.dropna(subset=["wp_delta"])
    top = top.reindex(top.wp_delta.abs().sort_values(ascending=False).index).head(6)
    show = pd.DataFrame({
        "Over": (top.over + 1).astype(int), "Batter": top.batter, "Bowler": top.bowler,
        "Event": [f"❌ WICKET ({r.player_out})" if r.wickets_on_ball else
                  ("🟢 SIX" if r.runs_batter == 6 else "🔵 FOUR" if r.runs_batter == 4 else
                   "Dot ball" if r.runs_total == 0 else f"{int(r.runs_total)} run(s)") for r in top.itertuples()],
        "Win chance swing": top.wp_delta.map(lambda v: f"{v:+.1%}"),
    })
    st.dataframe(show, hide_index=True, width="stretch")

# ---------------------------------------------------------------- clutch leaderboard
with tab_clutch:
    st.markdown(
        "**Win Probability Added (WPA)** credits every ball's change in win chance to the batter (+) and the "
        "bowler (−). A six when 30 are needed off 12 is worth far more than a six at 50/0. WPA captures that, "
        "and strike rate and economy don't. Measured in *wins added*.")
    seasons = sorted(states.season.unique())
    a, b = st.columns(2)
    lo, hi = a.select_slider("Seasons", options=seasons, value=(seasons[0], seasons[-1]))
    min_balls = b.slider("Minimum balls in chases", 20, 500, 100, step=20)
    d = states[(states.ball_seq > 0) & states.season.between(lo, hi)].sort_values(["date", "match_id", "ball_seq"])

    bat = d.groupby("batter").agg(WPA=("wp_delta", "sum"), Runs=("runs_batter", "sum"),
                                  Balls=("is_legal", "sum"), Chases=("match_id", "nunique"),
                                  Team=("batting_team", "last")).reset_index()
    bat = bat[bat.Balls >= min_balls]
    bat["Strike rate"] = (bat.Runs * 100 / bat.Balls).round(1)
    bowl = d.groupby("bowler").agg(WPA=("wp_delta", lambda v: -v.sum()), Runs=("runs_total", "sum"),
                                   Balls=("is_legal", "sum"), Chases=("match_id", "nunique"),
                                   Team=("bowling_team", "last")).reset_index()
    bowl = bowl[bowl.Balls >= min_balls]
    bowl["Economy"] = (bowl.Runs * 6 / bowl.Balls).round(2)

    def leader_chart(df, name_col):
        top = df.nlargest(15, "WPA").sort_values("WPA")
        fig = go.Figure(go.Bar(x=top.WPA, y=[label(n) for n in top[name_col]], orientation="h",
                               marker_color=[team_color(t) for t in top.Team], text=top.Team,
                               hovertemplate="%{y} (%{text}): %{x:+.2f} wins added<extra></extra>",
                               textposition="none"))
        fig = style(fig, 500, hover="closest")
        fig.update_xaxes(title="Wins added (bar colour = latest IPL team)", gridcolor=GRID, showgrid=True)
        return fig

    chips = "".join(f"<span style='display:inline-block;margin:3px 6px 3px 0;padding:3px 10px;border-radius:999px;"
                    f"background:rgba(255,255,255,.06);font-size:.85rem'><span style='display:inline-block;width:11px;"
                    f"height:11px;border-radius:50%;background:{team_color(t)};margin-right:6px'></span>{t}</span>"
                    for t in CURRENT_TEAMS)
    st.markdown(f"<div style='margin:4px 0 8px 0'><b>Team colours:</b> {chips}</div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.subheader("🏏 Clutch chasers")
        st.plotly_chart(leader_chart(bat, "batter"), width="stretch")
        st.dataframe(bat.sort_values("WPA", ascending=False).round(2).rename(columns={"batter": "Batter"}),
                     hide_index=True, width="stretch", height=250)
    with b:
        st.subheader("🎳 Chase stoppers")
        st.plotly_chart(leader_chart(bowl, "bowler"), width="stretch")
        st.dataframe(bowl.sort_values("WPA", ascending=False).round(2).rename(columns={"bowler": "Bowler"}),
                     hide_index=True, width="stretch", height=250)

# ---------------------------------------------------------------- player profile
with tab_player:
    role = st.radio("Role", ["🏏 Batter", "🎳 Bowler"], horizontal=True)
    dl = deliveries.assign(phase=phase_of_over(deliveries.over))
    if role.endswith("Batter"):
        counts = dl.groupby("batter").size().sort_values(ascending=False)
        player = st.selectbox("Player (type to search)", counts.index)
        p_df = dl[dl.batter == player]
        g = p_df.groupby("phase", observed=False).agg(
            Runs=("runs_batter", "sum"), Balls=("is_legal", "sum"),
            Fours=("runs_batter", lambda v: (v == 4).sum()), Sixes=("runs_batter", lambda v: (v == 6).sum()))
        outs = dl[dl.player_out == player].groupby("phase", observed=False).size()
        g["Outs"] = outs.reindex(g.index).fillna(0).astype(int)
        g["Strike rate"] = (g.Runs * 100 / g.Balls.replace(0, pd.NA)).astype(float).round(1)
        g["Average"] = (g.Runs / g.Outs.replace(0, pd.NA)).astype(float).round(1)
        metric, color, opp_col, opp_label = "Strike rate", BAT_BLUE, "bowler", "Bowler"
    else:
        counts = dl.groupby("bowler").size().sort_values(ascending=False)
        player = st.selectbox("Player (type to search)", counts.index)
        p_df = dl[dl.bowler == player]
        credited = p_df.wicket_kind.notna() & ~p_df.wicket_kind.isin(["run out", "retired hurt", "retired out",
                                                                       "obstructing the field"])
        p_df = p_df.assign(bowler_wkt=credited.astype(int))
        g = p_df.groupby("phase", observed=False).agg(
            Runs=("runs_total", "sum"), Balls=("is_legal", "sum"), Wickets=("bowler_wkt", "sum"),
            Dots=("runs_total", lambda v: (v == 0).sum()))
        g["Economy"] = (g.Runs * 6 / g.Balls.replace(0, pd.NA)).astype(float).round(2)
        g["Dot %"] = (g.Dots * 100 / g.Balls.replace(0, pd.NA)).astype(float).round(1)
        metric, color, opp_col, opp_label = "Economy", BOWL_ORANGE, "batter", "Batter"

    a, b = st.columns(2)
    with a:
        st.markdown(f"**{metric} by phase**")
        fig = go.Figure(go.Bar(x=g.index.astype(str), y=g[metric],
                               marker_color=[PITCH, GOLD, color] if len(g) == 3 else color,
                               text=g[metric], textposition="outside", hovertemplate="%{x}: %{y}<extra></extra>"))
        st.plotly_chart(style(fig, 320, hover="closest"), width="stretch")
    with b:
        st.markdown("**Phase breakdown**")
        st.dataframe(g.reset_index().rename(columns={"phase": "Phase"}), hide_index=True, width="stretch")

    st.markdown(f"**⚔️ Head-to-head: {player} vs. {opp_label.lower()}s** (min. 12 balls)")
    if opp_label == "Bowler":
        h = p_df.groupby(opp_col).agg(Balls=("is_legal", "sum"), Runs=("runs_batter", "sum"),
                                      Dismissals=("player_out", lambda v: (v == player).sum()))
    else:
        h = p_df.groupby(opp_col).agg(Balls=("is_legal", "sum"), Runs=("runs_total", "sum"),
                                      Dismissals=("bowler_wkt", "sum"))
    h = h[h.Balls >= 12]
    h["Strike rate"] = (h.Runs * 100 / h.Balls).round(1)
    st.dataframe(h.sort_values("Balls", ascending=False).reset_index().rename(columns={opp_col: opp_label}),
                 hide_index=True, width="stretch", height=300)

# ---------------------------------------------------------------- the model
with tab_model:
    pm = metrics["player_model"]
    st.markdown(f"""
### 🧠 How the predictions work
**Algorithm:** gradient-boosted decision trees (scikit-learn `HistGradientBoostingClassifier`) with
**monotonic constraints**, so the model always respects cricket logic (more wickets in hand can never lower a
team's chances, and a more expensive bowler can never hurt the chasing side).

**Two models:**
1. **Situation model:** runs needed, balls left, wickets left, required run rate. Used for match replays and WPA.
2. **Player-aware model:** the situation **plus** the striker's strike rate and dismissal rate and the bowler's
   economy and wicket rate. Player ratings use **only seasons before** the match being predicted, so the model never
   sees the future, and they are shrunk toward the league average so players with few balls don't look extreme.

**Validation:** trained on {metrics['train_seasons']}, tested on unseen {metrics['test_seasons']} seasons
({metrics['test_matches']} chases, {metrics['test_balls']:,} balls).
""")
    comp = pd.DataFrame({
        "Model": ["Baseline (always predict average)", "Situation model", "Player-aware model"],
        "Brier score (lower = better)": [f"{metrics['brier_baseline']:.3f}", f"{metrics['brier']:.3f}", f"{pm['brier']:.3f}"],
        "Accuracy": ["–", f"{metrics['accuracy']:.1%}", f"{pm['accuracy']:.1%}"],
        "Accuracy, last 5 overs": ["–", f"{metrics['accuracy_last_5_overs']:.1%}", f"{pm['accuracy_last_5_overs']:.1%}"],
        "Log loss": ["–", f"{metrics['log_loss']:.3f}", f"{pm['log_loss']:.3f}"],
    })
    st.dataframe(comp, hide_index=True, width="stretch")

st.divider()
st.caption("Data: ball-by-ball IPL data from [Cricsheet](https://cricsheet.org) (Open Data Commons Attribution "
           "License). Team colours are used only to identify teams. Built with Python, scikit-learn, Plotly and "
           "Streamlit.")
