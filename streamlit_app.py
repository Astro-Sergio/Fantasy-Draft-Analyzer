import io
from math import ceil
from typing import Dict, List, Tuple

import certifi
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="Fantasy Draft Analyzer", page_icon="🏈", layout="wide")
st.markdown("<style>.block-container{padding-top:0.8rem}</style>", unsafe_allow_html=True)

FANTASY_POS = {"QB", "RB", "WR", "TE", "K", "DST"}

# -------------------- small utils --------------------
def safe_get(url: str, *, as_json=False, timeout=30) -> Tuple[bool, object, str]:
    try:
        r = requests.get(
            url,
            timeout=timeout,
            verify=certifi.where(),
            headers={"User-Agent": "ff-draft-intel/1.0"},
        )
        if r.status_code != 200:
            return False, None, f"{r.status_code} {r.reason}"
        return True, (r.json() if as_json else r.content), ""
    except Exception as e:
        return False, None, str(e)

def normalize_name_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace(r"[^a-z\s'\.-]", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.str.replace(r"\b([a-z])\.\b", r"\1", regex=True)
    return s

def choose(cols: List[str], options: List[str]) -> str | None:
    for c in options:
        if c in cols:
            return c
    return None

def clamp_sizes(x: pd.Series, *, min_size=6, max_size=28) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce").fillna(1.0)
    s = s.clip(lower=0.1)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo <= 1e-9:
        return pd.Series(np.full(len(s), (min_size + max_size) / 2, dtype=float), index=s.index)
    scaled = (s - lo) / (hi - lo)
    return (min_size + scaled * (max_size - min_size)).round(1)

def nzmax(s: pd.Series) -> float:
    m = pd.to_numeric(s, errors="coerce").fillna(0).max()
    return float(m) if m > 0 else 1.0

def minmax01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    rng = float(s.max() - s.min())
    if rng <= 1e-9: 
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / rng

def round_number(adp: float, league_size: int) -> int:
    if pd.isna(adp) or adp >= 999:
        return 99
    return int(ceil(float(adp) / float(league_size)))

# -------------------- sidebar --------------------
with st.sidebar:
    st.header("Settings")
    season_year = st.selectbox("Season", list(range(2025, 2018, -1)), index=0)
    scoring = st.selectbox("Scoring", ["PPR", "Half", "Standard"], index=0)

    st.subheader("League")
    league_size = st.number_input("Teams", 8, 16, 12)
    starters_qb = st.number_input("QB starters", 1, 2, 1)
    starters_rb = st.number_input("RB starters", 1, 4, 2)
    starters_wr = st.number_input("WR starters", 1, 5, 3)
    starters_te = st.number_input("TE starters", 1, 2, 1)
    starters_k = st.number_input("K starters", 0, 2, 0)
    starters_dst = st.number_input("DST starters", 0, 2, 0)

    st.caption("VOR uses the starters above to set replacement level.")

    st.subheader("ADP / Projections (optional uploads)")
    adp_file = st.file_uploader("ADP CSV (Player, Pos, Team, adp)", type=["csv"])
    proj_file = st.file_uploader("Projections CSV (Player, Pos, Team, proj_fp)", type=["csv"])

# -------------------- data loading --------------------
@st.cache_data(ttl=24*3600)
def load_weekly(season: int) -> Tuple[pd.DataFrame, int, str]:
    base = "https://github.com/nflverse/nflverse-data/releases/download/player_stats"
    tried = []
    for yr in (season, season-1):
        url = f"{base}/stats_player_week_{yr}.csv"
        ok, content, err = safe_get(url)
        if ok:
            df = pd.read_csv(io.BytesIO(content), low_memory=False)
            msg = "" if yr == season else f"{season} not published yet — using {yr}."
            return df, yr, msg
        tried.append((yr, err))
    raise RuntimeError(f"Weekly fetch failed. Tried: {tried}")

@st.cache_data(ttl=6*3600)
def load_schedule(season: int) -> pd.DataFrame:
    # Try several common nflverse paths
    urls = [
        f"https://raw.githubusercontent.com/nflverse/nflfastR-data/master/schedules/sched_{season}.csv",
        f"https://raw.githubusercontent.com/nflverse/nflfastR-data/master/schedules/schedule_{season}.csv",
        f"https://raw.githubusercontent.com/nflverse/nflfastR-data/master/schedules/schedules_{season}.csv",
    ]
    for u in urls:
        ok, content, err = safe_get(u)
        if ok:
            try:
                df = pd.read_csv(io.BytesIO(content))
                # Standardize columns commonly present: week, home_team, away_team
                cols = df.columns
                wk = choose(cols, ["week", "game_week"])
                ht = choose(cols, ["home_team", "home"])
                at = choose(cols, ["away_team", "away"])
                if not all([wk, ht, at]):
                    continue
                df = df.rename(columns={wk:"week", ht:"home_team", at:"away_team"})
                df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)
                return df[["week","home_team","away_team"]]
            except Exception:
                continue
    return pd.DataFrame(columns=["week","home_team","away_team"])

@st.cache_data(ttl=3*3600)
def load_adp_ffc(year: int, scoring: str, teams: int) -> pd.DataFrame:
    path_map = {"PPR": "ppr", "Half": "half-ppr", "Standard": "standard"}
    path = path_map.get(scoring, "ppr")
    url = f"https://fantasyfootballcalculator.com/api/v1/adp/{path}?teams={teams}&year={year}"
    ok, data, err = safe_get(url, as_json=True)
    if not ok or not isinstance(data, dict) or "players" not in data:
        raise RuntimeError(err or "FFC returned no players")
    rows = []
    for p in data["players"]:
        rows.append({
            "Player": p.get("name"),
            "Pos": (p.get("position") or "").upper(),
            "Team": (p.get("team") or "").upper(),
            "adp": pd.to_numeric(p.get("adp"), errors="coerce")
        })
    adp = pd.DataFrame(rows).dropna(subset=["Player","Pos","adp"])
    adp["Pos"] = adp["Pos"].str.replace("DEF","DST")
    adp = adp[adp["Pos"].isin(FANTASY_POS)].copy()
    adp["name_key"] = normalize_name_series(adp["Player"])
    return adp

# -------------------- season totals & context --------------------
@st.cache_data(ttl=24*3600)
def build_season_totals(weekly: pd.DataFrame, used_season: int, scoring: str) -> pd.DataFrame:
    weekly = weekly.copy()
    cols = weekly.columns
    name_col = choose(cols, ["player_display_name","player_name"]) or "__name__"
    if name_col == "__name__":
        weekly["__name__"] = weekly.get("player_id", pd.Series(range(len(weekly)))).astype(str)
    team_col = choose(cols, ["recent_team","team","posteam","player_team"]) or "__team__"
    if team_col == "__team__":
        weekly["__team__"] = "UNK"
    pos_col = choose(cols, ["position","pos","position_group"]) or "__pos__"
    if pos_col == "__pos__":
        weekly["__pos__"] = "UNK"
    week_col = choose(cols, ["week"])

    sum_cols = [
        "passing_yards","passing_tds","interceptions",
        "rushing_yards","rushing_tds",
        "receptions","receiving_yards","receiving_tds",
        "targets","fantasy_points","fantasy_points_ppr"
    ]
    use_sums = {c:"sum" for c in sum_cols if c in cols}

    g = weekly.groupby([name_col, team_col, pos_col], dropna=False).agg(use_sums).reset_index()

    if week_col and week_col in cols:
        games = weekly.dropna(subset=[name_col]).groupby([name_col])[week_col].nunique().rename("Games")
        g = g.merge(games, left_on=name_col, right_index=True, how="left")
    else:
        g["Games"] = np.nan

    def s_or0(series):
        if isinstance(series, (int, float, np.floating)) or series is None:
            return pd.Series(np.zeros(len(g)))
        return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)

    pass_y = s_or0(g.get("passing_yards")); pass_td = s_or0(g.get("passing_tds")); ints = s_or0(g.get("interceptions"))
    rush_y = s_or0(g.get("rushing_yards")); rush_td = s_or0(g.get("rushing_tds"))
    recs   = s_or0(g.get("receptions"));   rec_y  = s_or0(g.get("receiving_yards")); rec_td = s_or0(g.get("receiving_tds"))
    tgts   = s_or0(g.get("targets"))

    nfl_ppr_col = "fantasy_points_ppr" if "fantasy_points_ppr" in g.columns else ("fantasy_points" if "fantasy_points" in g.columns else None)
    nfl_ppr = s_or0(g.get(nfl_ppr_col)) if nfl_ppr_col else pd.Series(np.zeros(len(g)))

    if scoring == "PPR":
        FP = nfl_ppr
    else:
        FP = 0.04*pass_y + 4*pass_td - 2*ints + 0.1*rush_y + 6*rush_td + 0.1*rec_y + 6*rec_td
        if scoring == "Half":
            FP = FP + 0.5*recs

    out = pd.DataFrame({
        "Player": g[name_col].astype(str),
        "Team": g[team_col].astype(str).str.upper(),
        "Pos": g[pos_col].astype(str).str.upper(),
        "FP": pd.to_numeric(FP, errors="coerce").fillna(0.0).round(1),
        "Games": pd.to_numeric(g["Games"], errors="coerce").fillna(0).astype(float)
    })
    out["PPG"] = (out["FP"] / out["Games"].replace(0, np.nan)).fillna(0.0).round(2)

    # attach components
    def attach(col, series):
        if isinstance(series, (int, float, np.floating)):
            v = pd.Series(np.zeros(len(out)))
        else:
            v = pd.to_numeric(series, errors="coerce").fillna(0.0)
        out[col] = v.values

    attach("Targets", tgts); attach("Receptions", recs)
    attach("Rush_Yds", rush_y); attach("Rush_TDs", rush_td)
    attach("Rec_Yds", rec_y);   attach("Rec_TDs", rec_td)
    attach("Pass_Yds", pass_y); attach("Pass_TDs", pass_td); attach("INTs", ints)

    out = out[out["Pos"].isin(FANTASY_POS)].copy()

    # team environment
    team_base = [c for c in ["passing_yards","passing_tds","rushing_yards","rushing_tds"] if c in weekly.columns]
    if team_base:
        t = (
            weekly.groupby([team_col], dropna=False)[team_base]
            .sum().reset_index().rename(columns={team_col:"Team"}).fillna(0.0)
        )
    else:
        t = pd.DataFrame({"Team": out["Team"].unique()})
        for c in ["passing_yards","passing_tds","rushing_yards","rushing_tds"]:
            t[c] = 0.0

    # simple pace proxy: offensive plays ~= attempts + carries + targets (best effort)
    comp_cols = []
    for c in ["attempts","carries","targets"]:
        if c in weekly.columns: comp_cols.append(c)
    if comp_cols:
        pace = weekly.groupby([team_col], dropna=False)[comp_cols].sum().sum(axis=1).reset_index().rename(columns={0:"off_plays", team_col:"Team"})
        t = t.merge(pace, on="Team", how="left")
    else:
        t["off_plays"] = 0.0

    t["team_pass_factor"]     = minmax01(t.get("passing_yards", 0.0))
    t["team_pass_td_factor"]  = minmax01(t.get("passing_tds", 0.0))
    t["team_rush_factor"]     = minmax01(t.get("rushing_yards", 0.0))
    t["team_rush_td_factor"]  = minmax01(t.get("rushing_tds", 0.0))
    t["team_pace_factor"]     = minmax01(t.get("off_plays", 0.0))

    out = out.merge(
        t[["Team","team_pass_factor","team_pass_td_factor","team_rush_factor","team_rush_td_factor","team_pace_factor"]],
        on="Team", how="left"
    ).fillna(0.0)

    # Opp/Env weighted opportunity
    carries_proxy = (out["Rush_Yds"] / 4.0).clip(lower=0)
    tgt_comp = 100.0 * out["Targets"] / nzmax(out["Targets"])
    car_comp = 100.0 * carries_proxy / nzmax(carries_proxy)
    rz_comp  = 50.0 * (out["team_pass_td_factor"] + out["team_rush_td_factor"])
    env_comp = 40.0 * (out["team_pass_factor"] + out["team_rush_factor"]) + 20.0 * out["team_pace_factor"]
    out["Opportunity_Score"] = (0.5*(0.6*tgt_comp + 0.4*car_comp) + 0.3*rz_comp + 0.2*env_comp).round(1)

    # base projection + context
    pass_weight = np.where(out["Pos"].isin(["WR","TE","QB"]), 0.15*(out["team_pass_factor"]+out["team_pass_td_factor"]), 0.05)
    run_weight  = np.where(out["Pos"].isin(["RB","QB"]),    0.15*(out["team_rush_factor"]+out["team_rush_td_factor"]), 0.05)
    pace_weight = 0.05 * out["team_pace_factor"]
    context = 1.0 + pass_weight + run_weight + pace_weight
    out["Projected"] = (out["FP"] * context).round(1)

    out["name_key"] = normalize_name_series(out["Player"])
    return out.reset_index(drop=True)

def apply_adp(board: pd.DataFrame, adp: pd.DataFrame) -> pd.DataFrame:
    out = board.copy()
    if adp is None or adp.empty:
        out["adp"] = 999.0; out["adp_delta"] = 0.0
        return out
    out = out.merge(adp[["name_key","adp"]], on="name_key", how="left")
    out["adp"] = pd.to_numeric(out["adp"], errors="coerce").fillna(999.0).clip(lower=1)
    out["adp_pos"] = out.sort_values(["Pos","FP"], ascending=[True,False]).groupby("Pos").cumcount()+1
    out["adp_delta"] = (out["adp_pos"] - out["adp"]).round(0)
    return out

def compute_vor(df: pd.DataFrame, teams: int, starters: Dict[str,int]) -> pd.DataFrame:
    out = df.copy()
    out["VOR"] = 0.0
    for pos, grp in out.groupby("Pos", group_keys=False):
        need = teams * starters.get(pos, 0)
        if need <= 0 or grp.empty: 
            continue
        base = grp.sort_values("FP", ascending=False)["FP"].reset_index(drop=True)
        idx = min(len(base)-1, max(0, need-1))
        replacement = float(base.iloc[idx])
        out.loc[out["Pos"].eq(pos), "VOR"] = (out.loc[out["Pos"].eq(pos), "FP"] - replacement).round(2)
    return out

def offense_weighting(board: pd.DataFrame) -> pd.DataFrame:
    out = board.copy()
    if not {"team_pass_factor","team_pass_td_factor","team_rush_factor","team_rush_td_factor","team_pace_factor"}.issubset(out.columns):
        out["Weighted_Projected"] = out["Projected"]; out["Weighted_Value_Score"] = 0.0
        return out
    off_pass = 0.6*out["team_pass_factor"] + 0.4*out["team_pass_td_factor"]
    off_rush = 0.6*out["team_rush_factor"] + 0.4*out["team_rush_td_factor"]
    pass_boost = np.where(out["Pos"].isin(["WR","TE","QB"]), 0.25*off_pass, 0.05*off_pass)
    rush_boost = np.where(out["Pos"].isin(["RB","QB"]),    0.25*off_rush, 0.05*off_rush)
    pace_boost = 0.05*out["team_pace_factor"]
    context = 1.0 + pass_boost + rush_boost + pace_boost
    out["Weighted_Projected"] = (out["Projected"] * context).round(1)
    vs = (out["Weighted_Projected"] / out["adp"].replace(0,1)).replace([np.inf,-np.inf],0.0)
    out["Weighted_Value_Score"] = (100.0*vs/ (float(vs.max()) if float(vs.max())>0 else 1.0)).round(2)
    return out

# usage-based quick estimator (xFP-ish flavor)
def usage_projection(board: pd.DataFrame, scoring: str) -> pd.Series:
    b = board.copy()
    if scoring == "PPR":
        return (0.08*b["Targets"] + 0.1*b["Rec_Yds"] + 6*b["Rec_TDs"] +
                0.06*b["Rush_Yds"] + 6*b["Rush_TDs"] +
                0.04*b["Pass_Yds"] + 4*b["Pass_TDs"] - 2*b["INTs"]).fillna(0.0)
    elif scoring == "Half":
        return (0.04*b["Targets"] + 0.1*b["Rec_Yds"] + 6*b["Rec_TDs"] +
                0.06*b["Rush_Yds"] + 6*b["Rush_TDs"] +
                0.04*b["Pass_Yds"] + 4*b["Pass_TDs"] - 2*b["INTs"]).fillna(0.0)
    else:
        return (0.1*b["Rec_Yds"] + 6*b["Rec_TDs"] +
                0.06*b["Rush_Yds"] + 6*b["Rush_TDs"] +
                0.04*b["Pass_Yds"] + 4*b["Pass_TDs"] - 2*b["INTs"]).fillna(0.0)

def breakout_trends(weekly: pd.DataFrame) -> pd.DataFrame:
    cols = weekly.columns
    ncol = choose(cols, ["player_display_name","player_name"])
    pcol = choose(cols, ["position","pos","position_group"])
    wcol = choose(cols, ["week"])
    if not all([ncol,pcol,wcol]): return pd.DataFrame()
    df = weekly[[ncol,pcol,wcol,"targets","carries","wopr"]].copy()
    for c in ["targets","carries","wopr"]:
        if c not in df.columns: df[c]=0.0
    df = df.rename(columns={ncol:"Player",pcol:"Pos",wcol:"week"})
    df["Pos"]=df["Pos"].astype(str).str.upper()
    df = df[df["Pos"].isin(["RB","WR","TE"])].copy()
    df["half"]=np.where(df["week"]>=10, "H2","H1")
    agg = (df.groupby(["Player","Pos","half"]).agg(targets=("targets","sum"),
                                                  carries=("carries","sum"),
                                                  wopr=("wopr","mean"),
                                                  games=("week","nunique")).reset_index())
    w = agg.pivot(index=["Player","Pos"], columns="half", values=["targets","carries","wopr","games"]).fillna(0.0)
    w.columns=[f"{a}_{b}" for a,b in w.columns.to_flat_index()]
    w = w.reset_index()
    w["delta_targets_pg"]=(w["targets_H2"]/w["games_H2"].replace(0,np.nan) - w["targets_H1"]/w["games_H1"].replace(0,np.nan)).fillna(0.0)
    w["delta_carries_pg"]=(w["carries_H2"]/w["games_H2"].replace(0,np.nan) - w["carries_H1"]/w["games_H1"].replace(0,np.nan)).fillna(0.0)
    w["delta_wopr"]= (w["wopr_H2"]-w["wopr_H1"]).fillna(0.0)
    def norm(s):
        s = pd.to_numeric(s, errors="coerce").fillna(0.0); rng=float(s.max()-s.min())
        return (s-s.min())/rng*100.0 if rng>1e-9 else pd.Series(np.zeros(len(s)))
    w["trend_targets"]=norm(w["delta_targets_pg"]); w["trend_carries"]=norm(w["delta_carries_pg"]); w["trend_wopr"]=norm(w["delta_wopr"])
    w["Breakout_Score"] = (
        np.where(w["Pos"].isin(["WR","TE"]), 0.5*w["trend_targets"]+0.4*w["trend_wopr"], 0.2*w["trend_targets"]+0.3*w["trend_wopr"]) +
        np.where(w["Pos"].isin(["RB"]), 0.5*w["trend_carries"], 0.2*w["trend_carries"])
    )
    w["name_key"]=normalize_name_series(w["Player"])
    return w[["name_key","Player","Pos","Breakout_Score","delta_wopr","delta_targets_pg","delta_carries_pg"]]

def weekly_consistency(weekly: pd.DataFrame) -> pd.DataFrame:
    cols = weekly.columns
    ncol=choose(cols,["player_display_name","player_name"])
    pcol=choose(cols,["position","pos","position_group"])
    wcol=choose(cols,["week"])
    fcol= "fantasy_points_ppr" if "fantasy_points_ppr" in cols else ("fantasy_points" if "fantasy_points" in cols else None)
    if not all([ncol,pcol,wcol,fcol]): return pd.DataFrame()
    df = weekly[[ncol,pcol,wcol,fcol]].copy().rename(columns={ncol:"Player",pcol:"Pos",fcol:"FPw"})
    g = df.groupby(["Player","Pos"])["FPw"].agg(["mean","std","count"]).reset_index()
    out=[]
    for pos, grp in g.groupby("Pos"):
        std = pd.to_numeric(grp["std"], errors="coerce").fillna(0.0)
        score = 100*(1.0 - (std/(float(std.max()) if float(std.max())>0 else 1.0)))
        out.append(pd.DataFrame({"name_key":normalize_name_series(grp["Player"]),
                                 "Reliability":score.round(1)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def playoff_and_sos_weights(schedule: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    # opponent fantasy points allowed by position (from the season data itself)
    # Build team vs position FP allowed
    cols = weekly.columns
    pteam = choose(cols, ["recent_team","posteam","team","player_team"]) or "team"
    opp   = choose(cols, ["opponent_team","defteam","defteam_name"])
    posc  = choose(cols, ["position","pos","position_group"])
    fcol  = "fantasy_points_ppr" if "fantasy_points_ppr" in cols else ("fantasy_points" if "fantasy_points" in cols else None)
    wkcol = choose(cols, ["week"])
    if not all([pteam, opp, posc, fcol, wkcol]) or schedule.empty:
        return pd.DataFrame({"Team":[],"Pos":[],"Playoff_Ease":[],"Season_SoS":[]})
    df = weekly[[pteam, opp, posc, fcol, wkcol]].copy().rename(
        columns={pteam:"Team", opp:"opp", posc:"Pos", fcol:"FPw", wkcol:"week"}
    )
    df["Pos"]=df["Pos"].astype(str).upper()
    df = df[df["Pos"].isin(["QB","RB","WR","TE"])].copy()
    allowed = df.groupby(["opp","Pos"])["FPw"].mean().reset_index().rename(columns={"opp":"Team","FPw":"Allowed_PPG"})
    # Season SoS per team/pos: opponents' Allowed_PPG mean over schedule
    sched = schedule.rename(columns={"home_team":"home","away_team":"away"}).copy()
    sched = sched[sched["week"].between(1, 18, inclusive="both")]
    team_rows=[]
    for _, r in sched.iterrows():
        team_rows += [{"Team":r["home"],"opp":r["away"],"week":r["week"]},
                      {"Team":r["away"],"opp":r["home"],"week":r["week"]}]
    games = pd.DataFrame(team_rows)
    ease = games.merge(allowed, left_on=["opp"], right_on=["Team"], how="left").drop(columns=["Team"])
    # for each original team pos combination take mean Season SoS
    out=[]
    for pos in ["QB","RB","WR","TE"]:
        pos_ease = ease.merge(allowed[allowed["Pos"]==pos][["Team","Allowed_PPG"]]
                              .rename(columns={"Team":"opp"}), on="opp", how="left")
        grp = pos_ease[pos_ease["Pos"]==pos].groupby(["Team"])["Allowed_PPG_x"].mean().reset_index()
        grp["Pos"]=pos; grp=grp.rename(columns={"Allowed_PPG_x":"Season_SoS"})
        # playoff weeks 15-17
        po = pos_ease[(pos_ease["Pos"]==pos) & (pos_ease["week"].between(15,17))].groupby("Team")["Allowed_PPG_x"].mean().reset_index()
        po=po.rename(columns={"Allowed_PPG_x":"Playoff_Ease"})
        m = grp.merge(po, on="Team", how="left")
        out.append(m)
    ease_tbl = pd.concat(out, ignore_index=True).fillna(0.0)
    # normalize to 0-100 (higher = easier opponents)
    ease_tbl["Season_SoS"]   = (100*minmax01(ease_tbl["Season_SoS"])).round(1)
    ease_tbl["Playoff_Ease"] = (100*minmax01(ease_tbl["Playoff_Ease"])).round(1)
    return ease_tbl

def ensemble_projections(board: pd.DataFrame, proj_file: io.BytesIO | None, scoring: str) -> pd.DataFrame:
    out = board.copy()
    # components
    base = pd.to_numeric(out["Weighted_Projected"], errors="coerce").fillna(out["Projected"])
    usage = usage_projection(out, scoring)
    parts = [base, usage]
    if proj_file is not None:
        try:
            ext = pd.read_csv(proj_file)
            if {"Player","Pos","Team","proj_fp"}.issubset(ext.columns):
                ext = ext.copy()
                ext["name_key"] = normalize_name_series(ext["Player"])
                parts.append(out.merge(ext[["name_key","proj_fp"]], on="name_key", how="left")["proj_fp"].fillna(0.0))
        except Exception:
            pass
    # weights: base 0.55, usage 0.25, external 0.20 (if present)
    if len(parts)==3:
        ens = 0.55*parts[0] + 0.25*parts[1] + 0.20*parts[2]
    else:
        ens = 0.7*parts[0] + 0.3*parts[1]
    out["Ensemble_Proj"] = pd.to_numeric(ens, errors="coerce").fillna(0.0).round(1)
    # rebuild value on ensemble
    v = (out["Ensemble_Proj"] / out["adp"].replace(0,1)).replace([np.inf,-np.inf],0.0)
    out["Ensemble_Value"] = (100.0*v / (float(v.max()) if float(v.max())>0 else 1.0)).round(2)
    return out

def apply_schedule_weights(board: pd.DataFrame, ease_tbl: pd.DataFrame) -> pd.DataFrame:
    if ease_tbl.empty: 
        board["Playoff_Ease"]=0.0; board["Season_SoS"]=0.0
        return board
    out = board.merge(ease_tbl, on=["Team","Pos"], how="left").fillna(0.0)
    # Blend playoff ease into Edge later; also small multiplier to Ensemble projection
    mult = 1.0 + 0.10*(out["Playoff_Ease"]/100.0) + 0.05*(out["Season_SoS"]/100.0)
    out["Ensemble_Proj"] = (out["Ensemble_Proj"] * mult).round(1)
    return out

def compute_edge(board: pd.DataFrame) -> pd.DataFrame:
    out = board.copy()
    V = 100.0 * out["Ensemble_Value"] / nzmax(out["Ensemble_Value"])
    R = pd.to_numeric(out.get("Reliability", 0), errors="coerce").fillna(0.0)
    O = 100.0 * out["Opportunity_Score"] / nzmax(out["Opportunity_Score"])
    B = 100.0 * out["Breakout_Score"] / nzmax(out["Breakout_Score"])
    vor = 100.0 * out["VOR"].clip(lower=0) / nzmax(out["VOR"].clip(lower=0))
    PE = pd.to_numeric(out.get("Playoff_Ease",0), errors="coerce").fillna(0.0)

    out["Edge_Score"] = (0.30*V + 0.18*vor + 0.18*B + 0.14*O + 0.10*R + 0.10*(PE/100.0) ).round(2)
    return out

def sleepers_and_busts(df: pd.DataFrame, *, sleepers_n=50, busts_n=50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cand = df[df["adp"] < 999].copy()
    if cand.empty: return df.head(0), df.head(0)
    val   = 100.0 * cand["Ensemble_Value"] / nzmax(cand["Ensemble_Value"])
    opp   = 100.0 * cand["Opportunity_Score"] / nzmax(cand["Opportunity_Score"])
    vor_p = 100.0 * cand["VOR"].clip(lower=0) / nzmax(cand["VOR"].clip(lower=0))
    rel   = pd.to_numeric(cand.get("Reliability",0), errors="coerce").fillna(0.0)
    br    = 100.0 * cand["Breakout_Score"] / nzmax(cand["Breakout_Score"])

    cand["Sleeper_Score"] = (0.40*val + 0.20*br + 0.20*opp + 0.15*vor_p + 0.05*rel).round(2)
    sleepers = cand.sort_values(["Sleeper_Score","Edge_Score"], ascending=False).head(sleepers_n)[
        ["Player","Team","Pos","Ensemble_Proj","PPG","adp","VOR","Opportunity_Score","Reliability","Breakout_Score","Sleeper_Score"]
    ]

    inv_val = 100.0 - val; inv_opp = 100.0 - opp
    vor_n   = 100.0 * cand["VOR"].clip(upper=0).abs() / nzmax(cand["VOR"].clip(upper=0).abs())
    inv_rel = 100.0 - rel
    cost    = 100.0 * (1.0/cand["adp"]) / max(1e-9, float((1.0/cand["adp"]).max()))
    cand["Bust_Score"] = (0.35*inv_val + 0.25*inv_opp + 0.15*vor_n + 0.15*cost + 0.10*inv_rel).round(2)
    busts = cand.sort_values(["Bust_Score","adp"], ascending=[False,True]).head(busts_n)[
        ["Player","Team","Pos","Ensemble_Proj","PPG","adp","VOR","Opportunity_Score","Reliability","Bust_Score"]
    ]
    return sleepers, busts

def handcuff_premium(board: pd.DataFrame) -> pd.Series:
    # proxy: RBs with low ADP (later picks) on teams high in rush TD factor
    rb = board[board["Pos"].eq("RB")].copy()
    prem = (0.6*rb["team_rush_td_factor"] + 0.4*rb["team_rush_factor"]) * (1.0 - minmax01(rb["adp"].replace(999, 999.0)))
    prem = 100.0*prem / nzmax(prem)
    s = pd.Series(np.zeros(len(board)), index=board.index)
    s.loc[rb.index] = prem.values
    return s.round(1)

def round_steals(df: pd.DataFrame, league_size: int) -> pd.DataFrame:
    view = df[df["adp"] < 999].copy()
    if view.empty: return view
    view["Round"] = view["adp"].apply(lambda x: round_number(x, league_size))
    steals = (
        view.sort_values(["Round","Edge_Score","Ensemble_Value"], ascending=[True,False,False])
        .groupby("Round", as_index=False).head(5)
        [["Round","Player","Team","Pos","adp","Edge_Score","Ensemble_Proj","VOR","Breakout_Score","Reliability"]]
        .sort_values(["Round","Edge_Score"], ascending=[True,False])
    )
    return steals

def stack_finder(df: pd.DataFrame) -> pd.DataFrame:
    qbs = df[(df["Pos"]=="QB") & (df["adp"]<999)][["Team","Player","PPG","Ensemble_Proj","Edge_Score"]].rename(
        columns={"Player":"QB","PPG":"QB_PPG","Ensemble_Proj":"QB_Proj","Edge_Score":"QB_Edge"}
    )
    rec = df[df["Pos"].isin(["WR","TE"]) & (df["adp"]<999)][["Team","Player","Pos","Ensemble_Proj","Edge_Score","team_pass_factor"]].rename(
        columns={"Player":"Receiver","Ensemble_Proj":"REC_Proj","Edge_Score":"REC_Edge"}
    )
    if qbs.empty or rec.empty: return df.head(0)
    pairs = qbs.merge(rec, on="Team", how="inner")
    qb_pct = minmax01(pairs["QB_PPG"]); rec_pct = minmax01(pairs["REC_Proj"]); env = pairs.get("team_pass_factor",0)
    pairs["Stack_Score"] = (100*(0.5*qb_pct + 0.4*rec_pct + 0.1*env)).round(2)
    return pairs.sort_values("Stack_Score", ascending=False)[["Team","QB","Receiver","Pos","QB_PPG","REC_Proj","Stack_Score"]].head(15)

# -------------------- load data --------------------
try:
    weekly_raw, used_season, info_msg = load_weekly(int(season_year))
    if info_msg: st.info(info_msg)
except Exception as e:
    st.error(f"Could not load weekly stats: {e}")
    st.stop()

board = build_season_totals(weekly_raw, used_season, scoring)

# ADP (upload overrides)
adp_df = None
if adp_file is not None:
    try:
        tmp = pd.read_csv(adp_file)
        if {"Player","Pos","Team","adp"}.issubset(tmp.columns):
            tmp = tmp.copy()
            tmp["Pos"]=tmp["Pos"].astype(str).str.upper().str.replace("DEF","DST")
            tmp["name_key"]=normalize_name_series(tmp["Player"])
            adp_df = tmp[tmp["Pos"].isin(FANTASY_POS)]
        else:
            st.warning("ADP CSV must contain: Player, Pos, Team, adp.")
    except Exception as e:
        st.warning(f"Failed to read ADP CSV: {e}")

if adp_df is None:
    try:
        adp_df = load_adp_ffc(int(season_year), scoring, int(league_size))
    except Exception as e:
        st.warning(f"FFC ADP failed: {e}. ADP will be 999 for unmatched players.")

board = apply_adp(board, adp_df)

starters = {"QB":starters_qb,"RB":starters_rb,"WR":starters_wr,"TE":starters_te,"K":starters_k,"DST":starters_dst}
board = compute_vor(board, int(league_size), starters)
board = offense_weighting(board)

# Consistency / Breakouts
rel = weekly_consistency(weekly_raw)
if not rel.empty:
    board = board.merge(rel, on="name_key", how="left")
    board["Reliability"] = pd.to_numeric(board["Reliability"], errors="coerce").fillna(0.0)
else:
    board["Reliability"]=0.0

br = breakout_trends(weekly_raw)
if not br.empty:
    board = board.merge(br.drop(columns=["Player","Pos"]), on="name_key", how="left")
    board["Breakout_Score"]=pd.to_numeric(board["Breakout_Score"], errors="coerce").fillna(0.0)
else:
    board["Breakout_Score"]=0.0

# Ensemble projections
board = ensemble_projections(board, proj_file, scoring)

# Schedule (SoS + playoff boost)
sched = load_schedule(int(used_season))
ease_tbl = playoff_and_sos_weights(sched, weekly_raw)
board = apply_schedule_weights(board, ease_tbl)

# Handcuff premium (RB only)
board["Handcuff_Premium"] = handcuff_premium(board)

# Final Edge
board = compute_edge(board)

# -------------------- KPIs --------------------
k1,k2,k3,k4 = st.columns(4)
with k1: st.metric("Players in view", f"{len(board):,}")
with k2: st.metric("Avg Edge", f"{board['Edge_Score'].mean():.2f}")
with k3: st.metric("Avg VOR", f"{board['VOR'].mean():.1f}")
with k4:
    matched = int((board["adp"]<999).sum()); pct = 100*matched/len(board) if len(board) else 0
    st.metric("ADP matched", f"{pct:.0f}%")

# -------------------- Tabs --------------------
tabs = st.tabs([
    "🎯 Value Targets",
    "📊 Player Analysis",
    "🌙 Sleepers & ⚠️ Busts",
    "🚀 Breakouts",
    "⚡ Opportunity",
    "💎 Edges: Steals & Stacks",
    "🧠 Draft Room",
    "ℹ️ Legend"
])

# 1) Value Targets
with tabs[0]:
    st.header("Top Value Targets")
    filt = board[board["Pos"].isin(FANTASY_POS)].copy()
    top_values = (filt.sort_values(["Edge_Score","Ensemble_Value","VOR"], ascending=False)
                  [["Player","Team","Pos","Ensemble_Proj","PPG","VOR","adp","Ensemble_Value","Edge_Score","Playoff_Ease","Season_SoS"]]
                  .head(30))
    st.dataframe(top_values, use_container_width=True)
    show = filt[(filt["adp"]<999) & filt["Pos"].isin(["QB","RB","WR","TE"])].copy()
    if not show.empty:
        show["bubble"]=clamp_sizes(show["Ensemble_Proj"])
        fig = px.scatter(
            show, x="adp", y="Ensemble_Value", color="Pos", size="bubble",
            hover_data={"Player":True,"Team":True,"PPG":True,"VOR":True,"bubble":False},
            title="Value vs ADP (Ensemble)"
        )
        fig.update_layout(height=520, xaxis_title="ADP (earlier = better)", yaxis_title="Ensemble Value")
        st.plotly_chart(fig, use_container_width=True)

# 2) Player Analysis
with tabs[1]:
    st.header("Player Deep Dive")
    view = board[board["Pos"].isin(FANTASY_POS)].copy()
    if not view.empty:
        who = st.selectbox("Select a player", sorted(view["Player"].unique().tolist()))
        r = view[view["Player"].eq(who)].iloc[0]
        c1,c2,c3 = st.columns(3)
        with c1: st.metric("PPG", f"{r['PPG']:.2f}"); st.metric("Total FP", f"{r['FP']:.1f}")
        with c2: st.metric("ADP", "N/A" if r["adp"]>=999 else f"{r['adp']:.1f}"); st.metric("VOR", f"{r['VOR']:.2f}")
        with c3: st.metric("Edge", f"{r['Edge_Score']:.2f}"); st.metric("Playoff Ease", f"{r.get('Playoff_Ease',0):.0f}")
        st.caption("Context: pass/run/pace weighting, breakout trend, opportunity, reliability, and schedule.")

# 3) Sleepers & Busts
with tabs[2]:
    st.header("Top 50 Sleepers & Busts")
    sleepers, busts = sleepers_and_busts(board, sleepers_n=50, busts_n=50)
    a,b = st.columns(2)
    with a:
        st.subheader("🌙 Sleepers")
        st.dataframe(sleepers, use_container_width=True)
    with b:
        st.subheader("⚠️ Busts")
        st.dataframe(busts, use_container_width=True)

# 4) Breakouts
with tabs[3]:
    st.header("Breakout Candidates (trend-based)")
    bview = (board[board["Pos"].isin(["RB","WR","TE"])]
             .sort_values(["Breakout_Score","Edge_Score","Ensemble_Value"], ascending=False)
             [["Player","Team","Pos","Breakout_Score","PPG","adp","VOR","Ensemble_Proj","Reliability"]]
             .head(50))
    st.dataframe(bview, use_container_width=True)

# 5) Opportunity
with tabs[4]:
    st.header("Opportunity vs Production")
    show = board[board["Pos"].isin(["QB","RB","WR","TE"])].copy()
    if not show.empty:
        show["bubble"]=clamp_sizes(show["FP"])
        fig2 = px.scatter(show, x="Opportunity_Score", y="FP", color="Pos", size="bubble",
                          hover_data={"Player":True,"Team":True,"PPG":True,"VOR":True,"bubble":False},
                          title="Opportunity Score vs Total FP (last season)")
        fig2.update_layout(height=560, xaxis_title="Opportunity Score", yaxis_title="Total FP")
        st.plotly_chart(fig2, use_container_width=True)

# 6) Edges: Steals & Stacks
with tabs[5]:
    st.header("Round Steals & Stack Finder")
    steals = round_steals(board, int(league_size))
    if steals.empty: st.info("Need ADP to compute round steals.")
    else:
        st.subheader("💎 Steals by Round")
        st.dataframe(steals, use_container_width=True)
    st.subheader("🔗 Stack Finder (QB + WR/TE)")
    stacks = stack_finder(board)
    if stacks.empty: st.info("Need ADP-matched QBs and pass catchers.")
    else: st.dataframe(stacks, use_container_width=True)

# 7) Draft Room
with tabs[6]:
    st.header("Live Draft Room")
    if "drafted" not in st.session_state: st.session_state.drafted = set()
    if "my_roster" not in st.session_state: st.session_state.my_roster = {"QB":0,"RB":0,"WR":0,"TE":0,"K":0,"DST":0}

    st.caption("Mark drafted players (by anyone) and get need-adjusted recommendations for your next pick.")
    left, right = st.columns([2,1])
    with left:
        to_mark = st.multiselect("Mark drafted", sorted(board["Player"].tolist()))
        if to_mark:
            for p in to_mark:
                st.session_state.drafted.add(p)
        clear = st.button("Clear drafted")
        if clear: st.session_state.drafted.clear()
    with right:
        pos_add = st.selectbox("Add to **your** roster count", ["QB","RB","WR","TE","K","DST"])
        if st.button("Add 1 at position"):
            st.session_state.my_roster[pos_add] = st.session_state.my_roster.get(pos_add,0)+1
        if st.button("Reset my roster"):
            st.session_state.my_roster = {"QB":0,"RB":0,"WR":0,"TE":0,"K":0,"DST":0}
        st.write("Your roster so far:", st.session_state.my_roster)

    taken_keys = normalize_name_series(pd.Series(list(st.session_state.drafted)))
    avail = board[~board["name_key"].isin(taken_keys)].copy()

    # position need multiplier (want to fill starters first)
    needs = {"QB":starters_qb,"RB":starters_rb,"WR":starters_wr,"TE":starters_te,"K":starters_k,"DST":starters_dst}
    have  = st.session_state.my_roster
    def need_mult(pos):
        need = max(0, int(needs.get(pos,0)) - int(have.get(pos,0)))
        return 1.25 if need>0 else 1.0

    avail["Need_Adjusted_Edge"] = avail.apply(lambda r: r["Edge_Score"]*need_mult(r["Pos"]), axis=1)
    # small stack nudge if same team as a drafted QB/WR/TE on your roster? (best-effort: use my_roster counts only for now)

    st.subheader("Best Pick Now (need-adjusted)")
    best = (avail.sort_values(["Need_Adjusted_Edge","Edge_Score","Ensemble_Value"], ascending=False)
            [["Player","Team","Pos","adp","Need_Adjusted_Edge","Edge_Score","Ensemble_Proj","VOR","Reliability","Playoff_Ease"]]
            .head(20))
    st.dataframe(best, use_container_width=True)

# 8) Legend
with tabs[7]:
    st.header("Legend — how to use every tab")
    st.markdown("""
**Core metrics**
- **FP / PPG** – Last season fantasy points / per game in your chosen scoring.
- **Projected** – FP with team pass/run/pace context applied.
- **Weighted_Projected** – Extra positional weighting: WR/TE/QB ↑ in pass–heavy; RB/QB ↑ in run–heavy.
- **Ensemble_Proj** – 55% Weighted_Projected + 25% usage-based estimator + 20% uploaded projections (if provided).
- **Value_Score** – Projected ÷ ADP (scaled 0–100).
- **Ensemble_Value** – Value but using Ensemble_Proj.
- **VOR** – Value over replacement at your starter depth (set in sidebar).
- **Opportunity_Score** – Player volume (targets/carries) + red-zone + team environment (pass/run/pace).
- **Reliability** – Higher = steadier weekly scoring (lower variance within position).
- **Breakout_Score** – 2nd-half usage trend vs 1st half (targets, carries, WOPR).
- **Season_SoS / Playoff_Ease** – 0–100, higher = easier opponents (Weeks 15–17 for Playoff_Ease).
- **Handcuff_Premium (RB)** – Late-ADP RBs on run-TD rich teams; higher = better contingency value.
- **Edge_Score** – 30% Ensemble_Value + 18% VOR + 18% Breakout + 14% Opportunity + 10% Reliability + 10% Playoff_Ease.

**Tabs**
- **Value Targets** – Highest Edge and Value; bubble chart uses Ensemble_Proj.
- **Player Analysis** – One-pager with context metrics for a selected player.
- **Sleepers & Busts** – 50 picks each, scored from value/opportunity/trend/reliability/cost.
- **Breakouts** – Players with the strongest second-half trend signals.
- **Opportunity** – Why volume (and environment) drives points.
- **Edges: Steals & Stacks** – Best 5 values per round (by your league size) + top QB stacks with WR/TE.
- **Draft Room** – Mark drafted players and get **Need-Adjusted** recommendations for your next pick.

**Data**
- **Weekly stats** from nflverse (public GitHub CSV). If the current season isn’t published yet, we automatically use last season.
- **ADP** from Fantasy Football Calculator API or your uploaded CSV (999 = missing).
- **Schedules** from nflverse (public GitHub CSV); used for SoS/Playoff_Ease.
- Upload **your own projections** (Player, Pos, Team, `proj_fp`) to strengthen the ensemble.
""")