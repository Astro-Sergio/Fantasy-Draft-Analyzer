import pandas as pd
import numpy as np

def _per_game(stats: pd.DataFrame) -> pd.DataFrame:
    df = stats.copy()
    g = df["g"].replace({0: 1})
    for c in ["rush_att","rush_yds","rush_td","targets","rec","rec_yds","rec_td"]:
        df[c+"_pg"] = df[c] / g
    return df

def _role_bump(depth: pd.DataFrame) -> pd.DataFrame:
    mult = {"starter": 1.10, "committee": 1.00, "backup": 0.80}
    d = depth.copy()
    d["role_mult"] = d["role"].map(mult).fillna(1.0)
    return d[["player","role_mult"]]

def _injury_risk(injuries: pd.DataFrame) -> pd.DataFrame:
    inj = injuries.copy()
    inj["risk"] = inj["games_missed_2yr"].clip(0, 20) / 5.0 + np.where(inj["age"] >= 30, 1.0, 0.0)
    return inj[["player","risk"]]

def _scoring_weights(scoring: str):
    if scoring == "PPR":
        ppr = 1.0
    elif scoring == "Half":
        ppr = 0.5
    else:
        ppr = 0.0
    return dict(ppr=ppr, rush_yd=0.1, rec_yd=0.1, td=6.0)

def make_projections(stats, depth, injuries, scoring, weight_usage, weight_eff, injury_penalty):
    s = _per_game(stats)
    d = _role_bump(depth)
    inj = _injury_risk(injuries)

    df = s.merge(d, on="player", how="left").merge(inj, on="player", how="left")
    df["role_mult"] = df["role_mult"].fillna(1.0)
    df["risk"] = df["risk"].fillna(0.0)

    df["usage_pg"] = df["rush_att_pg"] + df["targets_pg"]
    touches_pg = (df["rush_att_pg"] + df["rec_pg"]).replace({0: 1e-9})
    df["eff_yp_touch"] = (df["rush_yds_pg"] + df["rec_yds_pg"]) / touches_pg

    df["opp_adj_usage"] = df["usage_pg"] * df["role_mult"]
    df["base_index"] = weight_usage * df["opp_adj_usage"] + weight_eff * df["eff_yp_touch"]

    scale = df["base_index"].median() or 1.0
    mult = (df["base_index"] / scale).clip(lower=0.4, upper=1.8)

    df["proj_rush_att_pg"] = df["rush_att_pg"] * mult
    df["proj_targets_pg"] = df["targets_pg"] * mult
    df["proj_rec_pg"] = df["rec_pg"] * (mult * 1.02)
    df["proj_rush_yds_pg"] = df["rush_yds_pg"] * mult
    df["proj_rec_yds_pg"] = df["rec_yds_pg"] * mult
    df["proj_rush_td_pg"] = df["rush_td_pg"] * mult
    df["proj_rec_td_pg"] = df["rec_td_pg"] * mult

    sw = _scoring_weights(scoring)
    df["ppg"] = (
        df["proj_rec_pg"] * sw["ppr"] +
        df["proj_rush_yds_pg"] * sw["rush_yd"] +
        df["proj_rec_yds_pg"] * sw["rec_yd"] +
        (df["proj_rush_td_pg"] + df["proj_rec_td_pg"]) * sw["td"]
    )

    games = (17 - df["risk"].clip(0, 6)).clip(lower=12, upper=17)
    df["proj_pts"] = df["ppg"] * games

    var = (1.0 + df["risk"] * 0.15)
    df["floor"] = (df["proj_pts"] / var).round(1)
    df["ceiling"] = (df["proj_pts"] * var).round(1)

    return df[["player","team","pos","proj_pts","floor","ceiling","risk"]].copy()
