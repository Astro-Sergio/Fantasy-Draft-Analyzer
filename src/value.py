import pandas as pd
import numpy as np
from .utils import pos_replacement_rank

def add_value_columns(df, adp, league_size, starters, scoring):
    x = df.merge(adp, on=["player","team","pos"], how="left")
    x["adp"] = x["adp"].fillna(999.0)
    x = x.sort_values("proj_pts", ascending=False).reset_index(drop=True)
    x["rank"] = np.arange(1, len(x) + 1)
    x["adp_delta"] = (x["adp"] - x["rank"]).round(1)

    vor_parts = []
    for pos, grp in x.groupby("pos", group_keys=False):
        repl_rank = pos_replacement_rank(pos, league_size)
        repl_pts = grp.sort_values("proj_pts", ascending=False)["proj_pts"].iloc[min(repl_rank-1, len(grp)-1)]
        vor_parts.append(pd.Series((grp["proj_pts"] - repl_pts).values, index=grp.index))
    x["value_over_replacement"] = pd.concat(vor_parts).sort_index().round(1)
    x["risk"] = pd.cut(x["risk"], bins=[-1,1.0,3.0,100], labels=["Low","Med","High"])
    return x

def make_tiers(ranked: pd.DataFrame, gap: float = 25.0) -> pd.DataFrame:
    out = []
    for pos, grp in ranked.groupby("pos"):
        grp = grp.sort_values("proj_pts", ascending=False).copy()
        tier, prev, tiers = 1, None, []
        for pts in grp["proj_pts"]:
            if prev is None: tiers.append(tier); prev = pts; continue
            if prev - pts > gap: tier += 1; prev = pts
            tiers.append(tier)
        grp["tier"] = tiers
        out.append(grp)
    return pd.concat(out)
