import os
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SAMPLE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")

def ensure_columns(df: pd.DataFrame, required: list[str], name: str) -> pd.DataFrame:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    return df

def load_csv(path: str, required: list[str], name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return ensure_columns(df, required, name)

def load_all_data(use_sample: bool = True) -> dict[str, pd.DataFrame]:
    base = SAMPLE_PATH if use_sample else DATA_PATH

    adp = load_csv(os.path.join(base, "adp.csv"), ["player","team","pos","adp"], "ADP")
    stats = load_csv(os.path.join(base, "stats_last_year.csv"),
                     ["player","team","pos","g","rush_att","rush_yds","rush_td","targets","rec","rec_yds","rec_td"],
                     "Last year stats")
    injuries = load_csv(os.path.join(base, "injuries.csv"),
                        ["player","status","games_missed_2yr","age"], "Injuries")
    depth = load_csv(os.path.join(base, "depth_chart.csv"),
                     ["player","team","pos","role"], "Depth chart")
    schedule = load_csv(os.path.join(base, "schedule.csv"),
                        ["team","opp","week","def_vs_pass","def_vs_run","playoff_week"], "Schedule")
    return {"adp": adp, "stats": stats, "injuries": injuries, "depth": depth, "schedule": schedule}
