import pandas as pd
def suggest_early_mid_late_targets(ranked: pd.DataFrame):
    early = ranked[ranked["rank"] <= 36].sort_values(["adp_delta","value_over_replacement"], ascending=[False, False]).head(20)
    mid = ranked[(ranked["rank"] > 36) & (ranked["rank"] <= 84)].sort_values(["adp_delta","value_over_replacement"], ascending=[False, False]).head(20)
    late = ranked[ranked["rank"] > 84].sort_values(["adp_delta","value_over_replacement"], ascending=[False, False]).head(20)
    return early, mid, late
