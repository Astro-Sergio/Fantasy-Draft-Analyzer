POSITION_ORDER = ["RB","WR","QB","TE","DST","K"]
POS_REPLACEMENT_RANK = {"QB": 15, "RB": 30, "WR": 36, "TE": 14, "DST": 13, "K": 13}
def pos_replacement_rank(pos: str, league_size: int) -> int:
    baseline = POS_REPLACEMENT_RANK.get(pos, 999)
    return max(1, round(baseline * league_size / 12))
