# judge.py : classical gomoku evaluator (pattern levels + immediate win/block detection).
# Acts as the external "chess strength" the flies lack: every move the fly makes is scored
# to Q in [0,1]; the server converts Q to a dopamine reward r in [-1,1].
# This is a classical heuristic engine (no GPU, no training) - honest, deterministic, fast.
import numpy as np

DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))
LEVEL_TABLE = {0: 0.05, 1: 0.18, 2: 0.38, 3: 0.62, 4: 0.86, 5: 1.0}   # pattern level -> quality

def _pattern_level(b, x, y, p, size):
    """level of the pattern created if player p places at (x,y):
       5 win, 4 open four, 3 simple four, 2 open three, 1 closed three, 0 none"""
    worst_need = 0
    levels = []
    for dx, dy in DIRS:
        n, open_ends = 1, 0
        for sgn in (1, -1):
            xx, yy = x + dx*sgn, y + dy*sgn
            while 0 <= xx < size and 0 <= yy < size and b[yy, xx] == p:
                n += 1; xx += dx*sgn; yy += dy*sgn
            if 0 <= xx < size and 0 <= yy < size and b[yy, xx] == 0:
                open_ends += 1
        if n >= 5: lv = 5
        elif n == 4: lv = 4 if open_ends >= 1 else 0
        elif n == 3: lv = 2 if open_ends == 2 else (1 if open_ends == 1 else 0)
        elif n == 2: lv = 1 if open_ends == 2 else 0
        else: lv = 0
        levels.append(lv)
    # two intersecting open threes / four+three are as strong as an open four
    if levels.count(2) >= 2: return 3
    if 4 in levels: return 4
    return max(levels) if levels else 0

class GomokuJudge:
    name = "classic-pattern-evaluator v1"
    def __init__(self, size=15):
        self.size = size
    def wins_now(self, b, x, y, p):
        for dx, dy in DIRS:
            n = 1
            for sgn in (1, -1):
                xx, yy = x + dx*sgn, y + dy*sgn
                while 0 <= xx < self.size and 0 <= yy < self.size and b[yy, xx] == p:
                    n += 1; xx += dx*sgn; yy += dy*sgn
                if n >= 5: return True
        return False
    def move_quality(self, b, x, y, p):
        """Q in [0,1]: how good is placing p at (x,y) - attack and must-block value"""
        if self.wins_now(b, x, y, p): return 1.0
        if self.wins_now(b, x, y, 3 - p): return 0.92      # forced block of an immediate loss
        lv_own = _pattern_level(b, x, y, p, self.size)
        lv_blk = _pattern_level(b, x, y, 3 - p, self.size)
        q_own = LEVEL_TABLE[lv_own]
        q_blk = 0.88 * LEVEL_TABLE[lv_blk]                 # defence is worth slightly less
        return float(np.clip(max(q_own, q_blk), 0.0, 0.9))
    def winrate(self, b, candidates, p):
        """rough win probability for player p given the current board (sigmoid of best threats)"""
        best = {p: 0.0, 3 - p: 0.0}
        for (x, y) in candidates[:48]:
            for who in (p, 3 - p):
                if self.wins_now(b, x, y, who):
                    best[who] = 1.0
                else:
                    lv = _pattern_level(b, x, y, who, self.size)
                    best[who] = max(best[who], LEVEL_TABLE[lv])
        z = 2.2 * (best[p] - best[3 - p])
        return float(1.0 / (1.0 + np.exp(-z)))

# reward mapping (specified reward/punishment scheme):
#   r = clip((Q - 0.2) * 2.5, -1, +1)
#   Q = 1.00  winning move              -> r = +1.0  (max dopamine burst)
#   Q = 0.92  forced block              -> r = +1.0
#   Q = 0.60  strong four threat        -> r = +1.0
#   Q = 0.38  open three                -> r = +0.45
#   Q = 0.20  neutral                   -> r =  0.0
#   Q = 0.05  weak/ignore opponent      -> r = -0.38 (punishment)
#   Q = 0.00  pointless move            -> r = -0.50
#   game won  -> +1.0 ; game lost -> -1.0 ; manual buttons -> +-1.0
def q_to_reward(q):
    return float(np.clip((q - 0.2) * 2.5, -1.0, 1.0))
