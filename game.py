# game.py : board games for the two-fly arena.
# Gomoku15 : full five-in-a-row rules (primary game)
# Go9x9    : experimental minimal Go (capture, suicide ban, pass, rough Chinese score)
import numpy as np

class Gomoku15:
    size = 15
    name = "gomoku"
    def __init__(self):
        self.reset()
    def reset(self):
        self.b = np.zeros((self.size, self.size), dtype=np.int8)   # 0 empty, 1 black, 2 white
        self.moves = []
        self.winner = 0
    def legal_candidates(self, radius=2, cap=64):
        if not self.moves:
            c = self.size // 2
            return [(c, c)]
        occ = self.b != 0
        near = np.zeros_like(occ)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ys = slice(max(0, dy), self.size + min(0, dy))
                xs = slice(max(0, dx), self.size + min(0, dx))
                ys2 = slice(max(0, -dy), self.size + min(0, -dy))
                xs2 = slice(max(0, -dx), self.size + min(0, -dx))
                near[ys2, xs2] |= occ[ys, xs]
        cand = np.argwhere(near & ~occ)
        out = [(int(x), int(y)) for y, x in cand]
        return out[:cap]
    def place(self, x, y, p):
        self.b[y, x] = p
        self.moves.append((x, y, p))
        if self._win(x, y, p):
            self.winner = p
    def _win(self, x, y, p):
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            n = 1
            for s in (1, -1):
                xx, yy = x + dx*s, y + dy*s
                while 0 <= xx < self.size and 0 <= yy < self.size and self.b[yy, xx] == p:
                    n += 1; xx += dx*s; yy += dy*s
            if n >= 5:
                return True
        return False
    def eval_cell(self, x, y, p):
        """salience heuristic (engineered attention input, not brain output)"""
        s = 0.0
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            for who, wgt in ((p, 1.0), (3 - p, 0.9)):
                n, open_ends = 1, 0
                for sgn in (1, -1):
                    xx, yy = x + dx*sgn, y + dy*sgn
                    while 0 <= xx < self.size and 0 <= yy < self.size and self.b[yy, xx] == who:
                        n += 1; xx += dx*sgn; yy += dy*sgn
                    if 0 <= xx < self.size and 0 <= yy < self.size and self.b[yy, xx] == 0:
                        open_ends += 1
                if n >= 4: s += 50.0 * wgt
                elif n == 3 and open_ends == 2: s += 12.0 * wgt
                elif n == 3: s += 4.0 * wgt
                elif n == 2 and open_ends == 2: s += 2.0 * wgt
        return s
    def move_reward(self, x, y, p):
        """small automatic dopamine signal for pattern gains (bounded, disclosed)"""
        r = np.clip(self.eval_cell(x, y, p) / 60.0, -0.3, 0.5)
        if self.winner == p: r = 1.0
        return float(r)

class Go9x9:
    size = 9
    name = "go9x9 (experimental)"
    def __init__(self):
        self.reset()
    def reset(self):
        self.b = np.zeros((self.size, self.size), dtype=np.int8)
        self.moves = []; self.winner = 0
        self.captures = {1: 0, 2: 0}
        self.passes = 0
    def _neigh(self, x, y):
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            xx, yy = x+dx, y+dy
            if 0 <= xx < self.size and 0 <= yy < self.size:
                yield xx, yy
    def _group_libs(self, x, y):
        p = self.b[y, x]; seen = {(x, y)}; libs = set(); stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            for nx, ny in self._neigh(cx, cy):
                v = self.b[ny, nx]
                if v == 0: libs.add((nx, ny))
                elif v == p and (nx, ny) not in seen:
                    seen.add((nx, ny)); stack.append((nx, ny))
        return seen, libs
    def legal_candidates(self, radius=2, cap=64):
        if not self.moves:
            c = self.size // 2
            return [(c, c)]
        occ = self.b != 0
        near = np.zeros_like(occ)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ys = slice(max(0, dy), self.size + min(0, dy)); xs = slice(max(0, dx), self.size + min(0, dx))
                ys2 = slice(max(0, -dy), self.size + min(0, -dy)); xs2 = slice(max(0, -dx), self.size + min(0, -dx))
                near[ys2, xs2] |= occ[ys, xs]
        cand = np.argwhere(near & ~occ)
        out = []
        for x, y in cand:
            if self._would_be_legal(int(x), int(y), 1):     # same legality for both colours
                out.append((int(x), int(y)))
        return out[:cap]
    def _would_be_legal(self, x, y, p):
        self.b[y, x] = p
        _, libs = self._group_libs(x, y)
        ok = len(libs) > 0
        if not ok:
            for nx, ny in self._neigh(x, y):
                if self.b[ny, nx] == 3 - p:
                    g, l = self._group_libs(nx, ny)
                    if not l: ok = True
        self.b[y, x] = 0
        return ok
    def place(self, x, y, p):
        self.b[y, x] = p
        self.moves.append((x, y, p))
        self.passes = 0
        for nx, ny in self._neigh(x, y):
            if self.b[ny, nx] == 3 - p:
                g, l = self._group_libs(nx, ny)
                if not l:
                    for gx, gy in g: self.b[gy, gx] = 0
                    self.captures[p] += len(g)
    def pass_move(self, p):
        self.moves.append((-1, -1, p))
        self.passes += 1
        if self.passes >= 2:
            black = int((self.b == 1).sum()) + self.captures[1]
            white = int((self.b == 2).sum()) + self.captures[2] + 3   # rough komi
            self.winner = 1 if black > white else 2
    def eval_cell(self, x, y, p):
        s = 0.0
        for nx, ny in self._neigh(x, y):
            v = self.b[ny, nx]
            if v == p: s += 1.0
            elif v == 3 - p: s += 0.8
        if (x in (2, 6) and y in (2, 6)): s += 1.0                 # star points
        return s
    def move_reward(self, x, y, p):
        """called BEFORE place() by the server; pattern salience only"""
        return float(np.clip(0.1 * self.eval_cell(x, y, p), -0.2, 0.3))
