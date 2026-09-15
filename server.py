# server.py : two-fly board-game arena with a local 3D web view.
#   python server.py  ->  http://127.0.0.1:8765
# The simulation runs in a worker thread; the browser polls JSON + binary state.
# Plasticity state persists under saves/ (never touches flygo_data.npz).
import numpy as np
import json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from brain import load_bundle, FlyBrain
import game as G
from judge import GomokuJudge, q_to_reward

WEB_DIR = r"G:\guoying\flygo\web"
SAVE_DIR = r"G:\guoying\flygo\saves"
SAVE_DIR_PRISTINE = r"G:\guoying\flygo\saves\pristine"
FLY_MODEL = r"G:\guoying\flygo\fly_model.bin"

bundle = load_bundle()
N = bundle["W"].shape[0]
# readout from mushroom-body output neurons (MBON): the KC->MBON synapses we remodel
# ARE the decision pathway, so plasticity directly shapes future moves.
mb = bundle["mbon"]
if len(mb) >= 64:
    _r = np.random.default_rng(5)
    og = _r.choice(mb, 64, replace=False).reshape(64, 1).astype(np.int64)
    bundle["out_groups"] = og
    # pair each bucket's MBON with the strongest excitatory presynaptic KC available,
    # so stimulating a bucket's codebook reliably drives its own readout neuron
    pre, post, w0 = bundle["pl_pre"], bundle["pl_post"], bundle["pl_w0"]
    pairs = {}
    for bi in range(64):
        tgt = int(og[bi, 0])
        m = (post == tgt) & (w0 > 0)
        if m.any():
            pairs[bi] = int(pre[m][np.argmax(w0[m])])
    bundle["kc_pairs"] = pairs
else:
    bundle["kc_pairs"] = {}
brains = [FlyBrain(bundle, seed=1, jitter=0.15), FlyBrain(bundle, seed=7, jitter=0.15)]
# same base connectome (one species), but each fly owns an individually-jittered copy of the
# synaptic weights, its own sensory gains, and its own codebook -> genuinely different dynamics.
# jitter 0.15: strong enough for individual dynamics, small enough to avoid inhibitory collapse.
gm = G.Gomoku15()
judge = GomokuJudge(gm.size)
lock = threading.Lock()
cfg = {"playing": True, "instinct": True, "plasticity": True, "game": "gomoku",
       "window_ticks": 120, "input_current": 0.45, "instinct_alpha": 0.9, "auto_new_game": True,
       "step_once": False}
S = {"board": gm.b.tolist(), "moves": [], "move_no": 0, "to_move": 0, "winner": 0,
     "last_move": None, "tick_ms": 0.0, "move_s": 0.0, "thinking": False,
     "winrate": [0.5, 0.5], "judge": "",
     "games": 0, "wins": [0, 0], "q_hist": [[], []], "persist": {"loaded": False, "saved": ""},
     "profile": "trained",
     "flies": [{"rewards": 0, "remodel_events": 0, "syn_changed": 0, "dw_total": 0.0,
                "last_reward": 0.0, "dan_flash": 0, "bucket_rates": [0.0]*64,
                "q_avg": 0.0, "last_out": -1, "last_bucket": -1} for _ in range(2)],
     "config": cfg, "game_name": gm.name}
game_qs = [[], []]          # Q values of the current game, per fly
game_recorded = False       # game-end hook fires once per game

# ---------------- persistence: two independent profiles ------------------------
# "trained": the accumulating remodeled state (saves/)   "pristine": zero-training runs
# with initial synapses (saves/pristine/) so they never touch the trained state.
profile = "trained"

def _base_dir():
    return SAVE_DIR if profile == "trained" else SAVE_DIR_PRISTINE

def _fly_path(i):
    return os.path.join(_base_dir(), "fly%d_state.npz" % i)

def _open_session(mode):
    if profile == "trained":
        return open(os.path.join(SAVE_DIR, "session.json"), mode, encoding="utf-8")
    return open(os.path.join(SAVE_DIR_PRISTINE, "session.json"), mode, encoding="utf-8")

def _remove_in_saves(p):
    """delete only inside the saves/ tree (normalized + checked right here)"""
    ap = os.path.abspath(p)
    if ap.startswith(os.path.abspath(SAVE_DIR) + os.sep) and os.path.isfile(ap):
        os.remove(ap)

def save_all(reason):
    os.makedirs(_base_dir(), exist_ok=True)
    for i, b in enumerate(brains):
        b.save_state(_fly_path(i))
    with _open_session("w") as fh:
        json.dump({"games": S["games"], "wins": S["wins"], "q_hist": S["q_hist"],
                   "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "reason": reason}, fh)
    S["persist"]["saved"] = time.strftime("%m-%d %H:%M")

def try_load_all():
    global game_recorded
    if not (os.path.exists(_fly_path(0)) and os.path.exists(_fly_path(1))):
        S["persist"] = {"loaded": False, "saved": ""}
        return False
    ok = True
    with lock:
        for i, b in enumerate(brains):
            try:
                b.load_state(_fly_path(i))
            except Exception as e:
                print("load fly%d failed: %r" % (i, e)); ok = False
        if os.path.exists(os.path.join(_base_dir(), "session.json")):
            try:
                with _open_session("r") as fh:
                    sess = json.load(fh)
                S["games"] = int(sess.get("games", 0)); S["wins"] = list(sess.get("wins", [0, 0]))
                S["q_hist"] = list(sess.get("q_hist", [[], []]))
            except Exception as e:
                print("load session.json failed: %r" % e)
    S["persist"]["loaded"] = ok
    game_recorded = False
    print("persist loaded:", ok, "profile:", profile, "games:", S["games"])
    return ok

def _reset_learning_state():
    global game_qs, game_recorded
    for b in brains:
        b.reset_plasticity()
    game_qs = [[], []]; game_recorded = False
    S["games"] = 0; S["wins"] = [0, 0]; S["q_hist"] = [[], []]
    S["persist"] = {"loaded": False, "saved": ""}

def clear_persist():
    with lock:
        _reset_learning_state()
    _remove_in_saves(_fly_path(0))
    _remove_in_saves(_fly_path(1))
    _remove_in_saves(os.path.join(_base_dir(), "session.json"))

def set_profile(name):
    """switch between the trained state and a zero-training (pristine) run"""
    global profile
    if name not in ("trained", "pristine") or name == profile:
        return
    save_all("switch away from " + profile)
    profile = name
    with lock:
        _reset_learning_state()
        S["profile"] = profile
    try_load_all()
    new_game()

def on_game_end():
    """loser dopamine punishment + per-game Q history + autosave (fires once per game)"""
    global game_recorded
    if game_recorded or not gm.winner:
        return
    game_recorded = True
    widx = int(gm.winner) - 1
    if widx in (0, 1):
        S["wins"][widx] += 1
        brains[1 - widx].reward(-1.0)          # losing the game: punishment dopamine
        update_drive(1 - widx, -1.0); update_drive(widx, 1.0)
    for f in (0, 1):
        S["q_hist"][f].append(round(float(np.mean(game_qs[f])) if game_qs[f] else 0.0, 3))
        S["q_hist"][f] = S["q_hist"][f][-400:]
    S["games"] += 1
    save_all("auto: game %d end" % S["games"])
    print("game %d over, winner=%s  Qmean=[%.3f, %.3f]  saved (%s)" %
          (S["games"], gm.winner, S["q_hist"][0][-1], S["q_hist"][1][-1], profile))

def new_game():
    global gm, game_qs, game_recorded
    gm = G.Go9x9() if cfg["game"] == "go" else G.Gomoku15()
    game_qs = [[], []]; game_recorded = False
    for b in brains:
        b.clear_input()
    S["board"] = gm.b.tolist(); S["moves"] = []; S["move_no"] = 0
    S["winner"] = 0; S["last_move"] = None; S["game_name"] = gm.name

def snapshot():
    with lock:
        S["board"] = gm.b.tolist()
        S["moves"] = gm.moves[-16:]
        S["to_move"] = S["move_no"] % 2
        S["winner"] = int(gm.winner)
        for i, b in enumerate(brains):
            f = S["flies"][i]
            f["rewards"] = b.stats["rewards"]; f["remodel_events"] = b.stats["remodel_events"]
            f["syn_changed"] = b.stats["syn_changed"]; f["dw_total"] = round(b.stats["dw_total"], 3)
            f["last_reward"] = b.stats["last_reward"]
            f["dan_flash"] = 1 if b.dan_ticks > 0 else 0
            f["last_out"] = int(getattr(b, "last_out", -1))
            f["last_bucket"] = int(getattr(b, "last_bucket", -1))
            f["bucket_rates"] = [round(float(x), 3) for x in b.last_bucket_rates] if hasattr(b, "last_bucket_rates") else f["bucket_rates"]
            f["q_avg"] = round(float(np.mean(game_qs[i])) if game_qs[i] else 0.0, 3)
            f["drive"] = round(drive[i], 3)
            f["health"] = b.health()

def play_move(fly):
    b = brains[fly]
    p = fly + 1
    cands = gm.legal_candidates(radius=1, cap=64)
    if not cands:                                   # board region exhausted -> any empty cell
        empt = np.argwhere(gm.b == 0)
        if not len(empt):
            gm.winner = 3                           # full board, no five: draw
            return None, 0.0
        cands = [(int(x), int(y)) for y, x in empt[:64]]
    b.clear_input()
    # salience: normalized heuristic attention (engineered input gain, NOT the decision).
    evs = np.array([gm.eval_cell(x, y, p) for (x, y) in cands], dtype=np.float32)
    emax = float(evs.max()) + 1e-6
    for k, (x, y) in enumerate(cands):
        sal = 1.0 + (cfg["instinct_alpha"] * float(evs[k]) / emax if cfg["instinct"] else 0.0)
        b.stimulate_group(k, cfg["input_current"] * sal)
    nt = cfg["window_ticks"]
    acc = np.zeros(N, dtype=np.float32)
    t0 = time.time(); tsum = 0.0
    other = brains[1 - fly]
    for i in range(nt):
        t1 = time.time()
        time.sleep(0.002)                      # release the GIL so HTTP threads stay responsive
        b.tick()
        tsum += time.time() - t1
        acc += b.s_last
        if i % 3 == 0:
            other.tick()                       # the idle fly keeps living: its activity steers its flight
    b.tick_ms_cost = 1000.0 * tsum / nt
    rates = b.bucket_rates(acc, nt)
    b.last_bucket_rates = rates
    kk = int(np.argmax(rates[:len(cands)]))
    x, y = cands[kk]
    b.last_bucket = kk
    b.last_out = int(b.out_groups[kk][0]) if len(b.out_groups) else -1
    Q = judge.move_quality(gm.b, x, y, p)
    gm.place(x, y, p)
    game_qs[fly].append(float(Q))
    if gm.winner == p:
        r = 1.0
    else:
        r = q_to_reward(Q)
    wr = judge.winrate(gm.b, gm.legal_candidates(radius=2, cap=48), 3 - p)
    with lock:
        S["winrate"] = [round(1.0 - wr, 3), round(wr, 3)]
        S["judge"] = "Q=%.2f -> r=%+.2f" % (Q, r)
    b.reward(r)
    update_drive(fly, r)
    b.clear_input()
    S["move_s"] = time.time() - t0
    S["tick_ms"] = round(b.tick_ms_cost, 1)
    return (x, y), r

def do_one_move():
    fly = S["move_no"] % 2
    S["thinking"] = True
    mv, r = play_move(fly)
    S["thinking"] = False
    with lock:
        S["move_no"] += 1
        if mv: S["last_move"] = {"x": mv[0], "y": mv[1], "p": fly + 1, "r": round(r, 3)}
    snapshot()
    on_game_end()

def sim_loop():
    new_game_delay = 0.0
    while True:
        if not cfg["playing"]:
            if cfg.get("step_once"):
                cfg["step_once"] = False
                do_one_move()
            for b in brains: b.tick()              # paused: brains keep breathing
            time.sleep(0.12); continue
        if gm.winner:
            if cfg["auto_new_game"]:
                if new_game_delay == 0.0: new_game_delay = time.time() + 7
                if time.time() > new_game_delay:
                    new_game(); new_game_delay = 0.0
            for b in brains: b.tick()
            time.sleep(0.12); continue
        do_one_move()

# ---- neural steering readout: the 3D body's flight is driven by the fly's own activity ----
# left/right hemisphere asymmetry -> yaw, dorsal/ventral asymmetry -> lift, mean rate -> thrust.
_pos = bundle["pos"]
_mL = _pos[:, 0] < -0.04; _mR = _pos[:, 0] > 0.04
_mU = _pos[:, 1] > 0.12;  _mD = _pos[:, 1] < -0.12
steer_state = [{"yaw": 0.0, "thrust": 1.0, "lift": 0.0, "byaw": None, "blift": None},
               {"yaw": 0.0, "thrust": 1.0, "lift": 0.0, "byaw": None, "blift": None}]
def steer_readout(f):
    r = brains[f].rate
    rl, rr = float(r[_mL].mean()), float(r[_mR].mean())
    ru, rd = float(r[_mU].mean()), float(r[_mD].mean())
    tot = float(r.mean())
    s = steer_state[f]
    # the two hemispheres / dorsoventral halves have different neuron counts, so the raw
    # asymmetry has a constant offset: steer only with the DEVIATION from a slow baseline
    ay = (rr - rl) / (rr + rl + 1e-3); al = (ru - rd) / (ru + rd + 1e-3)
    s["byaw"] = ay if s["byaw"] is None else 0.985 * s["byaw"] + 0.015 * ay
    s["blift"] = al if s["blift"] is None else 0.985 * s["blift"] + 0.015 * al
    yaw = float(np.clip((ay - s["byaw"]) * 25.0, -1.0, 1.0))
    lift = float(np.clip((al - s["blift"]) * 25.0, -1.0, 1.0))
    thrust = float(np.clip(0.6 + tot * 0.6, 0.5, 1.6))
    s["yaw"] = 0.55 * s["yaw"] + 0.45 * yaw
    s["lift"] = 0.55 * s["lift"] + 0.45 * lift
    s["thrust"] = 0.55 * s["thrust"] + 0.45 * thrust
    return {"yaw": round(s["yaw"], 3), "thrust": round(s["thrust"], 3), "lift": round(s["lift"], 3),
            "rate": round(tot, 3), "lr": [round(rl, 3), round(rr, 3)]}

# ---- neural motor / behaviour loop ------------------------------------------------
# Every body motion is driven by the fly's own population activity. Region masks prefer the
# REAL motor-neuron pools (flygo_motor.npz built from MaleCNS type/superclass annotations);
# when that file is absent we fall back to spatial proxies and say so in /api/meta.
PHYS_SRV = {"near_current": 0.30, "burst_gain": 2.5, "drive_gain": 0.35, "drive_decay": 0.985}
_mLEG = (_pos[:, 2] < -0.30) & (_pos[:, 2] > -0.58)
_mHIND = _pos[:, 2] < -0.72
_mDANm = np.zeros(N, dtype=bool); _mDANm[bundle["dan"]] = True
_mNECK = _mU
_mWING = _mU
motor_source = "spatial_proxy"
MOTOR_NPZ = r"G:\guoying\flygo\flygo_motor.npz"
if os.path.exists(MOTOR_NPZ):
    try:
        _mz = np.load(MOTOR_NPZ)
        def _mask_of(key):
            arr = _mz[key].astype(np.int64) if key in _mz else np.array([], dtype=np.int64)
            mm = np.zeros(N, dtype=bool); mm[arr[arr < N]] = True; return mm
        if len(_mz["mn_leg"]):
            _mLEG = _mask_of("mn_leg")
        if len(_mz["mn_hind"]):
            _mHIND = _mask_of("mn_hind")
        if len(_mz["mn_neck"]):
            _mNECK = _mask_of("mn_neck")
        if len(_mz["mn_wing"]):
            _mWING = _mask_of("mn_wing")
        motor_source = "real_mn"
        print("motor binding: real motor-neuron pools "
              "(leg=%d hind=%d neck=%d wing=%d)" % (_mLEG.sum(), _mHIND.sum(), _mNECK.sum(), _mWING.sum()))
    except Exception as e:
        print("flygo_motor.npz unreadable, spatial proxies kept: %r" % e)
_base = [{"leg": None, "hind": None, "dors": None}, {"leg": None, "hind": None, "dors": None}]
drive = [0.35, 0.35]
last_penalty = [0.0, 0.0]
def _burst(f, key, r):
    b = _base[f]
    if b[key] is None: b[key] = r                     # baseline starts at the first observation
    b[key] = 0.97 * b[key] + 0.03 * r
    return float(np.clip((r - b[key]) / (b[key] + 1e-3) * PHYS_SRV["burst_gain"], -1.0, 1.0))
def motor_step(f, near, holding):
    b = brains[f]
    b.I_ext[_mLEG] = np.float32(PHYS_SRV["near_current"] * near)
    rate = b.rate
    leg = float(rate[_mLEG].mean()); hind = float(rate[_mHIND].mean())
    neck = float(rate[_mNECK].mean()); wing = float(rate[_mWING].mean())
    out = steer_readout(f)
    out["grasp"] = round(_burst(f, "leg", leg), 3)
    out["groom_f"] = round(float(np.clip(leg / ((_base[f]["leg"] or leg) + 1e-3) - 1.0, -1.0, 1.5)), 3)
    out["groom_h"] = round(_burst(f, "hind", hind), 3)
    out["wingflick"] = round(_burst(f, "dors", wing), 3)
    out["neck"] = round(float(np.clip(neck / ((_base[f]["dors"] or neck) + 1e-3) - 1.0, -1.0, 1.5)), 3)
    out["dan"] = round(float(rate[_mDANm].mean()), 3)
    out["drive"] = round(drive[f], 3)
    out["leg_rate"] = round(leg, 3)
    return out
def update_drive(f, r):
    """dopamine motivation: rewards raise it (addiction), punishment/time erode it"""
    drive[f] = float(np.clip(drive[f] * PHYS_SRV["drive_decay"] + PHYS_SRV["drive_gain"] * max(r, 0.0)
                             - 0.12 * max(-r, 0.0), 0.05, 1.0))

try_load_all()          # restore remodeled synapses from saves/ if present
threading.Thread(target=sim_loop, daemon=True).start()

# per-neuron class byte for the whole-CNS colored point cloud (flyvis-style):
# 0 brain intrinsic, 1 VNC intrinsic, 2 visual input, 3 KC, 4 MBON, 5 DAN, 6 motor, 7 descending
_CLS = np.zeros(N, dtype=np.uint8)
_CLS[_pos[:, 2] < -0.42] = 1
_CLS[bundle["in_groups"].reshape(-1)] = 2
_CLS[bundle["kc"]] = 3
_CLS[bundle["mbon"]] = 4
_CLS[bundle["dan"]] = 5
_CLS[bundle["out_groups"].reshape(-1)] = 7
if motor_source == "real_mn":
    _mz2 = np.load(MOTOR_NPZ)
    for _k in _mz2.files:
        _arr = _mz2[_k].astype(np.int64)
        _CLS[_arr[_arr < N]] = 6

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
    def _fly(self):
        return int(self.path.split("fly=")[1]) % 2 if "fly=" in self.path else 0
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/" or path == "/index.html":
            self._send(200, open(WEB_DIR + r"\index.html", "rb").read(), "text/html; charset=utf-8")
        elif path == "/main.js":
            self._send(200, open(WEB_DIR + r"\main.js", "rb").read(), "text/javascript")
        elif path == "/three.min.js":
            self._send(200, open(WEB_DIR + r"\three.min.js", "rb").read(), "text/javascript")
        elif path == "/fly_model.bin":
            self._send(200, open(FLY_MODEL, "rb").read(), "application/octet-stream")
        elif path == "/api/steer":
            self._send(200, json.dumps(steer_readout(self._fly())).encode())
        elif path == "/api/state":
            with lock:
                body = json.dumps(S).encode()
            self._send(200, body)
        elif path == "/api/act":
            self._send(200, brains[self._fly()].rate.astype(np.float32).tobytes(), "application/octet-stream")
        elif path == "/api/pos":
            self._send(200, bundle["pos"].astype(np.float32).tobytes(), "application/octet-stream")
        elif path == "/api/classes":
            self._send(200, _CLS.tobytes(), "application/octet-stream")
        elif path == "/api/syn":
            # KC-codebook -> MBON synapses of one fly: [n int32][pre n*int32][post n*int32][m n*float32]
            b = brains[self._fly()]
            kcs = b.kc_groups.reshape(-1) if len(b.kc_groups) else np.array([], dtype=np.int64)
            with lock:
                mask = np.isin(b.pl_pre, kcs)
                pre = b.pl_pre[mask].astype(np.int32)
                post = b.pl_post[mask].astype(np.int32)
                m = b.pl_m[mask].astype(np.float32)
            body = (np.array([len(pre)], dtype=np.int32).tobytes() +
                    pre.tobytes() + post.tobytes() + m.tobytes())
            self._send(200, body, "application/octet-stream")
        elif path == "/api/net":
            # decision-circuit subgraph of one fly: the neurons we stimulate/read + every real
            # synapse between them. [2 int32 n_nodes n_edges][nodes][pre][post][signed w]
            b = brains[self._fly()]
            with lock:
                roi = np.unique(np.concatenate([bundle["in_groups"].reshape(-1),
                                                b.kc_groups.reshape(-1),
                                                bundle["out_groups"].reshape(-1),
                                                bundle["mbon"],
                                                bundle["dan"]])).astype(np.int32)
                sub = bundle["W"][roi][:, roi].tocoo()
                pre = roi[sub.col].astype(np.int32)
                post = roi[sub.row].astype(np.int32)
                w = (sub.data * bundle["sign"][roi[sub.col]]).astype(np.float32)
            body = (np.array([len(roi), len(pre)], dtype=np.int32).tobytes() +
                    roi.tobytes() + pre.tobytes() + post.tobytes() + w.tobytes())
            self._send(200, body, "application/octet-stream")
        elif path == "/api/meta":
            meta = {"N": int(N),
                    "dan": bundle["dan"].astype(int).tolist(),
                    "in_groups": bundle["in_groups"].astype(int).reshape(-1).tolist(),
                    "out_groups": bundle["out_groups"].astype(int).reshape(-1).tolist(),
                    "mbon": bundle["mbon"].astype(int).tolist(),
                    "kc0": brains[0].kc_groups.astype(int).reshape(-1).tolist(),
                    "kc1": brains[1].kc_groups.astype(int).reshape(-1).tolist(),
                    "pairs0": {str(k): int(v) for k, v in brains[0].kc_pairs.items()},
                    "pairs1": {str(k): int(v) for k, v in brains[1].kc_pairs.items()},
                    "motor_source": motor_source,
                    "motor_counts": {"leg": int(_mLEG.sum()), "hind": int(_mHIND.sum()),
                                     "wing": int(_mWING.sum()), "neck": int(_mNECK.sum())}}
            self._send(200, json.dumps(meta).encode())
        else:
            self._send(404, b"{}")
    def do_POST(self):
        if self.path.startswith("/api/motor"):
            ln = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(ln) or b"{}")
            f = int(req.get("fly", 0)) % 2
            out = motor_step(f, float(np.clip(req.get("near", 0.0), 0.0, 1.0)), int(req.get("holding", 0)))
            self._send(200, json.dumps(out).encode()); return
        if self.path != "/api/control":
            self._send(404, b"{}"); return
        ln = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(ln) or b"{}")
        cmd = req.get("cmd", "")
        if cmd == "pause": cfg["playing"] = False
        elif cmd == "resume": cfg["playing"] = True
        elif cmd == "step": cfg["playing"] = False; cfg["step_once"] = True
        elif cmd == "reset": new_game()
        elif cmd == "reward" or cmd == "punish":
            fly = int(req.get("fly", 0)) % 2
            brains[fly].reward(1.0 if cmd == "reward" else -0.8)
            update_drive(fly, 1.0 if cmd == "reward" else -0.8)
        elif cmd == "penalty":                    # body hit the glass box: mild punishment, rate-limited
            fly = int(req.get("fly", 0)) % 2
            if time.time() - last_penalty[fly] > 1.0:
                last_penalty[fly] = time.time()
                brains[fly].reward(-0.4); update_drive(fly, -0.4)
        elif cmd == "plasticity": cfg["plasticity"] = not cfg["plasticity"]; [setattr(b, "plasticity_on", cfg["plasticity"]) for b in brains]
        elif cmd == "instinct": cfg["instinct"] = not cfg["instinct"]
        elif cmd == "instinct_alpha":
            cfg["instinct_alpha"] = float(np.clip(req.get("value", 0.9), 0.0, 2.0))
        elif cmd == "game": pass    # go mode disabled: scoring model compute still under evaluation
        elif cmd == "speed":
            wt = int(req.get("value", 120)); cfg["window_ticks"] = int(np.clip(wt, 40, 300))
        elif cmd == "save": save_all("manual")
        elif cmd == "load": try_load_all()
        elif cmd == "clear_persist": clear_persist()
        elif cmd == "profile": set_profile(str(req.get("value", "trained")))
        elif cmd == "phys":
            k = str(req.get("key", ""))
            if k in PHYS_SRV: PHYS_SRV[k] = float(np.clip(float(req.get("value", PHYS_SRV[k])), 0.0, 5.0))
        snapshot()
        self._send(200, json.dumps({"ok": True, "instinct_alpha": cfg["instinct_alpha"], "profile": profile,
                                    "phys": PHYS_SRV}).encode())

if __name__ == "__main__":
    print("serving http://127.0.0.1:8765")
    ThreadingHTTPServer(("127.0.0.1", 8765), H).serve_forever()
