# analyze_activity.py : headless propagation audit.
# Question 1: why did most neurons never change activity? (old regime: rec_scale 0.05)
# Question 2: with signed recurrence at rec_gain ~0.30, does activity propagate through the
#             REAL wiring while staying stable (no whole-brain ignition)?
# Reports: ever-spiked fraction by class, rates, and the structurally reachable set
# (BFS hops over W) vs the dynamically activated set.
import numpy as np
import sys
sys.path.insert(0, r"G:\guoying\flygo")
from brain import load_bundle, FlyBrain

bundle = load_bundle()
pz = np.load(r"G:\guoying\flygo\flygo_proprio.npz")
bundle["vis_pool"] = pz["vis_pool"]; bundle["dn_pool"] = pz["dn_pool"]

def bfs_reach(W, seeds, hops):
    reach = np.zeros(W.shape[0], dtype=bool); frontier = np.zeros(W.shape[0], dtype=bool)
    frontier[seeds] = True
    sizes = []
    for _ in range(hops):
        nxt = (W @ frontier.astype(np.float32)) != 0
        nxt &= ~reach
        reach |= nxt; frontier = nxt
        sizes.append(int(reach.sum()))
    return reach, sizes

def run(rec_scale, seed=1):
    b = FlyBrain(bundle, seed=seed, jitter=0.15, rec_scale=rec_scale)
    classes = {
        "vis_in": bundle["in_groups"].reshape(-1),
        "kc": bundle["kc"], "mbon": bundle["mbon"], "dan": bundle["dan"],
        "dn_out": bundle["out_groups"].reshape(-1),
        "prop_ch": pz["prop_ch"], "prop_jo": pz["prop_jo"],
    }
    evr = np.zeros(b.N, dtype=bool)
    # A) 120 ticks silence (no input) - noise alone must NOT ignite the brain
    for _ in range(120):
        b.tick(); evr |= b.s_last > 0
    idle = {"ever_active": float(evr.mean()), "rate_mean": float(b.rate.mean()),
            "rate_max": float(b.rate.max())}
    # B) a think window: stimulate 20 buckets like play_move does
    b.clear_input()
    for k in range(20):
        b.stimulate_group(k, 0.55)
    evr2 = np.zeros(b.N, dtype=bool)
    for _ in range(120):
        b.tick(); evr2 |= b.s_last > 0
    by_class = {n: float(evr2[idx].mean()) for n, idx in classes.items()}
    other = ~np.zeros(b.N, dtype=bool)
    for idx in classes.values(): other[idx] = False
    by_class["everything_else"] = float(evr2[other].mean())
    stim = {"ever_active_all": float(evr2.mean()), "rate_mean": float(b.rate.mean()),
            "v_sat": float((np.abs(b.v) > 0.95).mean()), "by_class": by_class}
    # C) reward burst (DAN + plasticity) then 60 more ticks
    b.reward(1.0)
    for _ in range(3): b.tick()
    b.clear_input()
    for k in range(20): b.stimulate_group(k, 0.55)   # keep same input during burst
    evr3 = np.zeros(b.N, dtype=bool)
    for _ in range(60):
        b.tick(); evr3 |= b.s_last > 0
    rew = {"ever_active_all": float(evr3.mean()), "rate_mean": float(b.rate.mean()),
           "v_sat": float((np.abs(b.v) > 0.95).mean())}
    return idle, stim, rew, b

print("=== OLD regime: rec_scale = 0.05 (unsigned-equivalent) ===")
i1, s1, r1, b1 = run(0.05)
print("idle:", {k: round(v, 4) for k, v in i1.items()})
print("stim:", {k: round(v, 4) if isinstance(v, float) else {a: round(c, 4) for a, c in v.items()} for k, v in s1.items()})
print("reward:", {k: round(v, 4) for k, v in r1.items()})

print("=== NEW regime: rec_scale = 0.30, signed recurrence, per-fly I/O redraw ===")
i2, s2, r2, b2 = run(0.30)
print("idle:", {k: round(v, 4) for k, v in i2.items()})
print("stim:", {k: round(v, 4) if isinstance(v, float) else {a: round(c, 4) for a, c in v.items()} for k, v in s2.items()})
print("reward:", {k: round(v, 4) for k, v in r2.items()})
print("fly0 rec_scale actually drawn:", round(b2.scale, 3))

seeds = np.unique(bundle["in_groups"][:20].reshape(-1))
reach, sizes = bfs_reach(b2.W, seeds, 3)
print("structural reachability from 20 stimulated input groups (BFS over W):", sizes,
      "-> max fraction", round(sizes[-1] / b2.N, 3))

# per-fly input independence: overlap of the two flies' in_groups
c = FlyBrain(bundle, seed=7, jitter=0.15, rec_scale=0.30)
ov = len(np.intersect1d(b2.in_groups.reshape(-1), c.in_groups.reshape(-1)))
print("in-group overlap between fly1 and fly7:", ov, "/", b2.in_groups.size)
