# brain.py : lightweight LIF simulation of the MaleCNS connectome for two-fly games.
# - one shared signed wiring matrix W (rows=post), per-fly state (v, refractory, traces)
# - dopamine: reward() injects current into the real PPL/PPM dopaminergic neurons
# - dynamic remodeling: three-factor rule dw = eta * r * pre_trace * post_trace,
#   applied ONLY to real KC->MBON synapses (mushroom body learning circuit)
import numpy as np
import json
from scipy.sparse import csr_matrix

DATA_NPZ = r"G:\guoying\flygo\flygo_data.npz"

def load_bundle():
    z = np.load(DATA_NPZ)
    W = csr_matrix((z["W_data"], z["W_indices"], z["W_indptr"]), shape=tuple(int(x) for x in z["W_shape"]))
    # postsynaptic normalization: cap every neuron's total incoming |weight| at 1
    # (keeps the wiring pattern, prevents excitatory runaway, synchronized groups still dominate)
    rs = np.abs(W).sum(axis=1).A1
    W = W.multiply(1.0 / np.maximum(rs, 1e-6)[:, None]).tocsr()
    W.data = W.data.astype(np.float32)
    # true KC->MBON subset (pre = KC, post = MBON) from the normalized matrix
    kcset = np.zeros(W.shape[0], dtype=bool); kcset[z["kc"]] = True
    mbset = np.zeros(W.shape[0], dtype=bool); mbset[z["mbon"]] = True
    coo = W.tocoo()
    mm = kcset[coo.col] & mbset[coo.row]
    pl_pre, pl_post, pl_w0 = coo.col[mm].astype(np.int64), coo.row[mm].astype(np.int64), coo.data[mm].astype(np.float32)
    return {"W": W, "sign": z["sign"], "pos": z["pos"], "ids": z["ids"],
            "in_groups": z["in_groups"], "out_groups": z["out_groups"],
            "dan": z["dan"], "kc": z["kc"], "mbon": z["mbon"],
            "pl_pre": pl_pre, "pl_post": pl_post, "pl_w0": pl_w0}

class FlyBrain:
    """One fly's brain. Shares the wiring matrix; owns its own state and plasticity."""
    def __init__(self, bundle, seed=1, dt=0.002, rec_scale=0.30, jitter=0.25):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if jitter > 0.0:
            # same base connectome (same species), individual synaptic strengths:
            # every synapse gets a fixed lognormal multiplier from this fly's own rng,
            # then rows are re-normalized so total incoming drive per neuron stays ~1.
            # Two flies therefore run genuinely different dynamics from the same wiring pattern.
            W0 = bundle["W"]
            j = self.rng.lognormal(0.0, jitter, W0.nnz).astype(np.float32)
            W = W0.copy()
            W.data = (W0.data * j).astype(np.float32)
            rs = np.abs(W).sum(axis=1).A1
            W = W.multiply(1.0 / np.maximum(rs, 1e-6)[:, None]).tocsr()
            W.data = W.data.astype(np.float32)
            self.W = W
        else:
            self.W = bundle["W"]
        self.sign = bundle["sign"]
        self.N = self.W.shape[0]
        self.dt = dt
        # recurrent coupling gain. Measured on the real wiring: a sensory volley covers >50%
        # of the incoming weight of a few hundred dedicated rows, but stimulated neurons have
        # a ~0.14 spike duty cycle, so effective synchrony stays ~0.07 — BELOW the firing
        # requirement at any gain that keeps the idle brain silent. Single-hop propagation
        # is therefore structurally suppressed by postsynaptic row-normalization (real
        # synapses are lognormal with dominant inputs — see roadmap). Evoked activity stays
        # in the stimulated circuit + MBON layer: sparse coding, like in-vivo fly imaging.
        # Each fly draws its own gain (+/-10%): same species, different excitability.
        self.scale = float(np.clip(self.rng.lognormal(np.log(rec_scale), 0.10), 0.12, 0.6))
        self.v = np.zeros(self.N, dtype=np.float32)
        self.refr = np.zeros(self.N, dtype=np.int16)
        self.vth = self.rng.uniform(0.26, 0.40, self.N).astype(np.float32)   # heterogeneous thresholds
        self.rate = np.zeros(self.N, dtype=np.float32)        # EMA spike rate (Hz-ish)
        self.pre_tr = np.zeros(self.N, dtype=np.float32)      # eligibility traces
        self.post_tr = np.zeros(self.N, dtype=np.float32)
        self.I_ext = np.zeros(self.N, dtype=np.float32)
        self.s_last = np.zeros(self.N, dtype=np.float32)
        self.dan_mask = np.zeros(self.N, dtype=bool); self.dan_mask[bundle["dan"]] = True
        self.dan_ticks = 0
        self.in_groups = bundle["in_groups"]; self.out_groups = bundle["out_groups"]
        # per-fly sensory interface: each fly draws its OWN input neurons from the full
        # visual pool (same anatomy, different individuals). out_groups is NOT redrawn:
        # the server pairs each bucket with its MBON readout + strongest presynaptic KC.
        # Requires vis_pool in the bundle (flygo_proprio.npz); without it both flies share
        # the bundle groups (disclosed fallback).
        vp = bundle.get("vis_pool")
        NG = len(self.in_groups) if len(self.in_groups) else 64
        # bucket fan-in: a real visual stimulus synchronously drives thousands of columns;
        # ni_per_bucket sizes each bucket's afferent volley so downstream neurons receive
        # assembly-level input and activity propagates through the REAL wiring.
        ni = int(bundle.get("ni_per_bucket", self.in_groups.shape[1]))
        if vp is not None and len(vp) >= NG * ni:
            # RETINOTOPIC sampling: each bucket owns one contiguous anatomical sub-region of
            # the optic lobe (z-layer x angular sector), not salt-and-pepper neurons.
            # Columnar partners of a real sub-region share downstream targets, so a bucket's
            # volley converges and activity propagates through the real wiring — a random
            # subset would give every downstream neuron 1-2 active inputs, below the ~55%
            # row-synchrony threshold that postsynaptic row-normalization imposes.
            pos = bundle["pos"]
            z = pos[vp, 2]; ang = np.arctan2(pos[vp, 1], pos[vp, 0])
            zr = float(z.max() - z.min())
            nsec = int(np.sqrt(NG) + 0.5)                      # e.g. NG=64 -> 8x8 sectors
            li = np.minimum(((z - z.min()) / max(zr, 1e-6) * nsec).astype(int), nsec - 1)
            si = np.minimum(((ang + np.pi) / (2*np.pi) * nsec).astype(int), nsec - 1)
            cell = li * nsec + si
            take = []
            rng_perm = self.rng.permutation(len(vp))          # own draw order inside cells
            for k in range(NG):
                members = vp[cell == (k % (nsec*nsec))] if k < nsec*nsec else vp
                if len(members) < ni:                         # pad from the whole pool
                    extra = self.rng.choice(np.setdiff1d(vp, members, assume_unique=False),
                                            ni - len(members), replace=False)
                    members = np.concatenate([members, extra])
                else:
                    members = self.rng.choice(members, ni, replace=False)
                take.append(members)
            self.in_groups = np.concatenate(take).astype(np.int64).reshape(NG, ni)
        # plasticity subset (true KC->MBON synapses); plasticity acts on a multiplicative
        # gain m: effective weight = base * (1 + m)  ->  scale-free, biologically tidy
        self.pl_pre = bundle["pl_pre"]; self.pl_post = bundle["pl_post"]
        self.pl_w0 = bundle["pl_w0"].copy()
        self.pl_m = np.zeros(len(self.pl_w0), dtype=np.float32)
        # engineered "feature -> mushroom body" projection: each candidate bucket excites its
        # own small KC subset; the two flies get DIFFERENT codebooks (individuality).
        # KCs are drawn from those that actually own KC->MBON synapses in this connectome,
        # so stimulated codebooks can reach the readout and the plasticity circuit.
        kc_all = bundle["kc"]
        NG = len(self.in_groups) if len(self.in_groups) else 64
        covered = np.unique(self.pl_pre) if len(self.pl_pre) else kc_all
        base = covered if len(covered) >= 32 else kc_all
        need = NG * 2
        if len(base):
            idx = np.tile(base, need // len(base) + 1)[:need].copy()
            self.rng.shuffle(idx)
            self.kc_groups = idx.reshape(NG, 2).astype(np.int64)
        else:
            self.kc_groups = np.array([], dtype=np.int64).reshape(0, 0)
        self.kc_pairs = bundle.get("kc_pairs", {})   # bucket -> strongly-coupled KC (server-paired)
        # own sensory interface: same anatomy (same species looking at the same board), but this
        # fly's sensory drive has its own per-group gain (+/-15% lognormal) and its own paired-KC
        # overdrive map, re-drawn from this fly's rng among KCs that actually synapse onto the
        # bucket's MBON readout.
        NG_ = len(self.in_groups)
        self.sens_gain = np.exp(self.rng.normal(0.0, 0.14, NG_)).astype(np.float32) if NG_ else np.zeros(0, np.float32)
        new_pairs = {}
        for k, v in self.kc_pairs.items():
            mbon = int(self.out_groups[k][0]) if k < len(self.out_groups) else -1
            cand = self.pl_pre[self.pl_post == mbon] if mbon >= 0 else []
            new_pairs[int(k)] = int(self.rng.choice(cand)) if len(cand) else int(v)
        self.kc_pairs = new_pairs
        # ---- per-bucket readout: the REAL MBONs this bucket's KC codebook synapses onto ----
        # Decision = population rate of the mushroom-body output neurons that this bucket's
        # codebook actually drives (plus the bucket's DN group). This is the layer where
        # three-factor plasticity acts, so dopamine literally rewrites the decision scores.
        self._pre2post = {}
        for _p, _q in zip(self.pl_pre.tolist(), self.pl_post.tolist()):
            self._pre2post.setdefault(_p, []).append(_q)
        self._build_readout()
        self.plasticity_on = True
        self.pending_reward = 0.0
        self.stats = {"rewards": 0, "remodel_events": 0, "syn_changed": 0, "dw_total": 0.0, "last_reward": 0.0}
        self.tick_ms_cost = 0.0

    def _build_readout(self):
        """readout_groups[k] = real MBONs postsynaptic to bucket k's KC codebook"""
        self.readout_groups = []
        NG_ = len(self.in_groups)
        for k in range(NG_):
            ks = set(self.kc_groups[k].tolist()) if len(self.kc_groups) and k < len(self.kc_groups) else set()
            if k in self.kc_pairs: ks.add(int(self.kc_pairs[k]))
            mb = sorted({q for kc in ks for q in self._pre2post.get(kc, ())})
            self.readout_groups.append(np.array(mb[:12], dtype=np.int64))

    # ---- sensory stimulation -----------------------------------------------
    def clear_input(self):
        self.I_ext.fill(0.0)

    def stimulate_pool(self, idx, current):
        """drive an arbitrary real neuron pool (e.g. mechanosensory/proprioceptive afferents)
        with a scalar current — body->brain feedback uses this"""
        if len(idx):
            self.I_ext[idx] = np.float32(current)

    def stimulate_group(self, k, current):
        """excite input bucket k (candidate move k) + its mushroom-body codebook,
        scaled by this fly's own sensory gain for that bucket"""
        if len(self.sens_gain) and k < len(self.sens_gain):
            current = current * float(self.sens_gain[k])
        if k < len(self.in_groups):
            self.I_ext[self.in_groups[k]] = current
        if len(self.kc_groups) and k < len(self.kc_groups):
            self.I_ext[self.kc_groups[k]] = current * 1.2
        if k in self.kc_pairs:
            self.I_ext[self.kc_pairs[k]] = current * 1.5

    # ---- dopamine -----------------------------------------------------------
    def reward(self, r):
        self.pending_reward = float(np.clip(r, -1.0, 1.0))
        self.dan_ticks = 25                                  # ~50 ms burst into real DANs
        self.stats["rewards"] += 1
        self.stats["last_reward"] = self.pending_reward

    # ---- one integration step ----------------------------------------------
    def tick(self):
        dt = self.dt
        noise = self.rng.normal(0.10, 0.08, self.N).astype(np.float32)
        # stimulus-driven regime: noise floor alone stays sub-threshold (v~0.10 < vth~0.26+),
        # activity comes from stimulation + recurrent amplification. (A tonic 0.22 test caused
        # whole-brain ignition: recurrent gain + plastic KC->MBON feedback are hair-trigger.)
        # NOTE: W already carries neurotransmitter sign (prepare_data.py multiplies sign into
        # the edge weights) — do NOT multiply by sign again: that squares it (+-1 -> +1) and
        # turns every inhibitory synapse into an excitatory one. Signed recurrence means
        # GABAergic/glutamatergic inputs now genuinely hyperpolarize their targets.
        I = noise + self.I_ext + self.scale * (self.W @ self.s_last)
        if len(self.pl_w0):
            # remodeled mushroom-body synapses feed back into the dynamics
            # (pl_w0 already carries sign — same rule as the main recurrent term)
            I += 2000.0 * np.bincount(self.pl_post,
                                    weights=(self.pl_w0 * (1.0 + self.pl_m)) * self.s_last[self.pl_pre],
                                    minlength=self.N).astype(np.float32)
        if self.dan_ticks > 0:
            I[self.dan_mask] += 1.2
            self.dan_ticks -= 1
        decay = np.float32(0.905)                             # exp(-dt/tau), tau = 20 ms
        v = self.v * decay + I * (1.0 - decay)
        spikes = (v >= self.vth) & (self.refr <= 0)
        self.refr[spikes] = 2
        self.refr[self.refr > 0] -= 1
        v[spikes] = 0.0
        np.clip(v, -0.2, 1.2, out=v)
        self.v = v
        self.s_last = spikes.astype(np.float32)
        # traces normalized to [0,1] (tau = 100 ms), rate EMA
        dec = np.float32(1.0 - dt / 0.1)
        self.pre_tr = self.pre_tr * dec + self.s_last * (1.0 - dec)
        self.post_tr = self.post_tr * dec + self.s_last * (1.0 - dec)
        self.rate += np.float32(0.06) * (self.s_last / dt - self.rate)
        # ---- dynamic remodeling: reward-gated three-factor plasticity ----
        if self.pending_reward != 0.0 and self.plasticity_on and len(self.pl_w0):
            r = self.pending_reward
            dm = 1.0 * r * self.pre_tr[self.pl_pre] * self.post_tr[self.pl_post]
            dm = np.clip(dm, -0.3, 0.3).astype(np.float32)
            self.pl_m = np.clip(self.pl_m + dm, -0.9, 4.0)
            changed = int((np.abs(dm) > 1e-3).sum())
            self.stats["remodel_events"] += 1
            self.stats["syn_changed"] += changed
            self.stats["dw_total"] += float(np.abs(dm).sum())
        self.pending_reward = 0.0
    def health(self):
        """cheap live diagnostics: is the network actually working"""
        r = self.rate
        return {"rate_mean": round(float(r.mean()), 3),
                "spike_frac": round(float((r > 1.0).mean()), 4),
                "v_sat_frac": round(float((np.abs(self.v) > 0.95).mean()), 4),
                "pl_abs_mean": round(float(np.abs(self.pl_m).mean()), 4) if len(self.pl_m) else 0.0,
                "pl_changed": int((np.abs(self.pl_m) > 0.05).sum()),
                "rec_scale": round(float(self.scale), 3)}

    # ---- readout --------------------------------------------------------------
    def bucket_rates(self, acc, n_ticks):
        """acc: float32 array (N,) accumulated spikes over the window.
        Per bucket: MBON readout population (plasticity lives here) + its DN group.
        Returns Hz-like rates, shape (n_buckets,)."""
        if n_ticks == 0:
            return np.zeros(len(self.readout_groups), dtype=np.float32)
        T = n_ticks * self.dt
        out = np.zeros(len(self.readout_groups), dtype=np.float32)
        for k, mb in enumerate(self.readout_groups):
            s = 0.0
            if len(mb):
                s += float(acc[mb].mean()) / T
            if k < len(self.out_groups) and len(self.out_groups[k]):
                s += 0.5 * float(acc[self.out_groups[k]].mean()) / T
            out[k] = s
        return out

    # ---- persistence: everything learned lives in pl_m (+ the codebook identity) ----
    # The original flygo_data.npz is NEVER written by the simulator; state files are
    # separate npz files under saves/.
    def save_state(self, path):
        np.savez_compressed(path,
                            pl_m=self.pl_m,
                            kc_groups=self.kc_groups,
                            seed=np.int64(self.seed),
                            stats_json=np.array(json.dumps(self.stats)))

    def load_state(self, path):
        z = np.load(path, allow_pickle=False)
        if len(z["pl_m"]) != len(self.pl_m):
            raise ValueError("state file does not match this connectome build")
        self.pl_m = z["pl_m"].astype(np.float32).copy()
        kg = z["kc_groups"].astype(np.int64)
        if kg.shape == self.kc_groups.shape:
            self.kc_groups = kg        # keep bucket->KC identity stable across restarts
            self._build_readout()      # readout MBONs follow the restored codebook
        self.stats = json.loads(str(z["stats_json"]))

    def reset_plasticity(self):
        """wipe everything learned, back to the pristine connectome"""
        self.pl_m = np.zeros(len(self.pl_w0), dtype=np.float32)
        self.stats = {"rewards": 0, "remodel_events": 0, "syn_changed": 0, "dw_total": 0.0, "last_reward": 0.0}
