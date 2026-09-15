# prepare_data.py : build the simulation bundle from the MaleCNS flat connectome.
# All input/output paths are plain literal constants under G:\guoying\flygo.
# Output: flygo_data.npz + prepare_report.json
import pyarrow.feather as F
import numpy as np
import scipy.sparse as sp
import json, time, re, os

ANN   = r"G:\guoying\datasets\male-cns-v1.0\body-annotations-male-cns-v1.0-minconf-0.5.feather"
NTF   = r"G:\guoying\datasets\male-cns-v1.0\body-neurotransmitters-male-cns-v1.0.feather"
CW    = r"G:\guoying\datasets\male-cns-v1.0\connectome-weights-male-cns-v1.0-minconf-0.5.feather"
SYNP  = r"G:\guoying\datasets\male-cns-v1.0\syn-points-male-cns-v1.0-minconf-0.5.feather"
NPZ   = r"G:\guoying\flygo\flygo_data.npz"
REPORT = r"G:\guoying\flygo\prepare_report.json"
os.makedirs(r"G:\guoying\flygo", exist_ok=True)
assert os.path.realpath(r"G:\guoying\flygo").startswith(os.path.realpath(r"G:\guoying"))

t0 = time.time()
def log(m): print("[%6.1fs] %s" % (time.time()-t0, m), flush=True)

# ---------- 1. neurons ----------
log("reading annotations ...")
a = F.read_table(ANN, columns=["bodyId", "type", "superclass"])
sup = a.column("superclass").to_pylist()
keep = np.array([s is not None for s in sup])
ids_all = a.column("bodyId").to_numpy()
ids = np.sort(ids_all[keep])
type_of = {int(b): (t or "") for b, t in zip(ids_all.tolist(), a.column("type").to_pylist())}
sup_of  = {int(b): (s or "") for b, s in zip(ids_all.tolist(), sup)}
N = len(ids)
log("retained neurons: %d" % N)
idx_of = {int(b): i for i, b in enumerate(ids)}

def pool_by_type(patterns, limit=100000):
    pat = re.compile(patterns, re.I)
    return np.array([idx_of[int(b)] for b in ids if pat.match(type_of.get(int(b), ""))][:limit], dtype=np.int64)

dan  = pool_by_type(r"^(PPL|PPM|SAG|DAN)")
kc   = pool_by_type(r"^KC")
mbon = pool_by_type(r"^MBON")
vis_sup = {"ol_intrinsic", "visual_projection", "ol_sensory", "visual_centrifugal"}
vis = np.array([i for i, b in enumerate(ids) if sup_of.get(int(b), "") in vis_sup], dtype=np.int64)
dn  = np.array([i for i, b in enumerate(ids) if sup_of.get(int(b), "") == "descending_neuron"], dtype=np.int64)
log("pools: DAN=%d KC=%d MBON=%d visual=%d DN=%d" % (len(dan), len(kc), len(mbon), len(vis), len(dn)))

rng = np.random.default_rng(42)
NG, NI, NO = 64, 24, 16
input_pool  = rng.choice(vis, min(NG*NI, len(vis)), replace=False)
output_pool = rng.choice(dn,  min(NG*NO, len(dn)), replace=False)
in_groups  = input_pool[:NG*NI].reshape(NG, NI).astype(np.int64)
out_groups = output_pool[:NG*NO].reshape(NG, NO).astype(np.int64)

# ---------- 2. neurotransmitter polarity ----------
log("reading neurotransmitters ...")
ntt = F.read_table(NTF, columns=["body", "predicted_nt"])
nb = ntt.column("body").to_numpy()
nn = np.array([x or "" for x in ntt.column("predicted_nt").to_pylist()])
order = np.argsort(nb, kind="stable")
nb, nn = nb[order], nn[order]
uniq_b, first_i = np.unique(nb, return_index=True)
sign_map = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "glycine": -1.0}
sign = np.full(N, 0.5, dtype=np.float32)
sel = np.clip(np.searchsorted(ids, uniq_b), 0, N-1)
hit = ids[sel] == uniq_b
vals = np.array([sign_map.get(x, 0.5) for x in nn[first_i]], dtype=np.float32)
sign[sel[hit]] = vals[hit]
log("nt polarity assigned: %d neurons" % int(hit.sum()))

# ---------- 3. connectome ----------
log("reading connectome (1.5e8 rows, filter to retained) ...")
cw = F.read_table(CW)
pre  = cw.column("body_pre").to_numpy()
post = cw.column("body_post").to_numpy()
w    = cw.column("weight").to_numpy()
del cw
m = np.isin(pre, ids) & np.isin(post, ids)
pre, post, w = pre[m], post[m], w[m]
log("retained edges: %d" % len(pre))
pi = np.searchsorted(ids, pre); qi = np.searchsorted(ids, post)
signed = w.astype(np.float32) * sign[pi]
del pre, post, w, m
W = sp.csr_matrix(sp.coo_matrix((signed, (qi.astype(np.int32), pi.astype(np.int32))), shape=(N, N)))
W.data = W.data.astype(np.float32)
W.sum_duplicates()
log("csr nnz: %d" % W.nnz)

# kc->mbon plasticity subset
kcmb_pre = np.array([], dtype=np.int64); kcmb_post = np.array([], dtype=np.int64); kcmb_w = np.array([], dtype=np.float32)
if len(kc) and len(mbon):
    kcset = np.zeros(N, dtype=bool); kcset[kc] = True
    mbset = np.zeros(N, dtype=bool); mbset[mbon] = True
    coo = W.tocoo()
    mm = kcset[coo.row] & mbset[coo.col]
    kcmb_post = coo.row[mm].astype(np.int64)
    kcmb_pre  = coo.col[mm].astype(np.int64)
    kcmb_w    = coo.data[mm].astype(np.float32)
    if len(kcmb_w) > 500000:
        selm = rng.choice(len(kcmb_w), 500000, replace=False)
        kcmb_pre, kcmb_post, kcmb_w = kcmb_pre[selm], kcmb_post[selm], kcmb_w[selm]
log("KC->MBON eligible synapses: %d" % len(kcmb_w))

# ---------- 4. positions ----------
log("reading syn-points columns (real centroids) ...")
pos = np.zeros((N, 3), dtype=np.float32)
have_pos = np.zeros(N, dtype=bool)
try:
    spf = F.read_table(SYNP, columns=["body", "x", "y", "z"])
    sb = spf.column("body").to_numpy()
    sx = spf.column("x").to_numpy().astype(np.float64)
    sy = spf.column("y").to_numpy().astype(np.float64)
    sz = spf.column("z").to_numpy().astype(np.float64)
    del spf
    km = np.isin(sb, ids)
    sb, sx, sy, sz = sb[km], sx[km], sy[km], sz[km]
    ui = np.searchsorted(ids, sb)
    cnt = np.bincount(ui, minlength=N).astype(np.float64)
    for c, arr in ((0, sx), (1, sy), (2, sz)):
        pos[:, c] = (np.bincount(ui, weights=arr, minlength=N) / np.maximum(cnt, 1)).astype(np.float32)
    have_pos = cnt > 0
    log("centroids for %d neurons" % int(have_pos.sum()))
except Exception as e:
    log("syn-points failed (%s) -> synthetic layout" % e)
missing = ~have_pos
if missing.any():
    th = rng.uniform(0, 2*np.pi, missing.sum()); ph = np.arccos(rng.uniform(-1, 1, missing.sum()))
    r = 0.75 + 0.25 * rng.random(missing.sum())
    pos[missing, 0] = (r*np.sin(ph)*np.cos(th)).astype(np.float32)
    pos[missing, 1] = (r*np.sin(ph)*np.sin(th)).astype(np.float32)
    pos[missing, 2] = (r*np.cos(ph)).astype(np.float32)
    log("synthetic fill for %d" % int(missing.sum()))
lo, hi = pos.min(0), pos.max(0)
pos = (pos - (lo+hi)/2) / ((hi-lo).max()/2)

# ---------- 5. save ----------
np.savez_compressed(NPZ,
    ids=ids, sign=sign, pos=pos,
    W_data=W.data, W_indices=W.indices, W_indptr=W.indptr, W_shape=np.array(W.shape),
    in_groups=in_groups, out_groups=out_groups,
    dan=dan, kc=kc, mbon=mbon,
    kcmb_pre=kcmb_pre, kcmb_post=kcmb_post, kcmb_w=kcmb_w)
report = {"neurons": int(N), "edges": int(W.nnz), "dan": int(len(dan)), "kc": int(len(kc)),
          "mbon": int(len(mbon)), "visual": int(len(vis)), "dn": int(len(dn)),
          "kcmb_synapses": int(len(kcmb_w)), "with_centroid": int(have_pos.sum())}
with open(REPORT, "w") as fh:
    json.dump(report, fh, indent=1)
log("DONE " + json.dumps(report))
