# build_proprio.py : build real mechanosensory/proprioceptive pools + full sensory pools
# from the MaleCNS v1.0 annotations, for per-fly sensory interfaces and body->brain feedback.
# Writes flygo_proprio.npz (flygo_data.npz is never touched).
# Pools (indices into the flygo_data.npz neuron order, i.e. the simulation's indices):
#   prop_ch   SNch*     chordotonal organs  (stretch/position -> true proprioception)
#   prop_touch SNta*    tarsal contact afferents (leg-tip touch)
#   prop_jo   JO-*      Johnston's organ (antennal wind/airspeed mechanosensation)
#   prop_asc  sensory_ascending  VNC->brain interoceptive highway
#   vis_pool  ol_sensory/visual_projection/visual_centrifugal (per-fly input redraw)
#   dn_pool   descending_neuron (per-fly output redraw)
import pyarrow.feather as F
import numpy as np
import re, json

ANN = r"G:\guoying\datasets\male-cns-v1.0\body-annotations-male-cns-v1.0-minconf-0.5.feather"
NPZ = r"G:\guoying\flygo\flygo_data.npz"
OUT = r"G:\guoying\flygo\flygo_proprio.npz"

z = np.load(NPZ)
ids = z["ids"]                                     # simulation neuron order (bodyId sorted)
idx_of = {int(b): i for i, b in enumerate(ids)}

a = F.read_table(ANN, columns=["bodyId", "type", "superclass"])
bids = a.column("bodyId").to_numpy()
types = np.array([t or "" for t in a.column("type").to_pylist()])
sups = np.array([s or "" for s in a.column("superclass").to_pylist()])

def pool(mask):
    return np.array(sorted(idx_of[int(b)] for b in bids[mask] if int(b) in idx_of), dtype=np.int64)

prop_ch    = pool(sups == "vnc_sensory")                                       # all VNC sensory afferents
prop_touch = pool((sups == "vnc_sensory") & np.array([t.startswith("SNta") for t in types]))
prop_jo    = pool(np.array([t.startswith("JO") for t in types]))
prop_asc   = pool(sups == "sensory_ascending")
vis_pool   = pool(np.isin(sups, ["ol_sensory", "visual_projection", "visual_centrifugal", "ol_intrinsic"]))
dn_pool    = pool(sups == "descending_neuron")

np.savez_compressed(OUT,
                    prop_ch=prop_ch, prop_touch=prop_touch, prop_jo=prop_jo,
                    prop_asc=prop_asc, vis_pool=vis_pool, dn_pool=dn_pool)
rep = {"prop_ch": len(prop_ch), "prop_touch": len(prop_touch), "prop_jo": len(prop_jo),
       "prop_asc": len(prop_asc), "vis_pool": len(vis_pool), "dn_pool": len(dn_pool)}
print(json.dumps(rep))
