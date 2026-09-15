# build_fly_model.py : pack the flybody anatomical fly into one binary blob for the web scene.
# Assembly is delegated to MuJoCo itself (mesh COM/principal-axis processing + geom poses),
# so every part lands exactly where flybody renders it. Vertices are exported in each rigid
# body's local frame; the body tree (pos/quat) is exported for animation.
# Output: flygo/fly_model.bin
#   layout: [uint32 json_len][json rig][float32/uint32 payload]
#   json: {nodes:[{name,parent,pos,quat,geoms:[meshIdx]}], meshes:[{name,off,nV,offI,nI,rgba,wing}],
#          scale, center, bbox}  — quat in three.js order (x,y,z,w)
import numpy as np
import json, struct
from pathlib import Path
import mujoco

SRC = Path(r"G:\guoying\repos\flybody\flybody\fruitfly\assets")
OUT = Path(r"G:\guoying\flygo\fly_model.bin")
TARGET_LEN = 2.6          # world units for the largest extent (wingspan)

m = mujoco.MjModel.from_xml_path(str(SRC / "fruitfly.xml"))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

def vertex_normals(V, F):
    N = np.zeros_like(V)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(b - a, c - a)                       # area-weighted
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    ln = np.linalg.norm(N, axis=1, keepdims=True)
    return N / np.maximum(ln, 1e-12)

# ---- nodes: MuJoCo bodies (parents always precede children) -------------------
nodes, body_index = [], {}
for bid in range(1, m.nbody):
    pid = int(m.body_parentid[bid])
    q = m.body_quat[bid]                              # (w,x,y,z)
    nodes.append({"name": m.body(bid).name,
                  "parent": m.body(pid).name if pid != 0 else None,
                  "pos": m.body_pos[bid].tolist(),
                  "quat": [float(q[1]), float(q[2]), float(q[3]), float(q[0])],
                  "geoms": []})
    body_index[bid] = len(nodes) - 1

# ---- meshes: processed vertices -> world (mj_forward) -> body-local ----------------
mesh_list, blob = [], bytearray()
gmin = np.full(3, 1e9); gmax = np.full(3, -1e9)
for g in range(m.ngeom):
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
    mid = int(m.geom_dataid[g]); bid = int(m.geom_bodyid[g])
    va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
    fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
    V = np.asarray(m.mesh_vert[va:va+vn], dtype=np.float64)
    F = np.asarray(m.mesh_face[fa:fa+fn], dtype=np.int64)
    Rg = d.geom_xmat[g].reshape(3, 3); tg = d.geom_xpos[g]
    W = V @ Rg.T + tg                                 # world (rest pose)
    gmin = np.minimum(gmin, W.min(0)); gmax = np.maximum(gmax, W.max(0))
    Rb = d.xmat[bid].reshape(3, 3); tb = d.xpos[bid]
    L = (W - tb) @ Rb                                 # body-local: Rb^T (W - tb)
    # dedupe the triangle soup (shared vertices -> smooth normals, ~5x smaller payload)
    uniq, inv = np.unique(np.round(L, 6), axis=0, return_inverse=True)
    L = uniq; F = inv.reshape(-1)[F]
    N = vertex_normals(L, F)
    verts = np.concatenate([L, N], axis=1).astype(np.float32)
    index = F.astype(np.uint32).ravel()
    matid = int(m.geom_matid[g])
    rgba = (m.mat_rgba[matid] if matid >= 0 else m.geom_rgba[g]).astype(float).tolist()
    name = m.mesh(mid).name
    while len(blob) % 4: blob += b'\x00'
    off = len(blob); blob += verts.tobytes()
    while len(blob) % 4: blob += b'\x00'
    offI = len(blob); blob += index.tobytes()
    mesh_list.append({"name": name, "body": m.body(bid).name, "off": off, "nV": int(len(verts)),
                      "offI": offI, "nI": int(len(index)), "rgba": rgba, "wing": "wing" in name.lower()})
    nodes[body_index[bid]]["geoms"].append(len(mesh_list) - 1)

extent = float((gmax - gmin).max())
scale = TARGET_LEN / extent
center = ((gmin + gmax) / 2).tolist()
print("fly bbox", gmin.round(4), gmax.round(4), "extent", round(extent, 4), "-> scale", round(scale, 4))
print("nodes", len(nodes), "meshes", len(mesh_list), "blob MB", round(len(blob) / 1e6, 2))
hdr = {"nodes": nodes, "meshes": mesh_list, "scale": scale, "center": center,
       "bbox": [gmin.tolist(), gmax.tolist()]}
hj = json.dumps(hdr).encode()
OUT.write_bytes(struct.pack("<I", len(hj)) + hj + bytes(blob))
print("wrote", OUT, round(OUT.stat().st_size / 1e6, 2), "MB")
