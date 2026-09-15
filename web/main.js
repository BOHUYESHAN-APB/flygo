// main.js — 双果蝇连接组对弈 3D (v4)
// - central board; two anatomical flies (flybody meshes + MJCF rig, assembled by MuJoCo)
// - stone placement: take off -> pick a stone from the dish -> fly to the target on a path
//   steered LIVE by the fly's own neural activity (/api/steer) -> release -> return.
//   Moves are QUEUED: an animation is never interrupted, so nothing ever teleports.
// - corner views: the decision circuit as a crisp NETWORK (nodes = the neurons we stimulate /
//   read out, wires = every real synapse between them); spikes light nodes + their wires.
const $ = id => document.getElementById(id);
const clamp = THREE.MathUtils.clamp;
const lerp = (a, b, t) => a + (b - a) * t;
const ease = t => t < 0.5 ? 2*t*t : 1 - Math.pow(-2*t + 2, 2) / 2;
const status = msg => { const p = $('phase'); if (p) p.textContent = msg; };

// ================= autonomy level =================
// '1' = Tier 1 (第一重·闭环): the BRAIN decides where to play; the body is executed by
//       deterministic code (tool-call style), with reflex fallbacks guaranteeing arrival,
//       grasp and placement. The game never stalls.
// '2' = Tier 2 (第二重·神经主导): the same decision, but the body is driven by the fly's
//       OWN motor signals — homing is weak, arrival is a "region reached" check, and
//       grasp/place trigger ONLY on the fly's tarsal burst; failure re-approaches (max 4
//       tries) before the disclosed Tier-1 fallback rescues the board.
let AUTO = '1';
try { AUTO = localStorage.getItem('flygo.autolvl') === '2' ? '2' : '1'; } catch(e){}

// shared radial-gradient sprite for glow dots (halos) and travelling synapse pulses
function makeSparkTex(){
  const c = document.createElement('canvas'); c.width = c.height = 64;
  const g = c.getContext('2d');
  const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0.0, 'rgba(255,255,255,1)');
  gr.addColorStop(0.3, 'rgba(255,255,255,0.55)');
  gr.addColorStop(1.0, 'rgba(255,255,255,0)');
  g.fillStyle = gr; g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}
const SPARK_TEX = makeSparkTex();
const N_PULSE = 72;                     // travelling light packets per brain
const NODE_R = 0.012;                   // decision-circuit sphere radius: small crisp dots
const NODE_IDLE = 0.90;                 // idle brightness — clearly visible class-hued dots

// ================= configurable physics / neural-mapping parameters =================
// [key, label, min, max, step, default, server?]  — persisted in localStorage; server keys are
// pushed to /api/control cmd=phys so the neural mapping gains live in the simulation too.
const PHYS_SPEC = [
  ['speed',       '飞行速度',           3,   16,  0.5,  8.5],
  ['turnGain',    '归航转向增益',       0.5, 8,   0.1,  2.5],
  ['neuralYaw',   '神经偏航增益',       0,   6,   0.1,  2.6],
  ['homingRamp',  '归航斜率 (s)',       0.5, 6,   0.1,  2.2],
  ['cruise',      '巡航高度',           1.5, 6,   0.1,  2.9],
  ['dive',        '下降速率',           0.8, 6,   0.1,  2.6],
  ['wingHz',      '翅频 (Hz)',          4,   30,  1,    11],
  ['jointK',      '关节弹簧速率',       2,   30,  1,    10],
  ['groomGain',   '梳理增益',           0,   2,   0.05, 1.0],
  ['attnGain',    '棋盘注意力权重',     0,   1,   0.05, 0.7],
  ['near_current','[服] 接近感觉电流',  0,   1,   0.02, 0.30, true],
  ['burst_gain',  '[服] 动作爆发增益',  0.5, 5,   0.1,  2.5,  true],
  ['drive_gain',  '[服] 多巴胺驱动增益',0,   1,   0.05, 0.35, true],
  ['drive_decay', '[服] 驱动衰减/手',   0.9, 1,   0.005,0.985,true],
];
const PHYS = {};
for (const s of PHYS_SPEC) PHYS[s[0]] = s[5];
try { Object.assign(PHYS, JSON.parse(localStorage.getItem('flygo.phys') || '{}')); } catch(e){}
function savePhys(){ try { localStorage.setItem('flygo.phys', JSON.stringify(PHYS)); } catch(e){} }
function buildPhysPanel(){
  const host = $('physRows'); if (!host) return;
  host.innerHTML = PHYS_SPEC.map(s =>
    `<div class="prow"><span>${s[1]}</span><input type="range" data-k="${s[0]}" min="${s[2]}" max="${s[3]}" step="${s[4]}" value="${PHYS[s[0]]}"><b id="pv_${s[0]}">${(+PHYS[s[0]]).toFixed(3).replace(/\.?0+$/,'')}</b></div>`).join('');
  host.querySelectorAll('input').forEach(inp => {
    inp.oninput = () => { const k = inp.dataset.k; PHYS[k] = +inp.value; $('pv_'+k).textContent = (+inp.value).toFixed(3).replace(/\.?0+$/,''); };
    inp.onchange = () => { savePhys(); const sp = PHYS_SPEC.find(s => s[0] === inp.dataset.k); if (sp && sp[6]) ctl('phys', {key: sp[0], value: PHYS[sp[0]]}); };
  });
  const rb = $('physReset'); if (rb) rb.onclick = () => { for (const s of PHYS_SPEC) PHYS[s[0]] = s[5]; savePhys(); buildPhysPanel(); pushServerPhys(); };
}
function pushServerPhys(){ for (const s of PHYS_SPEC) if (s[6]) ctl('phys', {key: s[0], value: PHYS[s[0]]}).catch(()=>{}); }

const renderer = new THREE.WebGLRenderer({canvas: $('c'), antialias: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x0b0e14);

// ================= main scene =================
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0e14);
scene.fog = new THREE.Fog(0x0b0e14, 44, 100);
const camera = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.1, 200);
camera.position.set(0, 16.5, 27.5); camera.lookAt(0, 4.6, 0);
scene.add(new THREE.AmbientLight(0x8fa0c0, 0.85));
const dl = new THREE.DirectionalLight(0xffffff, 1.0); dl.position.set(6, 14, 8); scene.add(dl);
const dl2 = new THREE.DirectionalLight(0x9db4ff, 0.4); dl2.position.set(-8, 10, -6); scene.add(dl2);

const SZ = 15, CELL = 1.0, OFF = -(SZ-1)*CELL/2;
const boardG = new THREE.Group(); scene.add(boardG);
{
  const slab = new THREE.Mesh(new THREE.BoxGeometry(SZ*CELL+1.2, 0.5, SZ*CELL+1.2),
                              new THREE.MeshStandardMaterial({color:0x7a5a34, roughness:0.85}));
  slab.position.y = -0.28; boardG.add(slab);
  const pts = [];
  for (let i = 0; i < SZ; i++){
    pts.push(OFF, 0.02, OFF+i*CELL, -OFF, 0.02, OFF+i*CELL);
    pts.push(OFF+i*CELL, 0.02, OFF, OFF+i*CELL, 0.02, -OFF);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  boardG.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({color:0x2e2317})));
}
const stoneGeo = new THREE.SphereGeometry(0.42, 20, 14);
const matB = new THREE.MeshStandardMaterial({color:0x101418, roughness:0.35, metalness:0.15});
const matW = new THREE.MeshStandardMaterial({color:0xe8ecef, roughness:0.4});
const boardPt = (x, y) => new THREE.Vector3(OFF+x*CELL, 0.35, OFF+y*CELL);
// stones are keyed by cell so nothing is ever re-created (no flicker); pending keys stay hidden
const stones = new Map();                       // key "x_y_p" -> mesh
const pendingKeys = new Set();
function syncStones(board){
  const want = new Set();
  for (let y = 0; y < SZ; y++) for (let x = 0; x < SZ; x++){
    const p = board[y][x]; if (!p) continue;
    const key = x+'_'+y+'_'+p; want.add(key);
    if (!stones.has(key)){
      const m = new THREE.Mesh(stoneGeo, p === 1 ? matB : matW);
      m.position.copy(boardPt(x, y)); m.visible = !pendingKeys.has(key); boardG.add(m); stones.set(key, m);
    }
  }
  for (const [key, m] of stones) if (!want.has(key)){ boardG.remove(m); stones.delete(key); pendingKeys.delete(key); }
}
const ring = new THREE.Mesh(new THREE.RingGeometry(0.5, 0.68, 28),
  new THREE.MeshBasicMaterial({color:0xffc24d, side:THREE.DoubleSide, transparent:true, opacity:0.95, blending:THREE.AdditiveBlending, depthWrite:false}));
ring.rotation.x = -Math.PI/2; ring.visible = false; scene.add(ring);
const beacon = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 3.2, 8),
  new THREE.MeshBasicMaterial({color:0xffc24d, transparent:true, opacity:0.28, blending:THREE.AdditiveBlending, depthWrite:false}));
beacon.visible = false; scene.add(beacon);
function setMarker(v, x, z){ ring.visible = beacon.visible = v; if (v){ ring.position.set(x, 1.02, z); beacon.position.set(x, 1.95, z); } }

// podiums + stone dishes
const PODIUM_TOP = 0.25;
const REST = [new THREE.Vector3(-10.9, 0, 3.2), new THREE.Vector3(10.9, 0, 3.2)];
const DISH = [new THREE.Vector3(-10.9, 0.33, 0.1), new THREE.Vector3(10.9, 0.33, 0.1)];
{
  const pm = new THREE.MeshStandardMaterial({color:0x1c2333, roughness:0.8});
  const dm = new THREE.MeshStandardMaterial({color:0x3a2f22, roughness:0.7});
  for (let f = 0; f < 2; f++){
    const p = new THREE.Mesh(new THREE.CylinderGeometry(3.1, 3.4, 0.6, 28), pm);
    p.position.set(REST[f].x, -0.05, 1.9); scene.add(p);
    const dish = new THREE.Mesh(new THREE.CylinderGeometry(1.25, 1.1, 0.16, 24), dm);
    dish.position.copy(DISH[f]); scene.add(dish);
    for (let k = 0; k < 4; k++){
      const a = k * Math.PI/2 + 0.4;
      const s = new THREE.Mesh(stoneGeo, f===0 ? matB : matW);
      s.position.set(DISH[f].x + Math.cos(a)*0.55, 0.73, DISH[f].z + Math.sin(a)*0.55); scene.add(s);
    }
  }
}

// ---------- glass box: the flies may fly anywhere inside; touching the glass is punished ----------
const BOX = {x: 15.5, z: 12.5, y0: 0.0, y1: 9.0};
{
  const glass = new THREE.Mesh(new THREE.BoxGeometry(BOX.x*2, BOX.y1 - BOX.y0, BOX.z*2),
    new THREE.MeshStandardMaterial({color: 0x9fd0ff, transparent: true, opacity: 0.045, roughness: 0.1, metalness: 0.0,
                                    side: THREE.BackSide, depthWrite: false}));
  glass.position.set(0, (BOX.y0 + BOX.y1)/2, 0); scene.add(glass);
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(BOX.x*2, BOX.y1 - BOX.y0, BOX.z*2)),
    new THREE.LineBasicMaterial({color: 0x5a86b8, transparent: true, opacity: 0.55}));
  edges.position.copy(glass.position); scene.add(edges);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(BOX.x*2, BOX.z*2),
    new THREE.MeshStandardMaterial({color: 0x0f1420, roughness: 0.95, transparent: true, opacity: 0.85}));
  floor.rotation.x = -Math.PI/2; floor.position.y = -0.36; scene.add(floor);
}
const lastHit = [0, 0];
function confine(fl, f, now){
  const p = fl.g.position, m = 0.9; let hit = false;
  if (p.x >  BOX.x - m){ p.x =  BOX.x - m; hit = true; }
  if (p.x < -BOX.x + m){ p.x = -BOX.x + m; hit = true; }
  if (p.z >  BOX.z - m){ p.z =  BOX.z - m; hit = true; }
  if (p.z < -BOX.z + m){ p.z = -BOX.z + m; hit = true; }
  if (p.y >  BOX.y1 - m){ p.y = BOX.y1 - m; hit = true; }
  if (p.y <  BOX.y0 + 0.7 && !(anim && (anim.state === 'grab' || anim.state === 'place' || anim.state === 'land' || anim.state === 'aim'))){ p.y = BOX.y0 + 0.7; hit = true; }
  if (hit){
    fl.heading = Math.atan2(-(0 - p.z), 0 - p.x) + (Math.random() - 0.5) * 0.8;   // bounce back toward the arena
    if (now - lastHit[f] > 1200){ lastHit[f] = now; fl.halo.material.color.setHex(0xff4060); fl.haloT = now; ctl('penalty', {fly: f}).catch(()=>{}); }
  }
  return hit;
}
let FLYMODEL = null;
const flies = [null, null];
const _q = new THREE.Quaternion(), _e = new THREE.Euler(), _v = new THREE.Vector3();
async function loadFlyModel(){
  const ab = await (await fetch('/fly_model.bin')).arrayBuffer();
  const jl = new DataView(ab).getUint32(0, true);
  const hdr = JSON.parse(new TextDecoder().decode(new Uint8Array(ab, 4, jl)));
  const base = 4 + jl;
  const geos = hdr.meshes.map(m => {
    const g = new THREE.BufferGeometry();
    const v = new Float32Array(ab.slice(base + m.off, base + m.off + m.nV*24));
    const ib = new THREE.InterleavedBuffer(v, 6);
    g.setAttribute('position', new THREE.InterleavedBufferAttribute(ib, 3, 0));
    g.setAttribute('normal', new THREE.InterleavedBufferAttribute(ib, 3, 3));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(ab.slice(base + m.offI, base + m.offI + m.nI*4)), 1));
    return g;
  });
  FLYMODEL = {hdr, geos};
}
function insertPivot(node){
  const parent = node.g.parent;
  const piv = new THREE.Group(); piv.position.copy(node.g.position);
  parent.remove(node.g); node.g.position.set(0, 0, 0); piv.add(node.g); parent.add(piv);
  return piv;
}
function buildFly(f){
  const {hdr, geos} = FLYMODEL;
  const nodes = {};
  const root = new THREE.Group();
  for (const n of hdr.nodes){
    const g = new THREE.Group(); g.name = n.name;
    if (n.pos) g.position.set(n.pos[0], n.pos[1], n.pos[2]);
    if (n.quat) g.quaternion.set(n.quat[0], n.quat[1], n.quat[2], n.quat[3]);
    for (const gi of n.geoms){
      const m = hdr.meshes[gi];
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(m.rgba[0], m.rgba[1], m.rgba[2]), roughness: m.wing ? 0.25 : 0.55, metalness: 0.06,
        transparent: m.rgba[3] < 1, opacity: m.rgba[3], side: m.wing ? THREE.DoubleSide : THREE.FrontSide,
        depthWrite: m.rgba[3] >= 1});
      g.add(new THREE.Mesh(geos[gi], mat));
    }
    nodes[n.name] = {g, rest: g.quaternion.clone()};
    (n.parent ? nodes[n.parent].g : root).add(g);
  }
  const S = hdr.scale * 2.35;
  root.scale.setScalar(S);
  root.position.set(-hdr.center[0]*S, -hdr.center[1]*S, -hdr.center[2]*S);
  const zup = new THREE.Group(); zup.rotation.x = -Math.PI/2; zup.add(root);   // MJCF z-up -> three y-up
  const outer = new THREE.Group(); outer.add(zup);
  const head = insertPivot(nodes.head);
  const wings = [insertPivot(nodes.wing_left), insertPivot(nodes.wing_right)];
  const legsT1 = ['left', 'right'].map(side => ({
    coxa: nodes['coxa_T1_'+side], femur: nodes['femur_T1_'+side], tibia: nodes['tibia_T1_'+side], s: side==='left' ? 1 : -1}));
  const legsT3 = ['left', 'right'].map(side => ({
    coxa: nodes['coxa_T3_'+side], femur: nodes['femur_T3_'+side], tibia: nodes['tibia_T3_'+side], s: side==='left' ? 1 : -1}));
  scene.add(outer); outer.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(outer);
  REST[f].y = PODIUM_TOP - box.min.y;
  outer.position.copy(REST[f]);
  outer.rotation.y = f === 0 ? 0 : Math.PI;   // model forward = +x
  const halo = new THREE.Mesh(new THREE.TorusGeometry(1.5, 0.07, 8, 32),
    new THREE.MeshBasicMaterial({color:0x33ff73, transparent:true, opacity:0, blending:THREE.AdditiveBlending, depthWrite:false}));
  halo.rotation.x = Math.PI/2; halo.position.y = box.min.y + 0.15; halo.visible = false; outer.add(halo);
  return {g: outer, nodes, head, wings, legsT1, legsT3, halo, haloT: -1e9, carryY: box.min.y + 0.62, heading: f===0 ? 0 : Math.PI, box};
}
function setDelta(node, ex, ey, ez){ node.g.quaternion.copy(node.rest).multiply(_q.setFromEuler(_e.set(ex, ey, ez))); }

// ================= connectome corner views: decision-circuit network =================
const brainSc = [new THREE.Scene(), new THREE.Scene()];
for (const sc of brainSc) sc.background = new THREE.Color(0x070a10);
const brainRoot = [new THREE.Group(), new THREE.Group()];
brainSc[0].add(brainRoot[0]); brainSc[1].add(brainRoot[1]);
const brainCams = [new THREE.PerspectiveCamera(40, 1, 0.01, 20), new THREE.PerspectiveCamera(40, 1, 0.01, 20)];
// dedicated renderers for the corner brain views. The main canvas sits BEHIND the UI
// panels, whose translucent blurred background (backdrop-filter + 88% opacity) was
// dimming and blurring everything rendered into subviewports. A child canvas draws ON
// TOP of the panel background: fully crisp, full brightness.
const brainR = [null, null];
function setupBrainRender(f){
  const el = $('bv'+f); if (!el) return;
  const cv = document.createElement('canvas');
  cv.style.cssText = 'width:100%;height:100%;display:block;border-radius:7px';
  el.appendChild(cv);
  const r = new THREE.WebGLRenderer({canvas: cv, antialias: true});
  r.setClearColor(0x070a10);
  brainR[f] = r;
}
const _m4 = new THREE.Matrix4();
const brainView = [ {rx: -0.15, ry: 0.25, zoom: 1.0}, {rx: -0.15, ry: -0.25, zoom: 1.0} ];
const brainHome = [ {rx: -0.15, ry: 0.25, zoom: 1.0}, {rx: -0.15, ry: -0.25, zoom: 1.0} ];
const bvHasSave = [ !!localStorage.getItem('flygo.brainviewA0'), !!localStorage.getItem('flygo.brainviewA1') ];
for (let f = 0; f < 2; f++){
  try { const s = JSON.parse(localStorage.getItem('flygo.brainviewA'+f) || 'null'); if (s) Object.assign(brainView[f], s); } catch(e){}
}
const saveView = f => { try { localStorage.setItem('flygo.brainviewA'+f, JSON.stringify(brainView[f])); } catch(e){} };
let POS = null, POSB = null, META = null, CLS = null;
const CLASS_COLORS = [   // whole-CNS cloud; readable mid-tones so structure stays visible idle
  [0.38, 0.50, 0.74],  // 0 brain intrinsic (steel blue)
  [0.32, 0.42, 0.64],  // 1 VNC intrinsic
  [0.30, 0.52, 0.80],  // 2 visual input
  [1.00, 0.82, 0.25],  // 3 KC
  [1.00, 0.55, 0.18],  // 4 MBON
  [0.25, 1.00, 0.55],  // 5 DAN
  [0.60, 0.45, 0.90],  // 6 motor neurons
  [1.00, 0.37, 0.48],  // 7 descending
];
const net = [null, null];          // per fly: {nodeIdx, nodeOf, nodes, glow, base, baseStatic, wires, pre, post, w, preNode, wireBase, synWire, synM}
const cloudRef = [null, null];     // per fly: whole-CNS silver point cloud {pts, base} — flashes with activity
const lastAct = [0, 0];
function nodeClassColor(f, i){
  if (META.danS.has(i))        return [0.25, 1.0, 0.55];      // dopamine
  if (META.outS.has(i))        return [1.0, 0.55, 0.18];      // MBON readout (one per bucket)
  if (META.mbS.has(i))         return [0.72, 0.40, 0.16];     // other MBON
  if (net[f].kcS.has(i))       return [1.0, 0.82, 0.25];      // KC codebook
  if (META.inS.has(i))         return [0.30, 0.65, 1.0];      // sensory input
  return [0.60, 0.66, 0.78];
}
function nodeSizeMul(f, i){
  if (META.danS.has(i))        return 1.9;
  if (META.outS.has(i))        return 2.3;
  if (META.mbS.has(i))         return 1.0;
  if (net[f].kcS.has(i))       return 1.7;
  return 0.7;
}
async function loadConnectome(){
  const [pb, cb2, meta] = await Promise.all([fetch('/api/pos').then(r => r.arrayBuffer()),
                                             fetch('/api/classes').then(r => r.arrayBuffer()),
                                             fetch('/api/meta').then(r => r.json())]);
  POS = new Float32Array(pb); META = meta; CLS = new Uint8Array(cb2);
  META.inS = new Set(META.in_groups); META.danS = new Set(META.dan);
  META.mbS = new Set(META.out_groups); for (const m of (META.mbon || [])) META.mbS.add(m);
  META.outS = new Set(META.out_groups);
  $('nn').textContent = META.N.toLocaleString();
  const n = POS.length/3;
  let mn = [1e9,1e9,1e9], mx = [-1e9,-1e9,-1e9];
  for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++){ const v = POS[3*i+d]; if (v < mn[d]) mn[d] = v; if (v > mx[d]) mx[d] = v; }
  const ext = Math.max(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]) || 1;
  const mid = [(mn[0]+mx[0])/2, (mn[1]+mx[1])/2, (mn[2]+mx[2])/2];
  const roi = [].concat(META.kc0, META.kc1, META.dan, META.in_groups, META.out_groups);
  const cb = [0,0,0]; for (const i of roi) for (let d = 0; d < 3; d++) cb[d] += (POS[3*i+d]-mid[d])/ext; for (let d = 0; d < 3; d++) cb[d] /= roi.length;
  POSB = new Float32Array(POS.length);
  for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) POSB[3*i+d] = (POS[3*i+d]-mid[d])/ext - cb[d];
  // whole-CNS point cloud (complete structure, reference-video style: silver specks on
  // black; per-neuron activity flashes it white-hot) + scene lights
  for (let f = 0; f < 2; f++){
    const col = new Float32Array(n*3), base = new Float32Array(n*3);
    for (let i = 0; i < n; i++){
      const c = CLASS_COLORS[CLS[i]] || CLASS_COLORS[0];
      // mostly-white specks with a 22% class tint (keeps legend meaning, kills color mush)
      base[3*i]   = Math.min(1, 0.78*0.72 + 0.22*c[0]);
      base[3*i+1] = Math.min(1, 0.78*0.75 + 0.22*c[1]);
      base[3*i+2] = Math.min(1, 0.78*0.82 + 0.22*c[2]);
      col[3*i] = base[3*i]*0.55; col[3*i+1] = base[3*i+1]*0.55; col[3*i+2] = base[3*i+2]*0.55;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(POSB, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    const p = new THREE.Points(g, new THREE.PointsMaterial({size: 1.4, sizeAttenuation: false,
      vertexColors: true, transparent: true, opacity: 0.5, depthWrite: false}));
    p.renderOrder = 0; brainRoot[f].add(p);
    cloudRef[f] = {pts: p, base};
    brainSc[f].add(new THREE.AmbientLight(0xffffff, 0.85));
    const dl = new THREE.DirectionalLight(0xffffff, 0.55); dl.position.set(0.6, 1.0, 1.4);
    brainSc[f].add(dl);
  }
}
async function loadNet(f){
  const ab = await (await fetch('/api/net?fly='+f)).arrayBuffer();
  const hdr = new Int32Array(ab, 0, 2); const nN = hdr[0], nE = hdr[1];
  const nodeIdx = new Int32Array(ab.slice(8, 8 + 4*nN));
  const pre = new Int32Array(ab.slice(8 + 4*nN, 8 + 4*nN + 4*nE));
  const post = new Int32Array(ab.slice(8 + 4*nN + 4*nE, 8 + 4*nN + 8*nE));
  const w = new Float32Array(ab.slice(8 + 4*nN + 8*nE, 8 + 4*nN + 12*nE));
  const nodeOf = new Map(); for (let i = 0; i < nN; i++) nodeOf.set(nodeIdx[i], i);
  const N = {nodeIdx, nodeOf, pre, post, w, kcS: new Set(META['kc'+f])};
  net[f] = N;
  // ---- REAL anatomical layout ----
  // Node positions are the actual 3D positions of these neurons in the fly CNS
  // (synapse-point centroids, normalized to the whole-CNS extent and re-centered on the
  // decision-circuit centroid). This is the real volumetric brain, not a schematic.
  const lay = new Float32Array(nN*3);
  let roiR = 0.001;
  for (let i = 0; i < nN; i++){
    const k = nodeIdx[i];
    lay[3*i] = POSB[3*k]; lay[3*i+1] = POSB[3*k+1]; lay[3*i+2] = POSB[3*k+2];
    if (POS[3*k+2] > -0.42){                       // frame the BRAIN (skip VNC) by default
      const d = Math.hypot(lay[3*i], lay[3*i+1], lay[3*i+2]);
      if (d > roiR) roiR = d;
    }
  }
  // home view frames the real circuit volume, tight enough that individual nodes resolve;
  // used as the dblclick reset target
  brainHome[f] = {rx: -0.15, ry: f === 0 ? 0.25 : -0.25, zoom: clamp(roiR * 1.55, 0.4, 3.0)};
  if (!bvHasSave[f]) Object.assign(brainView[f], brainHome[f]);
  // decision-circuit neurons as real shaded spheres, MERGED into one geometry with
  // vertex colors (the most compatible path; per-frame color writes only for active nodes)
  const base = new Float32Array(nN*3), nsz = new Float32Array(nN);
  for (let i = 0; i < nN; i++){
    const k = nodeIdx[i];
    const c = nodeClassColor(f, k);          // full-saturation class hue
    base[3*i] = c[0]; base[3*i+1] = c[1]; base[3*i+2] = c[2];
    nsz[i] = nodeSizeMul(f, k);
  }
  N.baseStatic = base; N.base = base.slice(); N.glow = new Float32Array(nN); N.nsz = nsz; N.lay = lay;
  {
    const proto = new THREE.SphereGeometry(1, 8, 6);
    const pv = proto.attributes.position.array, pn = proto.attributes.normal.array, pi = proto.index.array;
    N.protoPos = pv;
    const V = pv.length / 3, T = pi.length;
    const pos = new Float32Array(nN*V*3), nor = new Float32Array(nN*V*3), col = new Float32Array(nN*V*3);
    const idx = new Uint32Array(nN*T);
    let vo = 0, io = 0;
    for (let i = 0; i < nN; i++){
      const s = NODE_R * nsz[i], cx = lay[3*i], cy = lay[3*i+1], cz = lay[3*i+2];
      for (let v = 0; v < V; v++){
        const o = (vo+v)*3;
        pos[o] = pv[v*3]*s + cx; pos[o+1] = pv[v*3+1]*s + cy; pos[o+2] = pv[v*3+2]*s + cz;
        nor[o] = pn[v*3]; nor[o+1] = pn[v*3+1]; nor[o+2] = pn[v*3+2];
        col[o] = base[3*i]*NODE_IDLE; col[o+1] = base[3*i+1]*NODE_IDLE; col[o+2] = base[3*i+2]*NODE_IDLE;
      }
      for (let t = 0; t < T; t++) idx[io+t] = pi[t] + vo;
      vo += V; io += T;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    N.sphereVerts = V;
    N.nodes = new THREE.Mesh(g, new THREE.MeshLambertMaterial({vertexColors: true}));
    N.nodes.renderOrder = 2; brainRoot[f].add(N.nodes);
  }
  // wires: real synapses. Structure-critical edges (KC->MBON, into readout MBONs, DAN edges)
  // are always kept; the rest fill up to the budget ranked by |w|.
  const keepScore = e => {
    let s = Math.abs(w[e]);
    if (N.kcS.has(pre[e])) s += 10;                        // plastic decision pathway
    if (META.outS.has(post[e])) s += 8;                    // into readout
    if (META.danS.has(pre[e]) || META.danS.has(post[e])) s += 6;
    return s;
  };
  const ord = Array.from({length: nE}, (_, e) => e).sort((a, b) => keepScore(b) - keepScore(a));
  const keepN = Math.min(nE, 6500);
  const kept = ord.slice(0, keepN);
  kept.sort((a, b) => a - b);
  const ne = kept.length;
  N.preNode = new Int32Array(ne); N.postNode = new Int32Array(ne);
  N.wireBase = new Float32Array(ne*3); N.wireAmp = new Float32Array(ne);
  for (let j = 0; j < ne; j++){
    const e = kept[j];
    N.preNode[j] = nodeOf.get(pre[e]);
    N.postNode[j] = nodeOf.get(post[e]);
    const exc = w[e] > 0;
    if (META.danS.has(pre[e])){ N.wireBase[3*j] = 0.1; N.wireBase[3*j+1] = 0.75; N.wireBase[3*j+2] = 0.35; }
    else if (exc)             { N.wireBase[3*j] = 0.55; N.wireBase[3*j+1] = 0.38; N.wireBase[3*j+2] = 0.16; }
    else                      { N.wireBase[3*j] = 0.16; N.wireBase[3*j+1] = 0.34; N.wireBase[3*j+2] = 0.62; }
    N.wireAmp[j] = 0.58 + Math.min(1, Math.abs(w[e]) * 4) * 0.50;
  }
  // neon tubes merged into ONE geometry with vertex colors (additive blending on the
  // dark background = glowing fiber look; WebGL caps plain lines at 1px = invisible)
  {
    const proto = new THREE.CylinderGeometry(1, 1, 1, 5, 1, true);
    const pv = proto.attributes.position.array, pi = proto.index.array;
    const V = pv.length / 3, T = pi.length;
    const pos = new Float32Array(ne*V*3), col = new Float32Array(ne*V*3);
    const idx = new Uint32Array(ne*T);
    const up = new THREE.Vector3(0, 1, 0), dir = new THREE.Vector3(), mid = new THREE.Vector3();
    const q = new THREE.Quaternion(), sc = new THREE.Vector3(), va = new THREE.Vector3(), vv = new THREE.Vector3();
    const m4t = new THREE.Matrix4();
    let vo = 0, io = 0;
    for (let j = 0; j < ne; j++){
      const na = nodeOf.get(pre[kept[j]]), nb = nodeOf.get(post[kept[j]]);
      va.set(lay[3*na], lay[3*na+1], lay[3*na+2]);
      vv.set(lay[3*nb], lay[3*nb+1], lay[3*nb+2]);
      dir.subVectors(vv, va);
      const len = Math.max(1e-4, dir.length());
      q.setFromUnitVectors(up, dir.normalize());
      mid.addVectors(va, vv).multiplyScalar(0.5);
      sc.set(0.006, len, 0.006);
      m4t.compose(mid, q, sc);
      for (let v = 0; v < V; v++){
        const o = (vo+v)*3;
        vv.set(pv[v*3], pv[v*3+1], pv[v*3+2]).applyMatrix4(m4t);
        pos[o] = vv.x; pos[o+1] = vv.y; pos[o+2] = vv.z;
        col[o] = 0; col[o+1] = 0; col[o+2] = 0;
      }
      for (let t = 0; t < T; t++) idx[io+t] = pi[t] + vo;
      vo += V; io += T;
    }
    const wg = new THREE.BufferGeometry();
    wg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    wg.setAttribute('color', new THREE.BufferAttribute(col, 3));
    wg.setIndex(new THREE.BufferAttribute(idx, 1));
    N.tubeVerts = V;
    N.wires = new THREE.Mesh(wg, new THREE.MeshBasicMaterial({vertexColors: true,
      blending: THREE.AdditiveBlending, transparent: true, depthWrite: false, toneMapped: false}));
    N.wires.renderOrder = 1; brainRoot[f].add(N.wires);
  }
  // travelling light pulses: additive glow dots that run along kept synapses, the classic
  // "signal packet" look (TensorSpace-style). Each respawn prefers an ACTIVE edge, so the
  // traffic concentrates on whatever the fly is thinking about right now.
  {
    const pos = new Float32Array(N_PULSE*3), col = new Float32Array(N_PULSE*3);
    N.pulseE = new Int32Array(N_PULSE); N.pulseT = new Float32Array(N_PULSE); N.pulseSp = new Float32Array(N_PULSE);
    for (let s = 0; s < N_PULSE; s++){
      N.pulseE[s] = (Math.random()*ne) | 0;
      N.pulseT[s] = Math.random();
      N.pulseSp[s] = 0.5 + Math.random()*1.1;
      pos[3*s] = lay[3*N.preNode[N.pulseE[s]]];
    }
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    pg.setAttribute('color', new THREE.BufferAttribute(col, 3));
    N.sparks = new THREE.Points(pg, new THREE.PointsMaterial({size: 9, sizeAttenuation: false,
      map: SPARK_TEX, vertexColors: true, blending: THREE.AdditiveBlending,
      transparent: true, depthWrite: false, toneMapped: false}));
    N.sparks.renderOrder = 3; brainRoot[f].add(N.sparks);
  }
  // per-node glow sprites: one additive halo point per neuron (position static = lay).
  // frame() writes its color from the live glow value -> soft bloom around firing nodes.
  {
    const pos = new Float32Array(nN*3), col = new Float32Array(nN*3);
    pos.set(lay);
    const hg = new THREE.BufferGeometry();
    hg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    hg.setAttribute('color', new THREE.BufferAttribute(col, 3));
    N.halos = new THREE.Points(hg, new THREE.PointsMaterial({size: 12, sizeAttenuation: false,
      map: SPARK_TEX, vertexColors: true, blending: THREE.AdditiveBlending,
      transparent: true, depthWrite: false, toneMapped: false}));
    N.halos.renderOrder = 4; brainRoot[f].add(N.halos);
  }
  // plastic KC->MBON overlay maps onto kept wires
  const key = new Map(); for (let j = 0; j < ne; j++){ const e = kept[j]; key.set(pre[e]*200000 + post[e], j); }
  N.wireKey = key; N.synWire = null; N.synM = null; N.synPre = null;
  paintWires(f, -1);
}
async function loadSyn(f){
  const N = net[f]; if (!N) return;
  const ab = await (await fetch('/api/syn?fly='+f)).arrayBuffer();
  const buf = new Int32Array(ab); const n = buf[0]; if (!n) return;
  const pre = buf.subarray(1, 1+n), post = buf.subarray(1+n, 1+2*n);
  const m = new Float32Array(ab.slice(4*(1+2*n)));
  if (!N.synWire){
    const sw = [], sp = [];
    for (let i = 0; i < n; i++){ const e = N.wireKey.get(pre[i]*200000 + post[i]); if (e !== undefined){ sw.push(e); sp.push(pre[i]); } }
    N.synWire = Int32Array.from(sw); N.synPre = Int32Array.from(sp); N.synM = new Float32Array(sw.length);
  }
  let j = 0;
  for (let i = 0; i < n; i++){ if (N.wireKey.has(pre[i]*200000 + post[i])) N.synM[j++] = m[i]; }
}
// edge tubes: base color by class/sign, lit by presynaptic activity; KC->MBON overlay by
// plasticity m; active bucket's codebook synapses turn gold. Unlit vertex colors: color = brightness.
function paintWires(f, bucket){
  const N = net[f]; if (!N || !N.wires) return;
  const col = N.wires.geometry.attributes.color.array, V = N.tubeVerts, nE = N.preNode.length, gl = N.glow;
  for (let e = 0; e < nE; e++){
    const act = gl[N.preNode[e]];
    const a = Math.min(3.0, N.wireAmp[e] * (1 + 4.0 * act) + act * 1.2);
    const r = Math.min(1.6, N.wireBase[3*e]   * a + act * 0.70);
    const g = Math.min(1.6, N.wireBase[3*e+1] * a + act * 0.62);
    const b = Math.min(1.6, N.wireBase[3*e+2] * a + act * 0.42);
    const o = e*V*3;
    for (let v = 0; v < V; v++){ col[o+v*3] = r; col[o+v*3+1] = g; col[o+v*3+2] = b; }
  }
  if (N.synWire){
    const active = new Set();
    const arr = META['kc'+f]; if (bucket >= 0 && arr.length >= 2*(bucket+1)){ active.add(arr[2*bucket]); active.add(arr[2*bucket+1]); }
    for (let i = 0; i < N.synWire.length; i++){
      const e = N.synWire[i], t = N.synM[i]; let r, g, b;
      if (active.has(N.synPre[i]))   { r = 1.0; g = 0.9; b = 0.5; }
      else if (t > 0.02)  { const k = 0.5 + 0.5 * Math.min(1, t/1.5);  r = 1.0*k; g = 0.62*k; b = 0.10*k; }
      else if (t < -0.02) { const k = 0.5 + 0.5 * Math.min(1, -t/0.7); r = 0.22*k; g = 0.55*k; b = 1.0*k; }
      else continue;
      const o = e*V*3;
      for (let v = 0; v < V; v++){ col[o+v*3] = r; col[o+v*3+1] = g; col[o+v*3+2] = b; }
    }
  }
  N.wires.geometry.attributes.color.needsUpdate = true;
}
function paintActivity(f, rates, danFlash, lastOut, bucket){
  const N = net[f]; if (!N || !N.nodes) return;
  const gl = N.glow, idx = N.nodeIdx;
  for (let i = 0; i < idx.length; i++) gl[i] = Math.min(1, rates[idx[i]] / 2.5);
  if (danFlash) for (const d of META.dan){ const i = N.nodeOf.get(d); if (i !== undefined) gl[i] = Math.max(gl[i], 0.9); }
  const lo = N.nodeOf.get(lastOut); if (lo !== undefined) gl[lo] = 1.0;
  // whole-CNS cloud: every neuron's speck flashes with its own live rate (166,700 channels)
  const cr = cloudRef[f];
  if (cr && rates.length * 3 >= cr.base.length){
    const cc = cr.pts.geometry.attributes.color.array, cb = cr.base;
    for (let i = 0, o = 0; o < cb.length; i++, o += 3){
      const k = 0.55 + 2.3 * Math.min(1, rates[i] / 2.5);
      cc[o] = Math.min(1.5, cb[o]*k); cc[o+1] = Math.min(1.5, cb[o+1]*k); cc[o+2] = Math.min(1.5, cb[o+2]*k);
    }
    cr.pts.geometry.attributes.color.needsUpdate = true;
  }
  lastAct[f] = performance.now();
  paintWires(f, bucket);
}
function setupBrainDrag(f){
  const el = $('bv'+f); if (!el) return;
  el.style.cursor = 'grab'; let drag = null;
  el.addEventListener('pointerdown', e => { drag = {x: e.clientX, y: e.clientY}; el.setPointerCapture(e.pointerId); el.style.cursor = 'grabbing'; });
  el.addEventListener('pointermove', e => {
    if (!drag) return;
    brainView[f].ry += (e.clientX - drag.x) * 0.011;
    brainView[f].rx = clamp(brainView[f].rx + (e.clientY - drag.y) * 0.011, -1.5, 1.5);
    drag = {x: e.clientX, y: e.clientY}; saveView(f);
  });
  const end = () => { drag = null; el.style.cursor = 'grab'; };
  el.addEventListener('pointerup', end); el.addEventListener('pointercancel', end);
  el.addEventListener('wheel', e => { e.preventDefault(); brainView[f].zoom = clamp(brainView[f].zoom * (e.deltaY > 0 ? 1.08 : 0.93), 0.35, 3.0); saveView(f); }, {passive: false});
  el.addEventListener('dblclick', () => { Object.assign(brainView[f], brainHome[f]); saveView(f); });
}

// ================= state polling =================
let cur = null, seenMoveNo = null, offlineShown = false, lastWinnerShown = 0;
let anim = null; const animQueue = [];
const steer = [{yaw: 0, thrust: 1, lift: 0, rate: 0}, {yaw: 0, thrust: 1, lift: 0, rate: 0}];
const prevAct = [null, null];          // previous /api/act samples for the fluctuation correlation
const groomUntil = [0, 0];
function fmtStats(b){
  const h = b.health || {};
  const bad = (h.rate_mean !== undefined && h.rate_mean < 0.005);
  return `<tr><td>多巴胺事件</td><td class="num">${(+b.rewards).toLocaleString()}</td></tr>` +
         `<tr><td>重塑事件</td><td class="num">${(+b.remodel_events).toLocaleString()}</td></tr>` +
         `<tr><td>被重塑突触(累计)</td><td class="num">${(+b.syn_changed).toLocaleString()}</td></tr>` +
         `<tr><td>Σ|Δw|</td><td class="num">${Math.round(+b.dw_total).toLocaleString()}</td></tr>` +
         `<tr><td>最近奖励 r</td><td class="num">${(+b.last_reward >= 0 ? '+' : '')}${(+b.last_reward).toFixed(2)}</td></tr>` +
         `<tr><td>本局平均Q</td><td class="num">${(+b.q_avg || 0).toFixed(3)}</td></tr>` +
         `<tr${bad ? ' style="color:#ff7d90"' : ''}><td>发放率均值</td><td class="num">${(+h.rate_mean || 0).toFixed(2)} · 活跃 ${Math.round((+h.spike_frac || 0)*1000)/10}%</td></tr>` +
         `<tr><td>可塑突触 |m|均值</td><td class="num">${(+h.pl_abs_mean || 0).toFixed(3)}（改动 ${(+h.pl_changed || 0).toLocaleString()}）</td></tr>`;
}
function enqueueMove(s){
  if (!s.last_move) return;
  if (seenMoveNo === null || s.move_no <= seenMoveNo){ seenMoveNo = s.move_no; return; }
  seenMoveNo = s.move_no;
  const lm = s.last_move, f = lm.p - 1;
  if (!flies[f]) return;
  const key = lm.x+'_'+lm.y+'_'+lm.p;
  pendingKeys.add(key);
  animQueue.push({fly: f, p: lm.p, r: lm.r || 0, key, x: lm.x, y: lm.y});
  // if the client cannot keep up, reveal the oldest queued stones without a performance
  while (animQueue.length > 2){ const q = animQueue.shift(); revealKey(q.key); }
}
function revealKey(key){ pendingKeys.delete(key); const m = stones.get(key); if (m) m.visible = true; }
function startNextAnim(){
  if (anim || !animQueue.length) return;
  const q = animQueue.shift();
  const fl = flies[q.fly];
  anim = {kind: 'move', fly: q.fly, p: q.p, r: q.r, key: q.key, tgt: boardPt(q.x, q.y), state: 'aim', t: 0, age: 0, carried: null, near: 0};
  setMarker(false);
  if (q.r < -0.3) groomUntil[q.fly] = performance.now() + 3500;
  fl.heading = q.fly === 0 ? 0 : Math.PI;
}
function startExplore(f, now){
  if (anim) return;
  anim = {kind: 'explore', fly: f, state: 'takeoff', t: 0, age: 0, near: 0, carried: null, until: now + 3500 + 5000 * Math.random()};
  flies[f].heading = f === 0 ? 0 : Math.PI;
}
// standing behaviour loop: every fly reports its situation and receives its own motor readouts
async function behaviorTick(){
  for (let f = 0; f < 2; f++){
    if (!flies[f]) continue;
    const mine = anim && anim.fly === f && anim.kind === 'move';
    try {
      const r = await fetch('/api/motor', {method: 'POST', body: JSON.stringify({fly: f, near: mine ? anim.near : 0, holding: mine && anim.carried ? 1 : 0})});
      steer[f] = await r.json();
    } catch(e){}
  }
}
function finishAnim(){
  if (!anim) return;
  const fl = flies[anim.fly];
  if (anim.carried){ fl.g.remove(anim.carried); anim.carried = null; }
  if (anim.kind === 'move'){ revealKey(anim.key); setMarker(true, anim.tgt.x, anim.tgt.z); }
  resetLegs(fl);
  anim = null;
}
function drawQ(f){
  const cv = $('qc'+f); if (!cv || !cur) return;
  const w = cv.clientWidth || 300, h = 38;
  if (cv.width !== w) cv.width = w; if (cv.height !== h) cv.height = h;
  const ctx = cv.getContext('2d'); ctx.clearRect(0, 0, w, h);
  const hist = cur.q_hist ? cur.q_hist[f] : [];
  $('qs'+f).textContent = hist.length ? `每局平均Q曲线（${hist.length}局 · 胜 ${cur.wins[f]}）最新 ${hist[hist.length-1].toFixed(2)}` : '每局平均Q曲线（尚无完整对局）';
  if (!hist.length) return;
  ctx.strokeStyle = '#2e3a52'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.beginPath();
  const yr = h-4 - (h-8)*0.2; ctx.moveTo(4, yr); ctx.lineTo(w-4, yr); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = f === 0 ? '#6fc2ff' : '#ff8fb3'; ctx.lineWidth = 1.5; ctx.beginPath();
  const n = hist.length;
  for (let i = 0; i < n; i++){ const x = n === 1 ? 4 : 4 + (w-8)*i/(n-1), y = h-4 - (h-8)*clamp(hist[i], 0, 1); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
  ctx.stroke();
}
async function poll(){
  try{
    const s = await (await fetch('/api/state')).json();
    cur = s;
    enqueueMove(s);
    syncStones(s.board);
    if (!anim && !animQueue.length){ if (s.last_move) setMarker(true, OFF+s.last_move.x*CELL, OFF+s.last_move.y*CELL); else setMarker(false); }
    status(s.thinking ? `果蝇${s.to_move} 思考中…` : (s.winner ? (s.winner === 3 ? '平局' : `终局：${s.winner===1 ? '黑(果蝇0)' : '白(果蝇1)'}胜`) : `轮到果蝇${s.to_move}`));
    $('mv').textContent = s.move_no;
    $('mvs').textContent = s.move_s ? s.move_s.toFixed(1)+'s' : '–';
    $('tick').textContent = s.tick_ms + ' ms/tick';
    $('gcount').textContent = `${s.games}局 ${s.wins[0]}:${s.wins[1]}`;
    const w0 = Math.round(100*(s.winrate ? s.winrate[0] : 0.5)), w1 = 100 - w0;
    $('wr0').style.width = w0+'%'; $('wr0t').textContent = w0+'%';
    $('wr1').style.width = w1+'%'; $('wr1t').textContent = w1+'%';
    if (s.judge) $('judgeLine').textContent = '最近一手评估：' + s.judge + '（r>0 多巴胺奖励 / r<0 惩罚）';
    $('t0').innerHTML = fmtStats(s.flies[0]); $('t1').innerHTML = fmtStats(s.flies[1]);
    drawQ(0); drawQ(1);
    if (META && $('msrc')){
      const m = META.motor_counts;
      $('msrc').textContent = META.motor_source === 'real_mn'
        ? `肢体绑定:真实运动神经元(腿${m.leg}·翅${m.wing}·颈${m.neck})`
        : '肢体绑定:空间代理(未找到 flygo_motor.npz)';
    }
    $('pstat').textContent = s.persist.loaded ? `已加载重塑状态（第${s.games}局）· 最近保存 ${s.persist.saved || '—'}`
                           : (s.persist.saved ? `未加载 · 最近保存 ${s.persist.saved}` : '未加载（初始突触）');
    for (let f = 0; f < 2; f++) $('src'+f).textContent = s.persist.loaded ? `第${s.games}局重塑` : '初始突触';
    if (s.moves.length){
      const total = s.moves.length;
      $('moves').innerHTML = s.moves.slice().reverse().map((m, i) => {
        if (m[0] < 0) return `<div class="${m[2]===1 ? 'fly0' : 'fly1'}" style="font-weight:600">— ${m[2]===1 ? '黑' : '白'} 获胜 —</div>`;
        return `<div><span class="mno">${total-i}</span> <span class="${m[2]===1 ? 'fly0' : 'fly1'}">${m[2]===1 ? '黑' : '白'}</span> (${String.fromCharCode(65+m[0])}${SZ-m[1]})</div>`;
      }).join('');
    }
    const w = $('winner');
    if (s.winner && s.winner !== lastWinnerShown){
      lastWinnerShown = s.winner; w.style.display = 'block';
      w.textContent = s.winner === 3 ? '平局' : (s.winner === 1 ? '果蝇0（黑）' : '果蝇1（白）') + ' 获胜';
      setTimeout(() => w.style.display = 'none', 4000);
    }
    if (!s.winner) lastWinnerShown = 0;
    offlineShown = false;
    try{
      const [r0, r1] = await Promise.all([fetch('/api/act?fly=0'), fetch('/api/act?fly=1')]);
      const a0 = new Float32Array(await r0.arrayBuffer()), a1 = new Float32Array(await r1.arrayBuffer());
      // two-brain independence, both metrics matter:
      //   structure r  - correlation of the raw rate vectors. Same connectome (same species)
      //                  makes this naturally high; it says the wiring matches, nothing more.
      //   fluctuation r- correlation of the SAMPLE-TO-SAMPLE CHANGES. This is the real
      //                  synchrony test: independent brains keep this low.
      { let n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
        for (let i = 0; i < a0.length; i += 37){ const x = a0[i], y = a1[i]; n++; sx += x; sy += y; sxx += x*x; syy += y*y; sxy += x*y; }
        const vx = sxx/n - (sx/n)**2, vy = syy/n - (sy/n)**2;
        const rs = (vx > 1e-9 && vy > 1e-9) ? (sxy/n - (sx/n)*(sy/n)) / Math.sqrt(vx*vy) : 0;
        let rf = null;
        if (prevAct[0] && prevAct[0].length === a0.length){
          let m = 0, dx0 = 0, dy0 = 0, dx2 = 0, dy2 = 0, dxy = 0;
          for (let i = 0; i < a0.length; i += 37){
            const x = a0[i] - prevAct[0][i], y = a1[i] - prevAct[1][i]; m++;
            dx0 += x; dy0 += y; dx2 += x*x; dy2 += y*y; dxy += x*y;
          }
          const cx = dx2/m - (dx0/m)**2, cy = dy2/m - (dy0/m)**2;
          if (cx > 1e-9 && cy > 1e-9) rf = (dxy/m - (dx0/m)*(dy0/m)) / Math.sqrt(cx*cy);
        }
        prevAct[0] = a0.slice(); prevAct[1] = a1.slice();
        const el = $('indep');
        if (el) el.textContent = '结构r=' + rs.toFixed(2) + ' · 波动r=' + (rf === null ? '…' : rf.toFixed(2)); }
      for (let f = 0; f < 2; f++){ try { await loadSyn(f); } catch(e){} }
      paintActivity(0, a0, s.flies[0].dan_flash, s.flies[0].last_out, s.flies[0].last_bucket);
      paintActivity(1, a1, s.flies[1].dan_flash, s.flies[1].last_out, s.flies[1].last_bucket);
    }catch(e){}
  }catch(e){
    if (!offlineShown){ offlineShown = true; status('服务器离线，等待重连…'); }
  }
}

// ================= controls =================
async function ctl(cmd, extra){
  const r = await fetch('/api/control', {method: 'POST', body: JSON.stringify(Object.assign({cmd}, extra || {}))});
  return r.json();
}
$('bPause').onclick = () => ctl('pause');
$('bResume').onclick = () => ctl('resume');
$('bStep').onclick = () => ctl('step');
$('bReset').onclick = () => ctl('reset');
$('bInstinct').onclick = () => { ctl('instinct'); $('bInstinct').classList.toggle('on'); };
$('bPlastic').onclick  = () => { ctl('plasticity'); $('bPlastic').classList.toggle('on'); };
$('bSpeed').onchange = e => ctl('speed', {value: +e.target.value});
$('bAlpha').oninput = e => { $('bAlphaV').textContent = (+e.target.value).toFixed(2); };
$('bAlpha').onchange = e => ctl('instinct_alpha', {value: +e.target.value});
$('bSave').onclick = async () => { await ctl('save'); poll(); };
$('bLoad').onclick = async () => { await ctl('load'); poll(); };
$('bClear').onclick = async () => { if (confirm('清除当前档位两只果蝇的全部突触重塑并删除其存档？')) { await ctl('clear_persist'); poll(); } };
$('bProfT').onclick = async () => { await ctl('profile', {value: 'trained'}); poll(); };
$('bProfP').onclick = async () => { await ctl('profile', {value: 'pristine'}); poll(); };
function setAuto(v){
  AUTO = v;
  try { localStorage.setItem('flygo.autolvl', v); } catch(e){}
  if ($('bA1')) $('bA1').classList.toggle('on', v === '1');
  if ($('bA2')) $('bA2').classList.toggle('on', v === '2');
  const el = $('autoNote');
  if (el) el.textContent = v === '2'
    ? '二重·神经主导：归航弱化，抓/放只认运动爆发信号，失败自动重进近（4次后兜底）'
    : '一重·闭环：脑决定落点，身体由代码执行（到达/抓取/放置均有保底，永不卡局）';
}
$('bA1').onclick = () => setAuto('1');
$('bA2').onclick = () => setAuto('2');
setAuto(AUTO);
window.dop = (fly, sign) => ctl(sign > 0 ? 'reward' : 'punish', {fly});

// ================= fly behaviour =================
function resetLegs(fl){ for (const L of fl.legsT1){ setDelta(L.coxa, 0, 0, 0); setDelta(L.femur, 0, 0, 0); setDelta(L.tibia, 0, 0, 0); } }
// neural flight step. Heading integrates homing + the fly's live neural yaw; homing gain ramps
// with elapsed time so arrival is guaranteed (no time-outs, no snapping). Tier 2 halves the
// code homing and lets the fly's own yaw signal stay strong: the target is a REGION it must
// reach, arrival radius widens accordingly (0.55 vs 0.3).
function flyStep(fl, to, dt, f, elapsed){
  const st = steer[f], p = fl.g.position;
  const dx = to.x - p.x, dz = to.z - p.z, dist = Math.hypot(dx, dz);
  const desired = Math.atan2(-dz, dx);
  let err = desired - fl.heading; err = Math.atan2(Math.sin(err), Math.cos(err));
  const near = clamp(1 - dist/3.0, 0, 1);
  const ramp = clamp(elapsed/PHYS.homingRamp, 0, 1);                 // neural wander early, homing late
  const t2 = AUTO === '2';
  const neural = st.yaw * PHYS.neuralYaw * (1 - near) * (1 - (t2 ? 0.25 : 0.85)*ramp);
  fl.heading += (err * (PHYS.turnGain + 5.0*near + 3.0*ramp*(t2 ? 0.35 : 1.0)) + neural) * dt;
  const speed = Math.min(PHYS.speed * st.thrust, Math.max(0.8, dist / Math.max(dt, 1e-3) * 0.85));
  p.x += Math.cos(fl.heading) * speed * dt;
  p.z += -Math.sin(fl.heading) * speed * dt;
  p.y += ((to.y + 0.5 * st.lift * (1 - near)) - p.y) * 3.2 * dt;
  fl.g.rotation.set(0.08 * st.lift * (1 - near), fl.heading, -clamp(err, -1, 1) * 0.4 * (1 - near));
  confine(fl, f, performance.now());
  return dist < (t2 ? 0.55 : 0.3);
}
// free flight inside the glass box: heading/altitude/speed are purely the fly's own signals
function flyFree(fl, dt, f, now){
  const st = steer[f], p = fl.g.position;
  fl.heading += st.yaw * PHYS.neuralYaw * 1.3 * dt;
  // soft wall avoidance well before the glass (the hard limit + penalty is in confine)
  const mx = BOX.x - 2.5, mz = BOX.z - 2.5;
  if (Math.abs(p.x) > mx || Math.abs(p.z) > mz){
    const toC = Math.atan2(-(0 - p.z), 0 - p.x); let e = toC - fl.heading; e = Math.atan2(Math.sin(e), Math.cos(e));
    fl.heading += e * 2.5 * dt;
  }
  const speed = PHYS.speed * 0.55 * st.thrust;
  p.x += Math.cos(fl.heading) * speed * dt;
  p.z += -Math.sin(fl.heading) * speed * dt;
  p.y += (st.lift * 2.2 + (PHYS.cruise + 1.0 - p.y) * 0.6) * dt;
  fl.g.rotation.set(0.10 * st.lift, fl.heading, -st.yaw * 0.3);
  confine(fl, f, now);
}
function faceTowards(fl, to, dt, k){
  const p = fl.g.position; const desired = Math.atan2(-(to.z - p.z), to.x - p.x);
  let err = desired - fl.heading; err = Math.atan2(Math.sin(err), Math.cos(err));
  fl.heading += err * k * dt; fl.g.rotation.set(0, fl.heading, 0);
}
function updateAnim(now, dt){
  if (!anim) startNextAnim();
  if (!anim) return;
  const fl = flies[anim.fly], f = anim.fly;
  anim.t += dt; anim.age += dt;
  // watchdog: if one phase of a move somehow stops progressing, release the board
  // (stone revealed, fly returned) so the game can never stall on a stuck animation.
  if (anim.lastState !== anim.state){ anim.lastState = anim.state; anim.watch = now; }
  else if (anim.kind === 'move' && now - (anim.watch || 0) > 18000){
    console.warn('anim watchdog fired', anim.state); finishAnim(); return;
  }
  const rest = REST[f], dish = DISH[f], cruise = PHYS.cruise;
  const drv = (steer[f].drive !== undefined) ? steer[f].drive : 0.35;
  // exploration yields to a queued move and ends on its own timer
  if (anim.kind === 'explore' && anim.state !== 'back' && anim.state !== 'land' && (animQueue.length || now > anim.until)){
    anim.state = 'back'; anim.t = 0;
  }
  let airborne = true;
  switch (anim.state){
    case 'takeoff': {                                              // exploration: lift off, then free flight
      fl.g.position.y = rest.y + 1.7 * ease(Math.min(1, anim.t/0.4));
      if (anim.t > 0.4){ anim.state = 'explore'; anim.t = 0; }
      break; }
    case 'explore': {
      flyFree(fl, dt, f, now);
      break; }
    case 'aim': {
      airborne = false; faceTowards(fl, dish, dt, 3.5);
      const aimT = 0.3 * (1.3 - 0.9 * drv);                         // motivated flies take off sooner
      for (const L of fl.legsT1) setDelta(L.femur, 0, 0, -L.s * 0.25 * ease(Math.min(1, anim.t/aimT)));
      if (anim.t > aimT){ anim.state = 'lift'; anim.t = 0; }
      break; }
    case 'lift': {
      fl.g.position.y = rest.y + 1.7 * ease(Math.min(1, anim.t/0.35));
      faceTowards(fl, dish, dt, 4.5);
      if (anim.t > 0.35){ anim.state = 'toDish'; anim.t = 0; }
      break; }
    case 'toDish': {
      _v.set(dish.x, dish.y + 1.6, dish.z);
      anim.near = clamp(1 - Math.hypot(dish.x - fl.g.position.x, dish.z - fl.g.position.z) / 2.2, 0, 1);
      if (flyStep(fl, _v, dt, f, anim.t)){ anim.state = 'grab'; anim.t = 0; anim.acted = 0; }
      break; }
    case 'grab': {                                                 // hover down; the tarsal burst closes the legs
      anim.near = 1;
      const st = steer[f], dive = PHYS.dive * (0.8 + 0.5 * Math.max(0, -(st.lift || 0)));
      fl.g.position.x = lerp(fl.g.position.x, dish.x, 0.2); fl.g.position.z = lerp(fl.g.position.z, dish.z, 0.2);
      fl.g.position.y += ((dish.y + 0.95) - fl.g.position.y) * dive * dt;
      fl.g.rotation.set(0, fl.heading, 0);
      const low = fl.g.position.y < dish.y + 1.15;
      const burst = (st.grasp || 0) > 0.12;
      // the tarsal burst closes the legs. Tier 1 keeps a 2.4 s reflex fallback so the
      // board never waits; Tier 2 trusts the burst: no burst -> climb back and re-approach,
      // and only after 4 failed tries does the disclosed Tier-1 rescue step in.
      let grab = burst;
      if (!grab && low && anim.t > 2.4){
        if (AUTO === '1' || (anim.retries = (anim.retries || 0) + 1) > 4) grab = true;
        else { anim.state = 'toDish'; anim.t = 0; }
      }
      if (grab && !anim.carried){                                            // neural timing
        anim.carried = new THREE.Mesh(stoneGeo, anim.p === 1 ? matB : matW);
        anim.carried.position.set(0.15, fl.carryY - 0.1, 0); fl.g.add(anim.carried);
        anim.acted = anim.t;
      }
      const k = anim.carried ? clamp((anim.t - anim.acted) / 0.25, 0, 1) : 0;
      for (const L of fl.legsT1){ setDelta(L.femur, 0, 0, -L.s * 0.55 * k); setDelta(L.tibia, 0, 0, L.s * 0.5 * k); }
      if (anim.carried && anim.t - anim.acted > 0.3){ anim.state = 'rise'; anim.t = 0; }
      break; }
    case 'rise': {
      anim.near = 0;
      fl.g.position.y = lerp(fl.g.position.y, cruise, 0.14);
      faceTowards(fl, anim.tgt, dt, 3.5);
      if (anim.t > 0.3){ anim.state = 'travel'; anim.t = 0; }
      break; }
    case 'travel': {
      _v.set(anim.tgt.x, cruise, anim.tgt.z);
      anim.near = clamp(1 - Math.hypot(anim.tgt.x - fl.g.position.x, anim.tgt.z - fl.g.position.z) / 2.2, 0, 1);
      if (flyStep(fl, _v, dt, f, anim.t)){ anim.state = 'place'; anim.t = 0; anim.acted = -1; }
      break; }
    case 'place': {                                                // hover down onto the intersection; burst opens the legs
      anim.near = 1;
      const st = steer[f], dive = PHYS.dive * (0.8 + 0.5 * Math.max(0, -(st.lift || 0)));
      const releaseY = 0.45 - fl.carryY + 0.25;
      fl.g.position.x = lerp(fl.g.position.x, anim.tgt.x, 0.25); fl.g.position.z = lerp(fl.g.position.z, anim.tgt.z, 0.25);
      fl.g.position.y += (releaseY - fl.g.position.y) * dive * dt;
      fl.g.rotation.set(0, fl.heading, 0);
      const settled = Math.abs(fl.g.position.y - releaseY) < 0.08 &&
                      Math.hypot(fl.g.position.x - anim.tgt.x, fl.g.position.z - anim.tgt.z) < 0.05;
      const burst = (st.grasp || 0) > 0.12;
      // mirror of grab: the burst opens the legs to release. Tier 2 re-approaches instead.
      let drop = burst;
      if (!drop && settled && anim.t > 2.4){
        if (AUTO === '1' || (anim.retries = (anim.retries || 0) + 1) > 4) drop = true;
        else { anim.state = 'travel'; anim.t = 0; }                    // pull up, try the approach again
      }
      if (drop && anim.carried){
        fl.g.position.x = anim.tgt.x; fl.g.position.z = anim.tgt.z;   // exact: carried and placed stone coincide
        fl.g.remove(anim.carried); anim.carried = null;
        revealKey(anim.key);
        setMarker(true, anim.tgt.x, anim.tgt.z);
        fl.halo.material.color.setHex(anim.r < -0.05 ? 0xff5f7a : (anim.r > 0.05 ? 0x33ff73 : 0x88aaff)); fl.haloT = now;
        anim.acted = anim.t;
      }
      const k = anim.acted >= 0 ? 1 - clamp((anim.t - anim.acted) / 0.25, 0, 1) : 1;
      for (const L of fl.legsT1){ setDelta(L.femur, 0, 0, -L.s * 0.55 * k); setDelta(L.tibia, 0, 0, L.s * 0.5 * k); }
      if (anim.acted >= 0 && anim.t - anim.acted > 0.3){ anim.state = 'back'; anim.t = 0; }
      break; }
    case 'back': {
      anim.near = 0;
      _v.set(rest.x, rest.y + 1.7, rest.z);
      if (flyStep(fl, _v, dt, f, anim.t)){ anim.state = 'land'; anim.t = 0; }
      break; }
    case 'land': {
      const k = Math.min(1, anim.t/0.45);
      fl.g.position.x = lerp(fl.g.position.x, rest.x, 0.2); fl.g.position.z = lerp(fl.g.position.z, rest.z, 0.2);
      fl.g.position.y = lerp(fl.g.position.y, rest.y, 0.16);
      const home = f === 0 ? 0 : Math.PI;
      let err = home - fl.heading; err = Math.atan2(Math.sin(err), Math.cos(err)); fl.heading += err * 6 * dt;
      fl.g.rotation.set(0, fl.heading, 0);
      airborne = k < 0.6;
      if (k >= 1){ fl.g.position.copy(rest); fl.heading = home; fl.g.rotation.set(0, home, 0); finishAnim(); return; }
      break; }
  }
  for (let i = 0; i < 2; i++){
    const s = i === 0 ? 1 : -1;
    fl.wings[i].rotation.x = airborne ? s * Math.sin(now * 0.00628 * PHYS.wingHz + i) * 0.7 : s * Math.sin(now * 0.003) * 0.05;
  }
  if (airborne) for (const L of fl.legsT1) setDelta(L.coxa, 0, 0, -L.s * 0.3);
}
// standing behaviour: everything below is driven by the fly's own readouts (steer[f]) through
// spring-like joints — no keyframes. Motivation (dopamine drive) trades grooming for attention.
function idleFly(f, now, t, dt){
  const fl = flies[f]; if (!fl || (anim && anim.fly === f)) return;
  const st = steer[f], drv = st.drive !== undefined ? st.drive : 0.35;
  // low motivation -> occasional autonomous exploration flight inside the glass box
  if (!anim && !animQueue.length && cur && !(cur.thinking && cur.to_move === f) && Math.random() < dt * 0.06 * (1 - drv)){ startExplore(f, now); return; }
  const k = 1 - Math.exp(-PHYS.jointK * dt);
  const home = f === 0 ? 0 : Math.PI;
  fl.g.position.y = REST[f].y + 0.02 * Math.sin(t * 1.7 + f * 2) + 0.015 * ((st.thrust || 1) - 1);
  fl.heading = home; fl.g.rotation.set(0, home + 0.03 * Math.sin(t * 0.5 + f), 0);
  // head: attention to the board grows with drive; otherwise it scans with the hemispheric signal
  const attention = clamp(0.25 + PHYS.attnGain * drv, 0, 1);
  let yawB = 0, pitB = 0;
  if (cur && cur.last_move){
    _v.copy(boardPt(cur.last_move.x, cur.last_move.y)); fl.g.worldToLocal(_v);
    yawB = clamp(Math.atan2(-_v.z, _v.x), -0.7, 0.7);
    pitB = clamp(Math.atan2(_v.y - 0.5, Math.hypot(_v.x, _v.z)), -0.4, 0.3);
  }
  const gf = clamp((st.groom_f || 0), 0, 1.5) * PHYS.groomGain * (1 - 0.65 * drv);    // front-leg grooming (legs, eyes)
  const gh = clamp((st.groom_h || 0), 0, 1.0) * PHYS.groomGain * (1 - 0.5 * drv);     // hind-leg grooming (wings)
  const yawT = attention * yawB + (1 - attention) * (st.yaw || 0) * 0.6;
  const pitT = attention * pitB + (1 - attention) * (st.lift || 0) * 0.3 - 0.35 * gf;  // head dips while cleaning the eyes
  fl.head.rotation.z = lerp(fl.head.rotation.z, yawT, k);
  fl.head.rotation.y = lerp(fl.head.rotation.y, -pitT, k);
  // front legs: rubbing rhythm comes from the live left/right asymmetry of the leg region
  const rub = st.lr ? clamp((st.lr[1] - st.lr[0]) * 40, -1, 1) : 0;
  fl.gj = fl.gj || {c: [0, 0], fm: [0, 0], tb: [0, 0], hc: [0, 0], hf: [0, 0], w: [0, 0]};
  for (let i = 0; i < 2; i++){
    const L = fl.legsT1[i], s = L.s;
    const cT = s * (0.35 * gf) + rub * 0.15 * gf, fT = -s * (0.9 * gf) + s * rub * 0.25 * gf, tT = s * 0.4 * gf;
    fl.gj.c[i] = lerp(fl.gj.c[i], cT, k); fl.gj.fm[i] = lerp(fl.gj.fm[i], fT, k); fl.gj.tb[i] = lerp(fl.gj.tb[i], tT, k);
    setDelta(L.coxa, 0, 0, fl.gj.c[i]); setDelta(L.femur, 0, 0, fl.gj.fm[i]); setDelta(L.tibia, 0, 0, fl.gj.tb[i]);
    // hind legs reach up to the wings when the posterior cord bursts
    const H = fl.legsT3[i];
    fl.gj.hc[i] = lerp(fl.gj.hc[i], s * 0.45 * gh, k); fl.gj.hf[i] = lerp(fl.gj.hf[i], -s * 0.8 * gh, k);
    setDelta(H.coxa, 0, 0, fl.gj.hc[i]); setDelta(H.femur, 0, 0, fl.gj.hf[i]);
    // wings: tremor + flick bursts + tilt toward a grooming hind leg
    const wT = s * (Math.sin(now * 0.003 + i) * 0.04 + 0.5 * clamp(st.wingflick || 0, 0, 1) + 0.25 * gh);
    fl.gj.w[i] = lerp(fl.gj.w[i], wT, k);
    fl.wings[i].rotation.x = fl.gj.w[i];
  }
}
function updateHalos(now){
  for (const fl of flies){
    if (!fl) continue;
    const dt = (now - fl.haloT) / 1000;
    if (dt < 0.9){ fl.halo.visible = true; fl.halo.material.opacity = 0.85 * (1 - dt/0.9); fl.halo.scale.setScalar(0.7 + dt * 1.3); }
    else fl.halo.visible = false;
  }
}

// ================= render loop =================
let lastFrame = performance.now(), errShown = false;
function frame(now){
  const dt = Math.min(0.05, (now - lastFrame) / 1000); lastFrame = now;
  const t = now / 1000, W = innerWidth, H = innerHeight;
  renderer.setSize(W, H);
  camera.aspect = W/H; camera.updateProjectionMatrix();
  if (ring.visible){ ring.scale.setScalar(1 + 0.10 * Math.sin(t * 5.0)); beacon.material.opacity = 0.20 + 0.16 * (0.5 + 0.5 * Math.sin(t * 5.0)); }
  updateAnim(now, dt);
  idleFly(0, now, t, dt); idleFly(1, now, t, dt);
  updateHalos(now);
  renderer.setViewport(0, 0, W, H);
  renderer.render(scene, camera);
  for (let f = 0; f < 2; f++){
    const R = brainR[f]; if (!R) continue;
    const el = $('bv'+f); if (!el) continue;
    const w = el.clientWidth, h = el.clientHeight; if (w < 10 || h < 10) continue;
    if (R.domElement.width !== w || R.domElement.height !== h) R.setSize(w, h, false);
    const cam = brainCams[f]; cam.aspect = w / h;
    const bv = brainView[f];
    brainRoot[f].rotation.set(bv.rx, bv.ry, 0);
    cam.position.set(0, 0, 1.7 * bv.zoom); cam.lookAt(0, 0, 0); cam.updateProjectionMatrix();
    if (net[f] && net[f].nodes){
      const Nv = net[f], gl = Nv.glow, base = Nv.base;
      const colA = Nv.nodes.geometry.attributes.color, posA = Nv.nodes.geometry.attributes.position;
      const col = colA.array, pos = posA.array, pv = Nv.protoPos;
      const dec = Math.exp(-dt * 2.0);
      let any = false;
      const V = Nv.sphereVerts;
      for (let i = 0; i < gl.length; i++){
        if (gl[i] > 0.003){ gl[i] *= dec; any = true; }
        else if (gl[i] !== 0){ gl[i] = 0; any = true; }
      }
      if (any){
        // inactive = readable dark-hue silhouette; active = white-hot class color + size pulse
        for (let i = 0; i < gl.length; i++){
          const g = gl[i], gb = NODE_IDLE + 1.15 * g;
          const r = Math.min(1.6, base[3*i]   * gb + g * 0.95);
          const gg = Math.min(1.6, base[3*i+1] * gb + g * 0.90);
          const b = Math.min(1.6, base[3*i+2] * gb + g * 0.68);
          const s = NODE_R * Nv.nsz[i] * (0.8 + 0.55 * g);
          const cx = Nv.lay[3*i], cy = Nv.lay[3*i+1], cz = Nv.lay[3*i+2];
          const o = i*V*3;
          for (let v = 0; v < V; v++){
            const p = o + v*3;
            col[p] = r; col[p+1] = gg; col[p+2] = b;
            pos[p] = pv[v*3]*s + cx; pos[p+1] = pv[v*3+1]*s + cy; pos[p+2] = pv[v*3+2]*s + cz;
          }
        }
        colA.needsUpdate = true;
        posA.needsUpdate = true;
      }
    }
    // travelling synapse pulses: respawn prefers currently-active edges, brightness follows
    // presynaptic firing -> the traffic concentrates on whatever the fly is processing.
    if (net[f] && net[f].sparks){
      const Nv = net[f], gl = Nv.glow, lay = Nv.lay, preN = Nv.preNode, postN = Nv.postNode, wb = Nv.wireBase;
      const pp = Nv.sparks.geometry.attributes.position.array;
      const pc = Nv.sparks.geometry.attributes.color.array;
      const nE = preN.length;
      for (let s = 0; s < N_PULSE; s++){
        let e = Nv.pulseE[s], t = Nv.pulseT[s] + dt * Nv.pulseSp[s];
        if (t >= 1){
          let best = (Math.random()*nE)|0, bg = gl[preN[best]];
          for (let k = 0; k < 6; k++){ const c = (Math.random()*nE)|0; if (gl[preN[c]] > bg){ bg = gl[preN[c]]; best = c; } }
          e = best; Nv.pulseE[s] = e; t = 0;
        }
        Nv.pulseT[s] = t;
        const a3 = preN[e]*3, b3 = postN[e]*3;
        pp[3*s]   = lay[a3]   + (lay[b3]  - lay[a3])  *t;
        pp[3*s+1] = lay[a3+1] + (lay[b3+1]- lay[a3+1])*t;
        pp[3*s+2] = lay[a3+2] + (lay[b3+2]- lay[a3+2])*t;
        const g = Math.min(1, gl[preN[e]] * 1.6), amp = 0.10 + 2.2*g;
        pc[3*s]   = Math.min(1.6, wb[3*e]  *amp + g*0.90);
        pc[3*s+1] = Math.min(1.6, wb[3*e+1]*amp + g*0.80);
        pc[3*s+2] = Math.min(1.6, wb[3*e+2]*amp + g*0.55);
      }
      Nv.sparks.geometry.attributes.position.needsUpdate = true;
      Nv.sparks.geometry.attributes.color.needsUpdate = true;
    }
    // halo sprites: one additive glow dot per neuron, colored by its live firing rate
    if (net[f] && net[f].halos){
      const Nv = net[f], hc = Nv.halos.geometry.attributes.color.array;
      let hany = false;
      for (let i = 0; i < Nv.glow.length; i++){
        const g = Nv.glow[i], o = i*3;
        if (g > 0.004){
          const k = 0.25 + 2.1*g;
          hc[o] = Math.min(1.5, Nv.base[3*i]*k); hc[o+1] = Math.min(1.5, Nv.base[3*i+1]*k); hc[o+2] = Math.min(1.5, Nv.base[3*i+2]*k);
          hany = true;
        } else if (hc[o] !== 0 || hc[o+1] !== 0 || hc[o+2] !== 0){ hc[o] = 0; hc[o+1] = 0; hc[o+2] = 0; hany = true; }
      }
      if (hany) Nv.halos.geometry.attributes.color.needsUpdate = true;
    }
    R.render(brainSc[f], cam);
  }
}
function loop(now){
  requestAnimationFrame(loop);
  try { frame(now); }
  catch(e){ if (!errShown){ errShown = true; console.error(e); status('渲染异常: ' + e.message); } }
}

// ================= deterministic init =================
(async function init(){
  status('加载连接组与果蝇模型…');
  loop(performance.now());
  setupBrainRender(0); setupBrainRender(1);
  setupBrainDrag(0); setupBrainDrag(1);
  await Promise.all([loadConnectome().catch(e => console.error('connectome', e)),
                     loadFlyModel().catch(e => console.error('fly model', e))]);
  if (FLYMODEL){ flies[0] = buildFly(0); flies[1] = buildFly(1); }
  if (POSB) await Promise.all([loadNet(0).catch(e => console.error('net0', e)), loadNet(1).catch(e => console.error('net1', e))]);
  buildPhysPanel(); pushServerPhys();
  setInterval(behaviorTick, 150);
  poll(); setInterval(poll, 700);
})();
// read-only debug handle (page top-level bindings are not visible to devtools isolated worlds)
window.FG = { get flies(){ return flies; }, get net(){ return net; }, get brainView(){ return brainView; }, get auto(){ return AUTO; },
              get brainHome(){ return brainHome; }, get POSB(){ return POSB; }, get META(){ return META; } };
