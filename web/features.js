// JS port of sentisign/features.py. Must match Python to 1e-4 (checked by tests/test_features_js.mjs
// against tests/golden/*.npz-derived JSON). Arrays: hands [T][2][21][3], pose [T][33][3], anchors [T][5][3]; NaN = missing.
const L_SH = 11, R_SH = 12, EPS = 1e-6;
const isNaNArr = a => a.some(p => p.some(v => Number.isNaN(v)));
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm2 = a => Math.hypot(a[0], a[1]);
const norm3 = a => Math.hypot(a[0], a[1], a[2]);
const scale = (a, s) => [a[0] / s, a[1] / s, a[2] / s];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

export function fillMissingHands(H, maxGap = 15) {
  H = H.map(f => f.map(h => h.map(p => p.slice())));
  const n = H.length;
  for (let k = 0; k < 2; k++) {
    const ok = H.map(f => !isNaNArr(f[k]));
    if (ok.every(Boolean) || !ok.some(Boolean)) continue;
    const idx = ok.map((v, i) => (v ? i : -1)).filter(i => i >= 0);
    for (let i = 0; i < n; i++) {
      if (ok[i]) continue;
      const lo = idx.filter(j => j < i), hi = idx.filter(j => j > i);
      if (lo.length && hi.length) {
        const a = lo[lo.length - 1], b = hi[0];
        if (b - a - 1 > maxGap) continue;
        const w = (i - a) / (b - a);
        H[i][k] = H[a][k].map((p, q) => p.map((v, c) => (1 - w) * v + w * H[b][k][q][c]));
      } else if (lo.length && i - lo[lo.length - 1] <= maxGap) H[i][k] = H[lo[lo.length - 1]][k].map(p => p.slice());
      else if (hi.length && hi[0] - i <= maxGap) H[i][k] = H[hi[0]][k].map(p => p.slice());
    }
  }
  return H;
}

function bodyFrame(pose) {
  const ls = pose[L_SH], rs = pose[R_SH];
  const origin = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2, (ls[2] + rs[2]) / 2];
  return [origin, Math.max(norm2(sub(ls, rs)), EPS)];
}

function handshape(hand) {
  if (isNaNArr(hand)) return new Array(63).fill(0);
  const size = norm3(sub(hand[9], hand[0])) + EPS;
  return hand.flatMap(p => scale(sub(p, hand[0]), size));
}

function location(hand, origin, s, anchors) {
  if (isNaNArr(hand)) return new Array(21).fill(0);
  const out = [...scale(sub(hand[0], origin), s), ...scale(sub(hand[8], origin), s)];
  for (const a of anchors) out.push(...(isNaNArr([a]) ? [0, 0, 0] : scale(sub(hand[0], a), s)));
  return out;
}

function orientation(hand) {
  if (isNaNArr(hand)) return [0, 0, 0];
  const n = cross(sub(hand[5], hand[0]), sub(hand[17], hand[0]));
  return scale(n, norm3(n) + EPS);
}

// numpy .round() is round-half-to-even
const roundHalfEven = x => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r; };

export function sequenceFeatures(hands, pose, anchors, T = 60, fill = true) {
  if (fill) hands = fillMissingHands(hands);
  const n = hands.length, Hs = [], Ls = [], Os = [];
  for (let i = 0; i < n; i++) {
    const [o, s] = bodyFrame(pose[i]);
    const h = [], l = [], r = [];
    for (const hand of hands[i]) { h.push(...handshape(hand)); l.push(...location(hand, o, s, anchors[i].slice(0, 5))); r.push(...orientation(hand)); }
    Hs.push(h); Ls.push(l); Os.push(r);
  }
  const idx = Array.from({ length: T }, (_, t) => roundHalfEven(T === 1 ? 0 : (t * (n - 1)) / (T - 1)));
  const H = idx.map(i => Hs[i]), L = idx.map(i => Ls[i]), O = idx.map(i => Os[i]);
  const M = L.map((row, t) => row.map((v, c) => v - (t === 0 ? L[0][c] : L[t - 1][c])));
  return H.map((_, t) => [...H[t], ...L[t], ...M[t], ...O[t]]);   // (T, 216): shape|loc|move|orient
}
