"""Multi-stream sign features. Pure numpy so Python and browser (JS port) can be
checked against the same golden vectors.

Input per frame: hands (2,21,3) [L,R], pose (33,3) MediaPipe layout, face anchors (5,3)
(nose, chin, forehead, left cheek, right cheek) with NaN where missing.
Body frame: origin = shoulder midpoint, scale = shoulder width, so location is
signer-size independent and hand position relative to face/torso is preserved.
"""
import numpy as np

L_SH, R_SH = 11, 12
EPS = 1e-6


def body_frame(pose):
    ls, rs = pose[L_SH], pose[R_SH]
    origin = (ls + rs) / 2.0
    scale = np.linalg.norm(ls[:2] - rs[:2])
    return origin, max(scale, EPS)


def handshape(hand):
    """63-d wrist-relative, hand-size-normalised shape (location removed)."""
    if np.isnan(hand).any():
        return np.zeros(63, dtype=np.float32)
    rel = hand - hand[0]
    size = np.linalg.norm(hand[9] - hand[0]) + EPS
    return (rel / size).reshape(-1).astype(np.float32)


def location(hand, origin, scale, anchors):
    """Wrist + index-tip position in body frame (6) + wrist-to-anchor vectors (15)."""
    if np.isnan(hand).any():
        return np.zeros(21, dtype=np.float32)
    w = (hand[0] - origin) / scale
    t = (hand[8] - origin) / scale
    parts = [w, t]
    for a in anchors:
        parts.append(np.zeros(3) if np.isnan(a).any() else (hand[0] - a) / scale)
    return np.concatenate(parts).astype(np.float32)


def orientation(hand):
    """Palm normal (3) from wrist, index MCP, pinky MCP."""
    if np.isnan(hand).any():
        return np.zeros(3, dtype=np.float32)
    n = np.cross(hand[5] - hand[0], hand[17] - hand[0])
    return (n / (np.linalg.norm(n) + EPS)).astype(np.float32)


def frame_features(hands, pose, anchors):
    origin, scale = body_frame(pose)
    hs, loc, ori = [], [], []
    for h in hands:
        hs.append(handshape(h)); loc.append(location(h, origin, scale, anchors[:5])); ori.append(orientation(h))
    return np.concatenate(hs), np.concatenate(loc), np.concatenate(ori)


def fill_missing_hands(hands_seq, max_gap=15):
    """Linearly interpolate a hand across short MediaPipe dropouts (<= max_gap frames); edges are
    held from the nearest detection. A hand never seen stays NaN (-> zeros downstream)."""
    H = np.array(hands_seq, dtype=float).copy(); n = len(H)
    for k in range(H.shape[1]):
        ok = ~np.isnan(H[:, k]).any(axis=(1, 2))
        if ok.all() or not ok.any(): continue
        idx = np.where(ok)[0]
        for i in np.where(~ok)[0]:
            lo = idx[idx < i]; hi = idx[idx > i]
            if len(lo) and len(hi):
                if hi[0] - lo[-1] - 1 > max_gap: continue
                w = (i - lo[-1]) / (hi[0] - lo[-1]); H[i, k] = (1 - w) * H[lo[-1], k] + w * H[hi[0], k]
            elif len(lo) and i - lo[-1] <= max_gap: H[i, k] = H[lo[-1], k]
            elif len(hi) and hi[0] - i <= max_gap: H[i, k] = H[hi[0], k]
    return H


def sequence_features(hands_seq, pose_seq, anchor_seq, T=60, fill=True):
    """Returns dict of streams, each (T, d). Movement = frame-to-frame delta of location."""
    if fill: hands_seq = fill_missing_hands(hands_seq)
    n = len(hands_seq)
    H, Lc, O = [], [], []
    for i in range(n):
        h, l, o = frame_features(hands_seq[i], pose_seq[i], anchor_seq[i])
        H.append(h); Lc.append(l); O.append(o)
    H, Lc, O = map(np.array, (H, Lc, O))
    idx = np.linspace(0, n - 1, T).round().astype(int)  # resample to fixed length
    H, Lc, O = H[idx], Lc[idx], O[idx]
    M = np.diff(Lc, axis=0, prepend=Lc[:1])
    return {"shape": H, "loc": Lc, "move": M, "orient": O}


STREAM_DIMS = {"shape": 126, "loc": 42, "move": 42, "orient": 6}
TOTAL_DIM = sum(STREAM_DIMS.values())  # 216


def concat(streams):
    return np.concatenate([streams[k] for k in ("shape", "loc", "move", "orient")], axis=-1)
