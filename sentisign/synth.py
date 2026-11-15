"""Synthetic sign generator for pipeline smoke tests. NOT real data.
Each gloss = (handshape template, body-anchor location, movement path). Each signer adds
personal offset/scale/speed/noise so signer-independent splits are meaningful.
Body-relative glosses share handshape but differ in location (tests the location stream)."""
import numpy as np


def _rng(seed): return np.random.default_rng(seed)


def make_gloss_bank(n_gloss, seed=0, n_shapes=12):
    r = _rng(seed)
    shapes = r.normal(size=(n_shapes, 21, 3)) * 0.3
    bank = []
    for g in range(n_gloss):
        bank.append(dict(
            shape=shapes[g % n_shapes],            # many glosses share a handshape
            loc=r.uniform(-1, 1, 3) * [0.8, 1.0, 0.2],   # body-frame target (x,y,z)
            path=r.normal(size=3) * 0.35,          # movement direction
            orient=r.normal(size=3)))
    return bank


def make_signer(seed, var=1.0):
    """var scales between-signer variation (style: offset of signing space, speed, hand shape idiosyncrasy)."""
    r = _rng(10_000 + seed)
    return dict(off=r.normal(size=3) * 0.12 * var, scale=r.uniform(0.8, 1.25), speed=r.uniform(1 - .2*var, 1 + .25*var),
                tremor=r.uniform(0.01, 0.05), cam=r.normal(size=3) * 0.2, handsz=r.uniform(0.8, 1.2),
                shape_dev=r.normal(size=(21, 3)) * 0.06 * var, path_gain=r.uniform(1 - .3*var, 1 + .3*var) if var>0 else 1.0)


def render(gloss, signer, r, T=45, noise=1.0, drop=0.0):
    """Returns hands (T,2,21,3), pose (T,33,3), anchors (T,5,3) in 'camera' coords."""
    sh_w = 0.4 * signer["scale"]
    pose = np.zeros((33, 3)); pose[11] = [-sh_w/2, 0, 0]; pose[12] = [sh_w/2, 0, 0]
    pose = pose + signer["cam"]
    origin = (pose[11] + pose[12]) / 2
    face = origin + np.array([0, -0.5, 0]) * sh_w * 2
    anchors = np.stack([face, face + [0, .1*sh_w*2, 0], face + [0, -.1*sh_w*2, 0], face+[-.08*sh_w*2,0,0], face+[.08*sh_w*2,0,0]])
    t = np.linspace(0, 1, T)[:, None]
    ts = np.clip(t * signer["speed"], 0, 1)
    pos = origin + (gloss["loc"] + signer["off"]) * sh_w + gloss["path"] * signer.get("path_gain", 1) * sh_w * (ts - .5)
    pos = pos + r.normal(size=pos.shape) * signer["tremor"] * sh_w * noise
    hands = np.full((T, 2, 21, 3), np.nan)
    shape = (gloss["shape"] + signer.get("shape_dev", 0)) * signer["handsz"] * 0.08
    hand = shape + r.normal(size=shape.shape) * 0.004 * signer["handsz"] * noise
    hand = hand - hand[0]
    for i in range(T):
        if r.random() < drop: continue  # MediaPipe missed the hand this frame
        hands[i, 1] = hand + pos[i] + r.normal(size=(21, 3)) * 0.002 * noise
    return hands, np.repeat(pose[None], T, 0), np.repeat(anchors[None], T, 0)


def make_dataset(n_gloss=50, n_signers=12, reps=6, T=60, seed=0, var=1.0, noise=1.0, drop=0.0, fill=True):
    from .features import sequence_features, concat
    bank = make_gloss_bank(n_gloss, seed)
    signers = [make_signer(s, var) for s in range(n_signers)]
    r = _rng(seed + 1)
    X, y, sg = [], [], []
    for si, s in enumerate(signers):
        for g, gl in enumerate(bank):
            for _ in range(reps):
                h, p, a = render(gl, s, r, noise=noise, drop=drop)
                X.append(concat(sequence_features(h, p, a, T=T, fill=fill))); y.append(g); sg.append(si)
    return np.array(X, dtype=np.float32), np.array(y), np.array(sg)
