"""Emotion from face blendshapes (52-d, MediaPipe). Synthetic here; real data on Colab.
Per-user neutral calibration: subtract the user's mean neutral vector.
Grammar confound: raised brows in questions (browInnerUp) mimic Surprise."""
import numpy as np
EMOS = ["Neutral", "Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise"]
D = 52
AFFECT_IDX = np.arange(0, 40)       # affect-related blendshapes
GRAMMAR_IDX = np.arange(40, 52)     # brow/head cues used for questions/negation


def make_users(n, seed=0):
    r = np.random.default_rng(seed); return [dict(rest=r.normal(size=D) * 0.25, gain=r.uniform(.6, 1.4)) for _ in range(n)]


def proto(seed=1):
    r = np.random.default_rng(seed); p = np.zeros((7, D)); p[1:, :40] = r.normal(size=(6, 40)) * 0.28; return p


def sample(users, protos, n_per=40, grammar_p=0.25, seed=2):
    r = np.random.default_rng(seed); X, y, u = [], [], []
    for ui, us in enumerate(users):
        for e in range(7):
            for _ in range(n_per):
                x = us["rest"] + us["gain"] * protos[e] + r.normal(size=D) * 0.55
                if r.random() < grammar_p:  # question brows: grammar cue AND leaks into Surprise-like affect dims
                    x[GRAMMAR_IDX] += r.normal(1.2, .3, len(GRAMMAR_IDX)); x[:40] += 0.8 * protos[6, :40] * us['gain']
                X.append(x); y.append(e); u.append(ui)
    return np.array(X), np.array(y), np.array(u)


def calibrate(X, u, y, use=True):
    """Subtract each user's mean over their Neutral frames (what a 3-second calibration captures)."""
    if not use: return X.copy()
    Z = X.copy()
    for ui in np.unique(u):
        m = u == ui; Z[m] -= X[m & (y == 0)].mean(0)
    return Z
