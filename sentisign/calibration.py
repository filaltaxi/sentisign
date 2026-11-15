import numpy as np


def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def fit_temperature(logits, y, grid=np.concatenate([np.linspace(0.1, 5, 99), np.linspace(5.5, 30, 50)])):
    best, bt = 1e9, 1.0
    for t in grid:
        p = softmax(logits / t); nll = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        if nll < best: best, bt = nll, t
    return bt


def ece(probs, y, bins=15):
    conf = probs.max(1); pred = probs.argmax(1); acc = (pred == y)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def abstain_threshold(probs, y, target_acc=0.9):
    """Smallest confidence threshold whose accepted-set accuracy >= target."""
    conf = probs.max(1); acc = probs.argmax(1) == y
    for t in np.linspace(0, 0.99, 100):
        m = conf >= t
        if m.sum() >= 10 and acc[m].mean() >= target_acc: return float(t), float(m.mean())
    return 1.0, 0.0


def topk_fallback(probs, y, thr, k=3):
    """When the model abstains, show top-k choices; returns how often the right sign is among them."""
    low = probs.max(1) < thr
    if not low.any(): return None
    topk = np.argsort(-probs[low], 1)[:, :k]
    return float(np.mean([y[low][i] in topk[i] for i in range(low.sum())]))
