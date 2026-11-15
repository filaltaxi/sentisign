"""Continuous signing on REAL landmarks: stitch WLASL clips from the same signer into 3-5 sign sentences with
blended transitions (hands and body interpolated over 8 frames, clip edges trimmed 10%), then decode with a
sliding window. Compares (A) isolated-sign model vs (B) model trained on sentence windows with a 'between signs'
class. Thresholds tuned on validation signers; WER reported on unseen test signers. Writes reports/continuous_real.json."""
import sys, os, csv, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
from sentisign.features import sequence_features, concat
from sentisign.splits import signer_independent_split
from sentisign.model import MultiStreamSign
from sentisign.calibration import softmax

ap = argparse.ArgumentParser(); ap.add_argument("--subset", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=60); a = ap.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = "cuda" if torch.cuda.is_available() else "cpu"
WIN, STRIDE, TRANS = 40, 4, 8

rows = list(csv.DictReader(open(f"data/index_{a.subset}.csv")))
clips = []
for r in rows:
    z = np.load(r["path"])
    if np.isnan(z["hands"]).all() or len(z["hands"]) < 12: continue
    n = len(z["hands"]); c = int(.1 * n)
    clips.append(dict(h=z["hands"][c:n - c], p=z["pose"][c:n - c], a=z["anchors"][c:n - c], g=r["gloss"], s=r["signer"]))
gl = sorted({c["g"] for c in clips}); BLANK = len(gl); gi = {g: i for i, g in enumerate(gl)}
S = np.array([c["s"] for c in clips])
tr, va, te = signer_independent_split(S, val_frac=0.15, test_frac=0.2, seed=a.seed)
by_signer = lambda idx: {s: [i for i in idx if clips[i]["s"] == s] for s in set(S[idx])}


def stitch(ids):
    H, P, A, bounds, t = [], [], [], [], 0
    for k, i in enumerate(ids):
        c = clips[i]
        if k:  # blend from previous clip's last frame to this clip's first frame
            w = np.linspace(0, 1, TRANS + 2)[1:-1][:, None, None]
            prevH, prevP, prevA = H[-1][-1], P[-1][-1], A[-1][-1]
            hb = np.nan_to_num(prevH)[None] * (1 - w[..., None]) + np.nan_to_num(c["h"][0])[None] * w[..., None]
            H.append(hb); P.append(prevP[None] * (1 - w) + c["p"][0][None] * w); A.append(prevA[None] * (1 - w) + c["a"][0][None] * w); t += TRANS
        bounds.append((t, t + len(c["h"]))); H.append(c["h"]); P.append(c["p"]); A.append(c["a"]); t += len(c["h"])
    return np.concatenate(H), np.concatenate(P), np.concatenate(A), bounds


def sentences(idx, n):
    bs = {s: v for s, v in by_signer(idx).items() if len(v) >= 3}; out = []
    keys = sorted(bs)
    for _ in range(n):
        s = keys[rng.integers(len(keys))]; k = int(rng.integers(3, min(6, len(bs[s]) + 1)))
        out.append(list(rng.choice(bs[s], k, replace=False)))
    return out


def feat(h, p, a): return concat(sequence_features(h, p, a, T=WIN))


def windows(idx, n_sent):
    X, y = [], []
    for i in idx: X.append(feat(clips[i]["h"], clips[i]["p"], clips[i]["a"])); y.append(gi[clips[i]["g"]])
    for ids in sentences(idx, n_sent):
        h, p, an, b = stitch(ids)
        for st in range(0, len(h) - WIN + 1, STRIDE):
            cov = [max(0, min(hi, st + WIN) - max(lo, st)) / WIN for lo, hi in b]; k = int(np.argmax(cov))
            X.append(feat(h[st:st + WIN], p[st:st + WIN], an[st:st + WIN])); y.append(gi[clips[ids[k]]["g"]] if cov[k] >= .6 else BLANK)
    return np.nan_to_num(np.array(X, np.float32)), np.array(y)


def fit(X, y, ncls):
    mu, sd = X.mean((0, 1)), X.std((0, 1)) + 1e-6; Xt = torch.tensor((X - mu) / sd); yt = torch.tensor(y).long()
    w = torch.ones(ncls); w[-1] = .5 if ncls > len(gl) else 1
    m = MultiStreamSign(ncls).to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=5e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=a.epochs * (len(Xt) // 64 + 1)); lf = nn.CrossEntropyLoss(weight=w.to(dev), label_smoothing=.1)
    for ep in range(a.epochs):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]; xb = Xt[b].to(dev) + torch.randn(len(b), *Xt.shape[1:], device=dev) * .05
            opt.zero_grad(); lf(m(xb), yt[b].to(dev)).backward(); opt.step(); sch.step()
    return m.eval(), mu, sd


def wer(ref, hyp):
    d = np.arange(len(hyp) + 1)
    for i, x in enumerate(ref, 1):
        nd = [i]
        for j, h in enumerate(hyp, 1): nd.append(min(d[j] + 1, nd[j - 1] + 1, d[j - 1] + (x != h)))
        d = np.array(nd)
    return d[-1] / len(ref)


def decode(m, mu, sd, h, p, an, thr, blank):
    W = np.nan_to_num(np.array([feat(h[s:s + WIN], p[s:s + WIN], an[s:s + WIN]) for s in range(0, len(h) - WIN + 1, STRIDE)], np.float32))
    with torch.no_grad(): pr = softmax(m(torch.tensor((W - mu) / sd).to(dev)).cpu().numpy())
    out, prev, run = [], None, 0
    for q in pr:
        c = int(q.argmax()); lab = c if q.max() >= thr and not (blank and c == BLANK) else None
        run = run + 1 if lab == prev else 1; prev = lab
        if lab is not None and run == 2 and (not out or out[-1] != lab): out.append(lab)   # need 2 consecutive windows
    return out


def evaluate(m, mu, sd, sents, thr, blank):
    W = [wer([gi[clips[i]["g"]] for i in ids], decode(m, mu, sd, *stitch(ids)[:3], thr, blank)) for ids in sents]
    return float(np.mean(W)), float(np.mean(np.array(W) == 0))


t0 = time.time(); res = dict(subset=a.subset, seed=a.seed, n_test_signers=len(set(S[te])))
val_s, test_s = sentences(va, 60), sentences(te, 150)
for name, n_sent, blank in [("A_isolated_model", 0, False), ("B_sentence_windows_plus_blank", 400, True)]:
    X, y = windows(tr, n_sent); m, mu, sd = fit(X, y, len(gl) + (1 if blank else 0))
    best = min((evaluate(m, mu, sd, val_s, t, blank)[0], t) for t in (.15, .25, .35, .5, .65))
    w, perfect = evaluate(m, mu, sd, test_s, best[1], blank)
    res[name] = dict(WER=w, perfect_sentences=perfect, threshold=best[1], train_windows=len(y), blank_windows=int((y == len(gl)).sum()) if blank else 0)
    print(name, res[name], flush=True)
res["minutes"] = (time.time() - t0) / 60
os.makedirs("reports", exist_ok=True); json.dump(res, open(f"reports/continuous_real_{a.subset}.json", "w"), indent=1); print(res)
