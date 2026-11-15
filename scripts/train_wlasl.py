"""Train + evaluate the multi-stream TCN+BiLSTM+attention model on real WLASL landmarks.
--split official : WLASL's own train/val/test (signers overlap; comparable to published numbers and v1)
--split signer   : signer-independent (no signer in more than one of train/val/test)
--ablate loc,move: drop the body-frame location and movement streams (v1-style features)
Writes reports/<name>.json and reports/<name>.onnx."""
import sys, os, json, csv, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
from sentisign.features import sequence_features, concat, STREAM_DIMS
from sentisign.splits import signer_independent_split
from sentisign.model import MultiStreamSign
from sentisign.calibration import softmax, fit_temperature, ece, abstain_threshold, topk_fallback

ap = argparse.ArgumentParser()
ap.add_argument("--subset", type=int, default=100); ap.add_argument("--split", default="signer", choices=["signer", "official", "random", "all"])
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=150)
ap.add_argument("--ablate", default=""); ap.add_argument("--name", required=True)
ap.add_argument("--extra", default="", help="extra index csv added to TRAINING only (e.g. ASL Citizen)")
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("reports", exist_ok=True)

cache = f"data/features_{a.subset}.npz"
if not os.path.exists(cache):
    rows = list(csv.DictReader(open(f"data/index_{a.subset}.csv")))
    X, G, S, SP = [], [], [], []
    for r in rows:
        z = np.load(r["path"])
        if np.isnan(z["hands"]).all(): continue
        X.append(concat(sequence_features(z["hands"], z["pose"], z["anchors"]))); G.append(r["gloss"]); S.append(r["signer"]); SP.append(r["split"])
    np.savez_compressed(cache, X=np.array(X, np.float32), G=np.array(G), S=np.array(S), SP=np.array(SP))
def featurize(index_csv, cache_path):
    if not os.path.exists(cache_path):
        rows = list(csv.DictReader(open(index_csv)))
        X_, G_, S_, SP_ = [], [], [], []
        for r in rows:
            zz = np.load(r["path"])
            if np.isnan(zz["hands"]).all(): continue
            X_.append(concat(sequence_features(zz["hands"], zz["pose"], zz["anchors"]))); G_.append(r["gloss"]); S_.append(r["signer"]); SP_.append(r["split"])
        np.savez_compressed(cache_path, X=np.array(X_, np.float32), G=np.array(G_), S=np.array(S_), SP=np.array(SP_))
    zz = np.load(cache_path); return zz["X"], zz["G"], zz["S"], zz["SP"]

z = np.load(cache); X, G, S, SP = z["X"], z["G"], z["S"], z["SP"]
n_wlasl = len(X)
if a.extra:
    Xe, Ge, Se, SPe = featurize(a.extra, a.extra.replace(".csv", "_features.npz"))
    keep = np.isin(Ge, np.unique(G)); Xe, Ge, Se = Xe[keep], Ge[keep], Se[keep]
    X = np.concatenate([X, Xe]); G = np.concatenate([G, Ge]); S = np.concatenate([S, Se]); SP = np.concatenate([SP, np.array(["extra"] * len(Xe))])
X = np.nan_to_num(X)
gl = sorted(set(G)); y = np.array([gl.index(g) for g in G])
a.ablate = a.ablate.replace("+", ",")
if a.ablate:
    off = 0
    for k, d in STREAM_DIMS.items():
        if k in a.ablate.split(","): X[..., off:off+d] = 0
        off += d
W = np.arange(n_wlasl)   # splits are computed on WLASL only, so test sets are identical with/without --extra
if a.split == "signer":
    tr, va, te = [W[i] for i in signer_independent_split(S[W], val_frac=0.15, test_frac=0.2, seed=a.seed)]
elif a.split == "official":
    tr, va, te = [np.where(SP[W] == s)[0] for s in ("train", "val", "test")]
elif a.split == "all":   # deployment model: train on everything except a small val set for early stopping/calibration
    p = np.random.default_rng(a.seed).permutation(n_wlasl); n = n_wlasl
    tr, va, te = p[:int(.9*n)], p[int(.9*n):], p[int(.9*n):]
else:
    p = np.random.default_rng(a.seed).permutation(n_wlasl); n = n_wlasl
    tr, va, te = p[:int(.7*n)], p[int(.7*n):int(.8*n)], p[int(.8*n):]
if a.extra: tr = np.concatenate([tr, np.arange(n_wlasl, len(X))])
mu, sd = X[tr].mean((0, 1)), X[tr].std((0, 1)) + 1e-6; Xn = ((X - mu) / sd).astype(np.float32)
T = lambda i: (torch.tensor(Xn[i]), torch.tensor(y[i]).long())
Xtr, ytr = T(tr); Xva, yva = T(va); Xte, yte = T(te)

def aug(x):
    x = x + torch.randn_like(x) * 0.05
    x = torch.roll(x, int(torch.randint(-4, 5, (1,))), 1)
    m = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) < 0.1).float()   # frame dropout
    x = x * (1 - m)
    s = 1 + 0.1 * torch.randn(x.shape[0], 1, x.shape[2], device=x.device)       # per-feature jitter
    return x * s

m = MultiStreamSign(len(gl)).to(dev)
opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=5e-2)
steps = a.epochs * (len(tr) // 32 + 1)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
lf = nn.CrossEntropyLoss(label_smoothing=0.1)
best, bs, bad, t0 = -1, None, 0, time.time()
for ep in range(a.epochs):
    m.train(); perm = torch.randperm(len(Xtr))
    for i in range(0, len(perm), 32):
        b = perm[i:i+32]; xb = aug(Xtr[b].to(dev))
        opt.zero_grad(); lf(m(xb), ytr[b].to(dev)).backward(); opt.step(); sch.step()
    m.eval()
    with torch.no_grad(): acc = (m(Xva.to(dev)).argmax(1).cpu() == yva).float().mean().item()
    if acc > best: best, bs, bad = acc, {k: v.clone() for k, v in m.state_dict().items()}, 0
    else: bad += 1
    if ep % 10 == 0: print(f"  ep {ep} val {acc:.3f} best {best:.3f}", flush=True)
    if bad >= 30: break
m.load_state_dict(bs); m.eval()
with torch.no_grad(): lv, lt = m(Xva.to(dev)).cpu().numpy(), m(Xte.to(dev)).cpu().numpy()
Tm = fit_temperature(lv, y[va]); pt = softmax(lt / Tm); pr = softmax(lt)
thr, _ = abstain_threshold(softmax(lv / Tm), y[va], 0.8); acc_ = pt.argmax(1) == y[te]; mk = pt.max(1) >= thr
res = dict(name=a.name, extra=a.extra, n_extra=int(len(X) - n_wlasl), subset=a.subset, split=a.split, seed=a.seed, ablate=a.ablate, device=dev,
           gpu=torch.cuda.get_device_name(0) if dev == "cuda" else None,
           n_classes=len(gl), n_train=len(tr), n_val=len(va), n_test=len(te),
           n_signers_train=len(set(S[tr])), n_signers_test=len(set(S[te])),
           top1=float(acc_.mean()), top5=float(np.mean([y[te][i] in np.argsort(-pt[i])[:5] for i in range(len(pt))])),
           val_best=best, ece_raw=ece(pr, y[te]), ece_cal=ece(pt, y[te]), temperature=float(Tm),
           abstain_thr_for_80pct=thr, coverage=float(mk.mean()), acc_on_accepted=float(acc_[mk].mean()) if mk.any() else None,
           top3_when_abstaining=topk_fallback(pt, y[te], thr, 3), epochs_run=ep + 1, train_min=(time.time() - t0) / 60)
json.dump(res, open(f"reports/{a.name}.json", "w"), indent=1)
json.dump(dict(glosses=gl, mu=mu.tolist(), sd=sd.tolist(), temperature=float(Tm), abstain_thr=float(thr), T=60,
               streams=list(STREAM_DIMS.items())), open(f"reports/{a.name}_meta.json", "w"))
print(json.dumps(res), flush=True)
torch.onnx.export(m.cpu(), torch.zeros(1, 60, X.shape[-1]), f"reports/{a.name}.onnx", opset_version=17,
                  input_names=["x"], output_names=["logits"], dynamo=False)
