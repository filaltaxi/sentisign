"""Real-face emotion model from MediaPipe face blendshapes (52 values), trained on RAF-DB.
All 52 blendshapes are kept, brows included: the simulation showed hiding brow features makes raised question
brows read as Surprise more often, so the model needs them as context.
Auto-detects RAF-DB layout: label folders 1..7 (official RAF-DB order), emotion-named folders, or CSV (image,label).
Writes data/emotion_blendshapes.npz, reports/emotion_real.json and deploy/emotion.json (logistic weights for the browser).
Note: RAF-DB has no identities, so per-user neutral calibration is applied only in the app, not measured here."""
import argparse, csv, glob, json, os, sys, time, urllib.request
import numpy as np

RAF = {1: "Surprise", 2: "Fear", 3: "Disgust", 4: "Happy", 5: "Sad", 6: "Angry", 7: "Neutral"}
EMOS = ["Neutral", "Happy", "Sad", "Angry", "Fear", "Surprise", "Disgust"]
NAMES = {"surprise": "Surprise", "fear": "Fear", "disgust": "Disgust", "happy": "Happy", "happiness": "Happy", "sad": "Sad",
         "sadness": "Sad", "angry": "Angry", "anger": "Angry", "neutral": "Neutral"}
FACE_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"


def find_items(root):
    items = []
    csvs = [c for c in glob.glob(os.path.join(root, "**", "*label*.csv"), recursive=True)]
    imgs = {}
    for p in glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True) + glob.glob(os.path.join(root, "**", "*.png"), recursive=True):
        b = os.path.basename(p); imgs[b] = p; imgs[b.replace("_aligned", "")] = p   # RAF-DB csv names omit "_aligned"
    for p in set(imgs.values()):
        parts = p.replace("\\", "/").split("/")
        split = "test" if any(x.lower() in ("test", "val", "validation") for x in parts) else "train"
        lab = parts[-2]
        if lab.isdigit() and int(lab) in RAF: items.append((p, RAF[int(lab)], split))
        elif lab.lower() in NAMES: items.append((p, NAMES[lab.lower()], split))
    if items: return items, "folders"
    for c in csvs:
        split = "test" if "test" in os.path.basename(c).lower() else "train"
        for r in csv.reader(open(c)):
            if len(r) < 2 or not r[1].strip().isdigit() or int(r[1]) not in RAF: continue
            p = imgs.get(os.path.basename(r[0]))
            if p: items.append((p, RAF[int(r[1])], split))
    return items, "csv"


def extract(items):
    import cv2, mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    os.makedirs("models", exist_ok=True); fp = "models/face_landmarker.task"
    if not os.path.exists(fp): urllib.request.urlretrieve(FACE_URL, fp)
    fl = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=fp),
                                                   output_face_blendshapes=True, num_faces=1))
    X, y, sp, miss, t0 = [], [], [], 0, time.time()
    for k, (p, lab, split) in enumerate(items, 1):
        im = cv2.imread(p)
        if im is None: miss += 1; continue
        im = cv2.resize(im, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)      # RAF-DB aligned faces are 100x100
        im = cv2.copyMakeBorder(im, 60, 60, 60, 60, cv2.BORDER_REPLICATE)         # margin so the detector sees a whole face
        r = fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        if not r.face_blendshapes: miss += 1; continue
        X.append([c.score for c in r.face_blendshapes[0]]); y.append(EMOS.index(lab)); sp.append(split)
        if k % 1000 == 0: print(f"  {k}/{len(items)} images, {miss} without a detected face, {(time.time()-t0)/60:.1f} min", flush=True)
    return np.array(X, np.float32), np.array(y), np.array(sp), miss


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    cache = "data/emotion_blendshapes.npz"
    if os.path.exists(cache):
        z = np.load(cache); X, y, sp, miss = z["X"], z["y"], z["sp"], int(z["miss"])
    else:
        items, how = find_items(a.root); print(f"found {len(items)} labelled images via {how}", flush=True)
        if not items: sys.exit("ERROR: no labelled images found under " + a.root)
        X, y, sp, miss = extract(items); np.savez_compressed(cache, X=X, y=y, sp=sp, miss=miss)
    tr, te = sp == "train", sp == "test"
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Z = (X - mu) / sd
    res = dict(n_train=int(tr.sum()), n_test=int(te.sum()), faces_missed=miss, labels=EMOS,
               class_counts_test={EMOS[i]: int((y[te] == i).sum()) for i in range(7)})
    lr = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0).fit(Z[tr], y[tr]); p = lr.predict(Z[te])
    res["logistic"] = dict(acc=accuracy_score(y[te], p), macro_f1=f1_score(y[te], p, average="macro"), confusion=confusion_matrix(y[te], p).tolist())
    mlp = MLPClassifier((128, 64), alpha=1e-3, max_iter=400, early_stopping=True, random_state=0).fit(Z[tr], y[tr]); q = mlp.predict(Z[te])
    res["mlp"] = dict(acc=accuracy_score(y[te], q), macro_f1=f1_score(y[te], q, average="macro"))
    brow = [i for i in range(52) if i in (1, 2, 3, 4, 5)]   # browDownLeft/Right, browInnerUp, browOuterUpLeft/Right
    keep = [i for i in range(52) if i not in brow]
    lr2 = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Z[tr][:, keep], y[tr]); p2 = lr2.predict(Z[te][:, keep])
    res["logistic_without_brows"] = dict(acc=accuracy_score(y[te], p2), macro_f1=f1_score(y[te], p2, average="macro"))
    os.makedirs("reports", exist_ok=True); os.makedirs("deploy", exist_ok=True)
    json.dump(res, open("reports/emotion_real.json", "w"), indent=1)
    json.dump(dict(labels=EMOS, W=lr.coef_.tolist(), b=lr.intercept_.tolist(), mu=mu.tolist(), sd=sd.tolist()), open("deploy/emotion.json", "w"))
    print(json.dumps({k: v for k, v in res.items() if k != "logistic" or True}, default=str)[:1500], flush=True)


if __name__ == "__main__":
    main()
