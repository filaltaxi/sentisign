"""Extract per-frame hand + pose landmarks from WLASL videos with MediaPipe Tasks (VIDEO mode).
Face anchors come from pose landmarks (nose, mouth, eyes, ears) so no face model is needed for signs.
Writes data/npz/<video_id>.npz (hands (T,2,21,3) [L,R], pose (T,33,3), anchors (T,5,3)) and
data/index_<N>.csv (path,gloss,signer,split). Resumable: skips videos already extracted.
Usage: python scripts/extract_wlasl.py --root data/wlasl --subset 100 --workers 6"""
import argparse, json, os, sys, csv, time, urllib.request
from multiprocessing import Pool
import numpy as np

HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
POSE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODELS = "models"


def fetch_models():
    os.makedirs(MODELS, exist_ok=True)
    for url in (HAND_URL, POSE_URL):
        p = os.path.join(MODELS, url.rsplit("/", 1)[-1])
        if not os.path.exists(p):
            print("downloading", p, flush=True); urllib.request.urlretrieve(url, p)


def anchors_from_pose(pose):
    """nose, chin proxy (mouth centre pushed down), forehead proxy (eyes pushed up), L ear, R ear."""
    nose, eyes = pose[0], (pose[2] + pose[5]) / 2
    mouth = (pose[9] + pose[10]) / 2
    chin = mouth + (mouth - nose) * 0.8
    forehead = eyes + (eyes - mouth) * 0.8
    return np.stack([nose, chin, forehead, pose[7], pose[8]])


_hl = _pl = None
def _init():
    global _hl, _pl
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    _hl = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODELS, "hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
    _pl = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODELS, "pose_landmarker_lite.task")),
        running_mode=vision.RunningMode.VIDEO))


def extract(job):
    vid_path, out_path, f0, f1 = job
    if os.path.exists(out_path): return out_path, "skip"
    import cv2, mediapipe as mp
    global _hl, _pl
    _init()  # fresh landmarkers per video: VIDEO mode needs increasing timestamps
    cap = cv2.VideoCapture(vid_path); fps = cap.get(cv2.CAP_PROP_FPS) or 25
    H, P, A = [], [], []; i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        i += 1
        if i < f0 or (f1 > 0 and i > f1): continue
        h, w = fr.shape[:2]; sx = w / h   # make x isotropic with y
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        ts = int(i * 1000 / fps)
        hr = _hl.detect_for_video(img, ts); pr = _pl.detect_for_video(img, ts)
        hands = np.full((2, 21, 3), np.nan, np.float32)
        for lm, hd in zip(hr.hand_landmarks, hr.handedness):
            hands[0 if hd[0].category_name == "Left" else 1] = [[q.x * sx, q.y, q.z * sx] for q in lm]
        if pr.pose_landmarks:
            pose = np.array([[q.x * sx, q.y, q.z * sx] for q in pr.pose_landmarks[0]], np.float32)
        else:
            pose = np.full((33, 3), np.nan, np.float32)
        H.append(hands); P.append(pose); A.append(anchors_from_pose(pose))
    cap.release(); _hl.close(); _pl.close()
    if len(H) < 8: return out_path, "too_short"
    P = np.array(P)
    ok = ~np.isnan(P[:, 11, 0])
    if ok.sum() == 0: return out_path, "no_pose"
    for t in range(len(P)):        # fill missing pose frames from nearest detected frame
        if not ok[t]:
            j = np.where(ok)[0][np.argmin(np.abs(np.where(ok)[0] - t))]; P[t] = P[j]; A[t] = anchors_from_pose(P[j])
    np.savez_compressed(out_path, hands=np.array(H), pose=P, anchors=np.array(A))
    return out_path, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/wlasl"); ap.add_argument("--subset", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    fetch_models()
    meta = json.load(open(os.path.join(a.root, "WLASL_v0.3.json")))
    glosses = [e["gloss"] for e in meta][:a.subset]   # WLASL-N = first N glosses of the json
    vids = {}
    for root, _, files in os.walk(a.root):
        for f in files:
            if f.endswith(".mp4"): vids[f[:-4]] = os.path.join(root, f)
    os.makedirs("data/npz", exist_ok=True)
    jobs, rows, missing = [], [], 0
    for e in meta[:a.subset]:
        for inst in e["instances"]:
            v = vids.get(inst["video_id"])
            if v is None: missing += 1; continue
            out = os.path.join("data/npz", inst["video_id"] + ".npz")
            jobs.append((v, out, inst.get("frame_start", 1), inst.get("frame_end", -1)))
            rows.append(dict(path=out, gloss=e["gloss"], signer=inst["signer_id"], split=inst.get("split", "")))
    print(f"WLASL{a.subset}: {len(jobs)} videos found, {missing} listed but missing; workers={a.workers}", flush=True)
    t0, stats = time.time(), {}
    with Pool(a.workers) as pool:
        for k, (out, st) in enumerate(pool.imap_unordered(extract, jobs, chunksize=4), 1):
            stats[st] = stats.get(st, 0) + 1
            if k % 50 == 0 or k == len(jobs):
                el = time.time() - t0
                print(f"  {k}/{len(jobs)} {stats} {el/60:.1f} min elapsed, ~{el/k*(len(jobs)-k)/60:.1f} min left", flush=True)
    good = [r for r in rows if os.path.exists(r["path"])]
    with open(f"data/index_{a.subset}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "gloss", "signer", "split"]); w.writeheader(); w.writerows(good)
    print(f"index written: {len(good)} usable videos, {len(set(r['gloss'] for r in good))} glosses, "
          f"{len(set(r['signer'] for r in good))} signers", flush=True)


if __name__ == "__main__":
    main()
