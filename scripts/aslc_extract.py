"""ASL Citizen -> landmarks for glosses that also appear in WLASL-N. Reads videos straight from the zip
(no full unzip, saves ~43 GB). Writes data/npz_aslc/<id>.npz and data/index_aslc_<N>.csv
(path,gloss,signer,split) with WLASL gloss names and signer ids prefixed 'aslc_'."""
import argparse, csv, io, json, os, re, sys, tempfile, time, zipfile
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.extract_wlasl import extract, fetch_models

ZIP = None


def norm(g):
    g = g.lower().strip()
    g = re.sub(r"[^a-z ]", "", re.sub(r"\d+$", "", g))
    return g.strip()


def work(job):
    name, out, gloss = job
    if os.path.exists(out):
        try:
            import numpy as np
            with np.load(out) as d: [d[k].shape for k in d.files]
            return out, "skip"
        except Exception:
            os.remove(out)  # half-written file from an interrupted run: redo it
    with zipfile.ZipFile(ZIP) as z, tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "v.mp4")
        with z.open(name) as src, open(p, "wb") as dst: dst.write(src.read())
        return extract((p, out, 1, -1))


def init(zp):
    global ZIP
    ZIP = zp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True); ap.add_argument("--subset", type=int, default=300)
    ap.add_argument("--wlasl_root", default="data/wlasl"); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--max_per_gloss", type=int, default=60)
    a = ap.parse_args()
    fetch_models()
    meta = json.load(open(os.path.join(a.wlasl_root, "WLASL_v0.3.json")))
    wl = {norm(e["gloss"]): e["gloss"] for e in meta[:a.subset]}
    z = zipfile.ZipFile(a.zip)
    names = z.namelist()
    vids = {os.path.basename(n): n for n in names if n.lower().endswith(".mp4")}
    csvs = [n for n in names if n.lower().endswith(".csv") and "split" in n.lower()]
    print(f"zip: {len(vids)} videos, split files: {csvs}", flush=True)
    rows, jobs, per = [], [], {}
    seen_gl = set()
    for c in csvs:
        split = os.path.splitext(os.path.basename(c))[0]
        rd = csv.DictReader(io.TextIOWrapper(z.open(c), encoding="utf-8-sig"))
        cols = {k.lower(): k for k in rd.fieldnames}
        pc = next(cols[k] for k in cols if "participant" in k); vc = next(cols[k] for k in cols if "video" in k)
        gc = next(cols[k] for k in cols if k.startswith("gloss"))
        for r in rd:
            g = wl.get(norm(r[gc]))
            if g is None: continue
            seen_gl.add(g)
            if per.get(g, 0) >= a.max_per_gloss: continue
            v = vids.get(os.path.basename(r[vc]))
            if v is None: continue
            per[g] = per.get(g, 0) + 1
            out = os.path.join("data/npz_aslc", os.path.splitext(os.path.basename(v))[0] + ".npz")
            jobs.append((v, out, g)); rows.append(dict(path=out, gloss=g, signer="aslc_" + r[pc], split="train"))
    z.close()
    print(f"overlap: {len(seen_gl)} of {len(wl)} WLASL{a.subset} glosses found in ASL Citizen; {len(jobs)} clips to extract (cap {a.max_per_gloss}/gloss)", flush=True)
    os.makedirs("data/npz_aslc", exist_ok=True)
    t0, stats = time.time(), {}
    with Pool(a.workers, initializer=init, initargs=(a.zip,)) as pool:
        for k, (out, st) in enumerate(pool.imap_unordered(work, jobs, chunksize=4), 1):
            stats[st] = stats.get(st, 0) + 1
            if k % 200 == 0 or k == len(jobs):
                el = time.time() - t0
                print(f"  {k}/{len(jobs)} {stats} {el/60:.1f} min, ~{el/k*(len(jobs)-k)/60:.1f} min left", flush=True)
    good = [r for r in rows if os.path.exists(r["path"])]
    with open(f"data/index_aslc_{a.subset}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "gloss", "signer", "split"]); w.writeheader(); w.writerows(good)
    print(f"ASL Citizen index: {len(good)} clips, {len(set(r['gloss'] for r in good))} glosses, {len(set(r['signer'] for r in good))} signers", flush=True)


if __name__ == "__main__":
    main()
