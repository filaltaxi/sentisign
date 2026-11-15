import json, glob
rows = []
for f in sorted(glob.glob("reports/*.json")):
    try: r = json.load(open(f))
    except Exception: continue
    if isinstance(r, dict) and "top1" in r and "split" in r: rows.append(r)
print(f"{'run':34s} {'split':8s} {'top1':>6s} {'top5':>6s} {'ECE':>6s} {'cov@80':>6s} {'test signers':>12s} {'extra':>6s}")
for r in rows:
    print(f"{r['name']:34s} {r['split']:8s} {r['top1']*100:6.1f} {r['top5']*100:6.1f} {r['ece_cal']:6.3f} {r['coverage']*100:6.1f} {r['n_signers_test']:12d} {r.get('n_extra', 0):6d}")
for f, keys in (("reports/t5.json", ("t5_bleu", "t5_chrf", "t5_exact", "baseline_bleu", "baseline_chrf", "baseline_exact", "gen_ms_p50")),
                ("reports/emotion_real.json", ("n_train", "n_test", "faces_missed")),
                ("reports/measured_latency.json", None), ("reports/continuous_real_100.json", None)):
    try: r = json.load(open(f))
    except Exception: continue
    print("\n" + f + ":", json.dumps({k: r[k] for k in keys} if keys else r, default=str)[:600])
    if "emotion" in f: print("  logistic", r["logistic"]["macro_f1"], "mlp", r["mlp"]["macro_f1"], "no-brows", r["logistic_without_brows"]["macro_f1"])
