"""Measure first-audio latency per engine on GPU. Each engine wrapped so a failed install is logged, not fatal."""
import argparse, json, time, sys
try:
    import perth  # Chatterbox watermarker; its native part fails to load on some Windows installs -> fall back to the no-op one
    if getattr(perth, "PerthImplicitWatermarker", None) is None: perth.PerthImplicitWatermarker = perth.DummyWatermarker
except Exception:
    pass
sys.path.insert(0, "."); from sentisign.language import TTS
ap = argparse.ArgumentParser(); ap.add_argument("--engines", default="chatterbox"); ap.add_argument("--n", type=int, default=50); ap.add_argument("--out")
a = ap.parse_args(); TEXT = ["I would like some water, please.", "Where is the nearest hospital?", "Thank you so much for helping me!"]
res = {}
for eng in a.engines.split(","):
    try:
        if eng == "chatterbox":
            from chatterbox.tts import ChatterboxTTS; model = ChatterboxTTS.from_pretrained(device="cuda")
            run = lambda t, emo: model.generate(t, exaggeration=TTS[emo][0], cfg_weight=TTS[emo][1])
        elif eng == "kokoro":
            from kokoro import KPipeline; pipe = KPipeline(lang_code="a")
            run = lambda t, emo: [a for _, _, a in pipe(t, voice="af_heart")]
        else:
            raise RuntimeError(f"{eng}: add wrapper (CosyVoice2 needs its repo cloned)")
        ts = []
        for i in range(a.n):
            emo = list(TTS)[i % 7]; t0 = time.time(); run(TEXT[i % 3], emo); ts.append((time.time() - t0) * 1000)
        ts.sort(); res[eng] = {"p50_ms": ts[len(ts)//2], "p95_ms": ts[int(len(ts)*.95)], "note": "full-utterance time; streaming first-chunk is lower"}
    except Exception as e:
        res[eng] = {"error": repr(e)}
json.dump(res, open(a.out, "w"), indent=1); print(res)
