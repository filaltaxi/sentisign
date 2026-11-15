"""SentiSign v2 server. Serves the browser app and model files, and turns glosses + emotion into a spoken sentence.
Speculative generation: the client sends 'prepare' after every recognised sign (during the pause); the sentence and
audio are computed then, so 'speak' usually returns from cache. Falls back gracefully:
Flan-T5 LoRA -> rule baseline; Chatterbox -> Kokoro -> text only."""
import asyncio, base64, io, json, os, sys, time, wave
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sentisign.language import baseline, tts_params

app = FastAPI()
pool = ThreadPoolExecutor(1)          # one GPU worker: requests are serialised, no VRAM contention
LOG = os.path.join(ROOT, "logs", "server_latency.jsonl")


class Composer:
    def __init__(self):
        self.kind = "rule baseline"
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            from peft import PeftModel
            path = os.path.join(ROOT, "models", "t5_lora")
            if os.path.isdir(path):
                self.tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
                base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
                self.m = PeftModel.from_pretrained(base, path).merge_and_unload().eval()
                self.dev = "cuda" if torch.cuda.is_available() else "cpu"; self.m.to(self.dev)
                self.kind = "flan-t5-base + LoRA"
        except Exception as e:
            print("composer fallback:", e)

    def __call__(self, glosses, emotion):
        if self.kind == "rule baseline": return baseline(glosses)
        import torch
        prompt = f"Translate ASL gloss to natural English. Emotion: {emotion}. Gloss: {' '.join(g.upper() for g in glosses)}"
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.dev)
        with torch.no_grad(): out = self.m.generate(input_ids=ids, max_new_tokens=40, num_beams=4)
        return self.tok.decode(out[0], skip_special_tokens=True)


class Voice:
    def __init__(self):
        self.kind = None
        try:
            import torch
            try:
                import perth  # Chatterbox watermarker; its native part fails to load on some Windows installs -> fall back to the no-op one
                if getattr(perth, "PerthImplicitWatermarker", None) is None: perth.PerthImplicitWatermarker = perth.DummyWatermarker
            except Exception:
                pass
            from chatterbox.tts import ChatterboxTTS
            self.cb = ChatterboxTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu"); self.kind = "chatterbox"
        except Exception as e:
            print("chatterbox unavailable:", e)
            try:
                from kokoro import KPipeline
                self.kp = KPipeline(lang_code="a"); self.kind = "kokoro"
            except Exception as e2:
                print("kokoro unavailable:", e2)

    def __call__(self, text, emotion):
        import numpy as np
        if self.kind == "chatterbox":
            ex, cfg = tts_params(emotion)
            wav = self.cb.generate(text, exaggeration=ex, cfg_weight=cfg).squeeze().cpu().numpy(); sr = self.cb.sr
        elif self.kind == "kokoro":
            wav = np.concatenate([np.asarray(a) for _, _, a in self.kp(text, voice="af_heart")]); sr = 24000
        else:
            return None
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes((np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes())
        return base64.b64encode(buf.getvalue()).decode()


composer, voice = Composer(), Voice()
print(f"composer: {composer.kind} | voice: {voice.kind or 'text only'}")


def produce(glosses, emotion, sentence=None):
    t0 = time.perf_counter(); text = sentence or composer(glosses, emotion); t1 = time.perf_counter()
    wav = voice(text, emotion) if text else None; t2 = time.perf_counter()
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f: f.write(json.dumps(dict(n=len(glosses), compose_ms=(t1 - t0) * 1e3, tts_ms=(t2 - t1) * 1e3, voice=voice.kind)) + "\n")
    return text, wav


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept(); loop = asyncio.get_running_loop(); cache = {}
    try:
        while True:
            m = json.loads(await sock.receive_text())
            key = (tuple(m.get("glosses", [])), m.get("emotion", "Neutral"))
            if m["type"] == "prepare" and key[0]:
                cache.clear(); cache[key] = loop.run_in_executor(pool, produce, list(key[0]), key[1])
                text, _ = await cache[key]; await sock.send_text(json.dumps({"type": "sentence", "text": text}))
            elif m["type"] == "speak":
                t0 = time.perf_counter()
                if m.get("sentence"): fut = loop.run_in_executor(pool, produce, list(key[0]), key[1], m["sentence"])
                else: fut = cache.get(key) or loop.run_in_executor(pool, produce, list(key[0]), key[1])
                text, wav = await fut
                await sock.send_text(json.dumps({"type": "sentence", "text": text}))
                await sock.send_text(json.dumps({"type": "audio", "wav": wav, "server_ms": (time.perf_counter() - t0) * 1e3} if wav else {"type": "text_only"}))
                cache.clear()
    except WebSocketDisconnect:
        pass


app.mount("/models", StaticFiles(directory=os.path.join(ROOT, "deploy")), name="models")
app.mount("/", StaticFiles(directory=os.path.join(ROOT, "web"), html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
