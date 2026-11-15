"""Flan-T5-base + LoRA for gloss->English with emotion tag. Data: jsonl {"gloss": "ME WANT WATER", "emotion": "Happy", "text": "..."}.
Plain PyTorch loop (no `datasets`/pyarrow, which conflict with the numpy<2 pin that Chatterbox needs)."""
import argparse, json, os, random, sys, time, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model
import sacrebleu
sys.path.insert(0, "."); from sentisign.language import baseline
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="google/flan-t5-base"); ap.add_argument("--train"); ap.add_argument("--eval"); ap.add_argument("--out")
ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=5e-4)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(a.model); m = AutoModelForSeq2SeqLM.from_pretrained(a.model)
m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q", "v"], task_type="SEQ_2_SEQ_LM")).to(dev)
m.print_trainable_parameters()
prompt = lambda e: f"Translate ASL gloss to natural English. Emotion: {e['emotion']}. Gloss: {e['gloss']}"
load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
train, test = load(a.train), load(a.eval)
print(f"{len(train)} train, {len(test)} test", flush=True)
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr)
steps = a.epochs * ((len(train) + a.bs - 1) // a.bs); sch = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, total_iters=steps)
use_bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()
m.train(); k = 0; t0 = time.time()
for ep in range(a.epochs):
    random.Random(ep).shuffle(train)
    for i in range(0, len(train), a.bs):
        b = train[i:i + a.bs]
        x = tok([prompt(e) for e in b], max_length=64, truncation=True, padding=True, return_tensors="pt").to(dev)
        y = tok([e["text"] for e in b], max_length=64, truncation=True, padding=True, return_tensors="pt").input_ids.to(dev)
        y[y == tok.pad_token_id] = -100
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_bf16):
            loss = m(**x, labels=y).loss
        opt.zero_grad(); loss.backward(); opt.step(); sch.step(); k += 1
        if k % 100 == 0: print(f"  ep {ep} step {k}/{steps} loss {loss.item():.3f} {(time.time()-t0)/60:.1f} min", flush=True)
os.makedirs("models", exist_ok=True); m.save_pretrained("models/t5_lora"); tok.save_pretrained("models/t5_lora")
m.eval(); preds, t_gen = [], []
with torch.no_grad():
    for e in test:
        ids = tok(prompt(e), return_tensors="pt").input_ids.to(dev)
        t1 = time.perf_counter(); out = m.generate(input_ids=ids, max_new_tokens=40, num_beams=4); t_gen.append((time.perf_counter() - t1) * 1e3)
        preds.append(tok.decode(out[0], skip_special_tokens=True))
refs = [e["text"] for e in test]; base = [baseline(e["gloss"].split()) for e in test]
res = {"t5_bleu": sacrebleu.corpus_bleu(preds, [refs]).score, "t5_chrf": sacrebleu.corpus_chrf(preds, [refs]).score,
       "baseline_bleu": sacrebleu.corpus_bleu(base, [refs]).score, "baseline_chrf": sacrebleu.corpus_chrf(base, [refs]).score,
       "t5_exact": sum(p.strip() == r.strip() for p, r in zip(preds, refs)) / len(refs), "baseline_exact": sum(b.strip() == r.strip() for b, r in zip(base, refs)) / len(refs),
       "gen_ms_p50": sorted(t_gen)[len(t_gen) // 2], "n_test": len(refs), "train_min": (time.time() - t0) / 60,
       "samples": list(zip([e["gloss"] for e in test][:25], [e["emotion"] for e in test][:25], preds[:25], refs[:25]))}
json.dump(res, open(a.out, "w"), indent=1); print(json.dumps({k: v for k, v in res.items() if k != "samples"}))
