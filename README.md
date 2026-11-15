# SentiSign

Sign language to emotion-aware speech, in the browser.

SentiSign watches a signer through a webcam, recognises isolated American Sign Language signs, reads the signer's facial expression, writes a natural English sentence and speaks it in a voice that matches the emotion. Sign and emotion recognition run in the browser, so video never leaves the device; only the recognised words and the emotion label go to a small local server for sentence writing and speech.

```
webcam ─► MediaPipe hands + pose + face (browser)
            ├─► sign model, TCN + BiLSTM + attention, ONNX (browser) ─► "YESTERDAY BUY APPLE"
            └─► emotion model on 52 face blendshapes (browser)          ─► Happy
                                   │  WebSocket
                                   ▼
            Flan-T5-base + LoRA (server)   ─► "I bought an apple yesterday!"
            Chatterbox TTS, Kokoro fallback ─► speech with matching expressiveness
```

## Results

All numbers were measured on an RTX 4050 laptop GPU. Sign accuracy is top-1 on **signers the model never saw in training** unless stated otherwise.

| Component | Result | Baseline |
| --- | --- | --- |
| Sign recognition, 300 signs, unseen signers (mean of 3 seeds) | **80.0%** | 40.0% trained on WLASL alone |
| Sign recognition, 100 signs, unseen signers (mean of 3 seeds) | **76.2%** | 45.8% |
| Sign recognition, 300 signs, official WLASL split | **71.3%** | 46.4% |
| Facial emotion, RAF-DB, 7 classes | **61.1%** accuracy, 51.4% macro-F1 | — |
| Gloss to English, Flan-T5-base + LoRA (synthetic test set) | **BLEU 78.5**, 72.7% exact | Rule baseline BLEU 0.4 |
| Speech, full sentence, median | Chatterbox **1.40 s**, Kokoro 0.06 s | — |
| Continuous signing, stitched real clips | **52.2%** word error | 63.8% with the isolated-sign model |

Per-run details are in [`reports/summary.txt`](reports/summary.txt) and the JSON files beside it.

What moved the sign numbers was data, not architecture. Adding [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) took training from 46 to 96 signers, with the same model and the same WLASL test signers, and doubled accuracy on unseen signers. An ablation shows the body-frame location features are worth about 9 to 10 points on their own (44.1% vs 34.2% on 100 signs, WLASL only).

### Honest limits

- The language model's test sentences come from the same templates as its training data (with held-out sign combinations), so BLEU 78.5 shows the task was learned, not that it handles free-form signing.
- Continuous signing is unsolved: about half the words in a stitched sentence are wrong, so the app expects a short pause between signs.
- The vocabulary is the 300 WLASL300 glosses ([`data/vocab300.txt`](data/vocab300.txt)). There is no sign for "I" or "me"; the language model supplies the subject.

## How it works

- **Features** ([`sentisign/features.py`](sentisign/features.py), [`web/features.js`](web/features.js)): each hand becomes four streams in a body frame centred on the shoulders: handshape, location relative to face anchors, movement and palm orientation (216 values per frame, 60 frames per sign). The Python and JavaScript versions match to 1.2e-7 on golden test vectors ([`tests/golden_js.json`](tests/golden_js.json)).
- **Sign model** ([`sentisign/model.py`](sentisign/model.py)): one dilated TCN per stream, a BiLSTM, attention pooling and a classifier; 1.05M parameters, 4.2 MB as ONNX. Temperature-calibrated, with an abstain threshold that shows the top 3 when unsure.
- **Emotion** ([`scripts/emotion_real.py`](scripts/emotion_real.py)): logistic regression on MediaPipe's 52 face blendshapes, with per-user neutral calibration. Eyebrows are kept as input because in ASL raised brows often mark a question.
- **Language** ([`scripts/t5_lora.py`](scripts/t5_lora.py)): Flan-T5-base with a rank-16 LoRA adapter on the attention query and value layers (1.77M trainable of 249M weights), trained for 12 minutes on 17,687 gloss-to-English pairs.
- **Speech** ([`server/app.py`](server/app.py)): Chatterbox with per-emotion exaggeration and guidance settings, Kokoro as a fast fallback. The server writes the sentence and starts speech during the signer's pause, so most of the generation time is hidden.

## Run it (Windows, NVIDIA GPU)

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Double-click `run.bat` once. It creates `env/`, installs PyTorch (CUDA 12.4) and the dependencies, downloads WLASL from Kaggle (needs a Kaggle API token in `%USERPROFILE%\.kaggle\`) and trains the sign models. To only run the app with the included weights, stop after the install step and run `pip install -r requirements.txt` in `env`.
3. Double-click `start_app.bat`. The browser opens `http://127.0.0.1:8000`; allow the camera. The first start downloads Flan-T5-base and Chatterbox from Hugging Face.

The full pipeline is three scripts run in order: `run.bat` (WLASL), `run_aslc.bat` (adds ASL Citizen, which you download yourself under its licence) and `run_stage3.bat` (deployment model, emotion, language model, voice benchmark, continuous signing and a server self-test).

## Repository layout

```
sentisign/   features, model, splits, calibration, endpointing, emotion, language helpers
scripts/     data extraction, training and evaluation scripts
server/      FastAPI app: sentence writing and speech over a WebSocket
web/         browser app: MediaPipe, ONNX Runtime Web, endpointing, UI
deploy/      trained sign and emotion models plus MediaPipe task files
models/      Flan-T5 LoRA adapter
reports/     measured results
```

## Licence

Code: MIT ([`LICENSE`](LICENSE)). Trained weights are for research and non-commercial use only, because of the datasets they were trained on ([`MODEL_LICENSE.md`](MODEL_LICENSE.md)).
