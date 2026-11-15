---
title: SentiSign
emoji: 🤟
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.8.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: ASL signs and facial emotion to spoken English
---

Live demo of [SentiSign](https://github.com/filaltaxi/sentisign). Sign and emotion recognition run in your browser, so your video never leaves your device. Only the recognised words and an emotion label reach this Space, which writes the sentence (Flan-T5-base + LoRA) and speaks it with an emotion-matched voice (Chatterbox) on a shared ZeroGPU. Trained weights are for research and non-commercial use.
