# Hugging Face Space (Docker SDK, free CPU). Builds the SentiSign app from the GitHub repo.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface \
    SENTISIGN_VOICE=kokoro HOST=0.0.0.0 PORT=7860
WORKDIR /home/user
RUN git clone --depth 1 https://github.com/filaltaxi/sentisign app
WORKDIR /home/user/app
RUN pip install --no-cache-dir --user -r space/requirements.txt
# Download Flan-T5-base and Kokoro at build time so the first visitor does not wait for them.
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM as M; AutoTokenizer.from_pretrained('google/flan-t5-base'); M.from_pretrained('google/flan-t5-base')" && \
    python -c "from kokoro import KPipeline; p = KPipeline(lang_code='a'); list(p('Hello.', voice='af_heart'))"
EXPOSE 7860
CMD ["python", "server/app.py"]
