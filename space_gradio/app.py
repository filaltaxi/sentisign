"""SentiSign on a Hugging Face Gradio Space with ZeroGPU.
Serves the same browser app as the laptop server; the browser calls two Gradio API endpoints
(/compose, /speak) that borrow a shared GPU per call. Run from the repo root: python space_gradio/app.py"""
import os, sys
os.environ.setdefault("SENTISIGN_MODE", "gradio")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "server"))
import spaces                      # must be imported before torch touches CUDA
import gradio as gr
import app as server               # loads Flan-T5 + LoRA and Chatterbox (falls back to rule baseline / text only)


@spaces.GPU(duration=20)
def compose(glosses: str, emotion: str) -> str:
    return server.composer(glosses.split(), emotion or "Neutral") if glosses.strip() else ""


@spaces.GPU(duration=60)
def speak(glosses: str, emotion: str, sentence: str):
    text, wav = server.produce(glosses.split(), emotion or "Neutral", sentence or None)
    return text, wav or ""


with gr.Blocks(title="SentiSign API") as demo:
    gr.Markdown("SentiSign API. Open the app at the Space's root URL.")
    g, e, s, t, w = gr.Textbox(label="glosses"), gr.Textbox(label="emotion"), gr.Textbox(label="sentence"), gr.Textbox(label="text"), gr.Textbox(label="wav (base64)")
    gr.Button("compose").click(compose, [g, e], t, api_name="compose")
    gr.Button("speak").click(speak, [g, e, s], [t, w], api_name="speak")
demo.queue(default_concurrency_limit=2)

app = gr.mount_gradio_app(server.app, demo, path="/gradio")
server.mount_static(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
