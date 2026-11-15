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


# ZeroGPU only registers @spaces.GPU functions when Gradio itself starts the server (demo.launch).
# The browser app is served through Gradio's own file route, so the Space root redirects to it.
WEB, DEPLOY = os.path.join(ROOT, "web"), os.path.join(ROOT, "deploy")
APP_URL = "/gradio_api/file=" + os.path.join(WEB, "index.html")
REDIRECT = "() => { location.replace('%s'); }" % APP_URL
with gr.Blocks(title="SentiSign") as demo:
    gr.Markdown(f"Opening the SentiSign app... If nothing happens, [click here]({APP_URL}).")
    with gr.Row(visible=False):
        g, e, s, t, w = gr.Textbox(), gr.Textbox(), gr.Textbox(), gr.Textbox(), gr.Textbox()
        gr.Button("compose").click(compose, [g, e], t, api_name="compose")
        gr.Button("speak").click(speak, [g, e, s], [t, w], api_name="speak")
demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
                server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
                allowed_paths=[WEB, DEPLOY], js=REDIRECT)
