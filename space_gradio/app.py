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


# ZeroGPU only registers @spaces.GPU functions when Gradio itself starts the server (demo.launch),
# so the browser app is added to Gradio's own FastAPI app after launch, ahead of Gradio's routes.
REDIRECT = "() => { if (!location.pathname.startsWith('/sentisign')) location.replace('/sentisign/'); }"
with gr.Blocks(title="SentiSign") as demo:
    gr.Markdown("Opening the SentiSign app... If nothing happens, go to [/sentisign/](/sentisign/).")
    with gr.Row(visible=False):
        g, e, s, t, w = gr.Textbox(), gr.Textbox(), gr.Textbox(), gr.Textbox(), gr.Textbox()
        gr.Button("compose").click(compose, [g, e], t, api_name="compose")
        gr.Button("speak").click(speak, [g, e, s], [t, w], api_name="speak")
demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    from starlette.routing import Mount, Route
    from starlette.responses import JSONResponse
    from starlette.staticfiles import StaticFiles
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
                server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)), prevent_thread_lock=True, js=REDIRECT)
    routes = demo.app.router.routes
    routes.insert(0, Route("/api/mode", lambda request: JSONResponse({"mode": "gradio"})))
    routes.insert(0, Mount("/models", app=StaticFiles(directory=os.path.join(ROOT, "deploy")), name="models"))
    routes.insert(0, Mount("/sentisign", app=StaticFiles(directory=os.path.join(ROOT, "web"), html=True), name="web"))
    demo.block_thread()
