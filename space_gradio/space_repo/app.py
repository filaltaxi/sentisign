# Hugging Face Space entry point: fetches the SentiSign code from GitHub and starts it.
import os, runpy, subprocess, sys
# chatterbox-tts pins torch 2.6, which has no kernels for the Space's Blackwell GPU; install it without its pins.
subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "chatterbox-tts==0.1.7"], check=True)
SRC = os.path.abspath("sentisign_src")
if not os.path.isdir(SRC):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/filaltaxi/sentisign", SRC], check=True)
os.chdir(SRC)
runpy.run_path(os.path.join(SRC, "space_gradio", "app.py"), run_name="__main__")
