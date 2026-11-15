# Hugging Face Space entry point: fetches the SentiSign code from GitHub and starts it.
import os, runpy, subprocess
SRC = os.path.abspath("sentisign_src")
if not os.path.isdir(SRC):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/filaltaxi/sentisign", SRC], check=True)
os.chdir(SRC)
runpy.run_path(os.path.join(SRC, "space_gradio", "app.py"), run_name="__main__")
