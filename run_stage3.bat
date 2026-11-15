@echo off
setlocal
cd /d "%~dp0"
set "LOG=%CD%\logs\run_stage3_log.txt"
set "PY=%CD%\env\python.exe"
echo ==== Stage 3 started %DATE% %TIME% ==== >> "%LOG%"
echo SentiSign v2 stage 3: waits for the ASL Citizen stage, then builds the full app. Progress: logs\run_stage3_log.txt . Do not close.

rem ---- 0. wait until the ASL Citizen stage has finished (so downloads do not compete) ----
:wait
findstr /c:"ALL DONE" "logs\run_aslc_log.txt" >nul 2>&1 && goto :go
findstr /c:"STOPPED WITH ERROR" "logs\run_aslc_log.txt" >nul 2>&1 && goto :go
echo   %TIME% waiting for ASL Citizen stage >> "%LOG%"
timeout /t 300 /nobreak >nul
goto :wait

:go
if not exist "%USERPROFILE%\.kaggle\access_token" goto :pkgs
set /p KAGGLE_API_TOKEN=<"%USERPROFILE%\.kaggle\access_token"
:pkgs
echo [1/8] installing app, language and voice packages >> "%LOG%"
"%PY%" -m pip install transformers peft datasets sacrebleu accelerate sentencepiece fastapi "uvicorn[standard]" soundfile >> "%LOG%" 2>&1
"%PY%" -m pip install chatterbox-tts >> "%LOG%" 2>&1 || echo   chatterbox install failed, will try kokoro >> "%LOG%"
"%PY%" -m pip install kokoro >> "%LOG%" 2>&1 || echo   kokoro install failed >> "%LOG%"
"%PY%" -c "import torch; print('torch still CUDA:', torch.__version__, torch.cuda.is_available())" >> "%LOG%" 2>&1

echo [2/8] deployment sign model: all WLASL300 data plus ASL Citizen >> "%LOG%"
set "EXTRA="
if exist "data\index_aslc_300.csv" set "EXTRA=--extra data\index_aslc_300.csv"
if not exist "reports\deploy300.json" "%PY%" -u scripts\train_wlasl.py --subset 300 --split all --seed 0 --name deploy300 %EXTRA% >> "%LOG%" 2>&1
if not exist deploy mkdir deploy
copy /y "reports\deploy300.onnx" "deploy\sign.onnx" >nul
copy /y "reports\deploy300_meta.json" "deploy\sign_meta.json" >nul
copy /y "models\hand_landmarker.task" deploy\ >nul
copy /y "models\pose_landmarker_lite.task" deploy\ >nul

echo [3/8] emotion model on RAF-DB real faces >> "%LOG%"
if not exist "data\rafdb\.done" (
  "%CD%\env\Scripts\kaggle.exe" datasets download -d shuvoalok/raf-db-dataset -p data\rafdb --unzip >> "%LOG%" 2>&1 && echo ok > "data\rafdb\.done"
)
if exist "data\rafdb\.done" ("%PY%" -u scripts\emotion_real.py --root data\rafdb >> "%LOG%" 2>&1) else (echo   RAF-DB download failed, emotion stays Neutral in the app >> "%LOG%")
if exist "models\face_landmarker.task" copy /y "models\face_landmarker.task" deploy\ >nul
if not exist "models\face_landmarker.task" "%PY%" -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task','deploy/face_landmarker.task')" >> "%LOG%" 2>&1

echo [4/8] gloss-to-English data and Flan-T5-base LoRA fine-tune >> "%LOG%"
"%PY%" -c "import csv; g=sorted({r['gloss'] for r in csv.DictReader(open('data/index_300.csv'))}); open('data/vocab300.txt','w').write(chr(10).join(g))"
"%PY%" -u scripts\make_gloss_data.py --n 20000 --vocab data\vocab300.txt >> "%LOG%" 2>&1
if not exist "reports\t5.json" "%PY%" -u scripts\t5_lora.py --model google/flan-t5-base --train data\gloss_en_train.jsonl --eval data\gloss_en_test.jsonl --out reports\t5.json >> "%LOG%" 2>&1

echo [5/8] voice benchmark (Chatterbox vs Kokoro) >> "%LOG%"
if not exist "reports\measured_latency.json" "%PY%" -u scripts\tts_bench.py --engines chatterbox,kokoro --n 21 --out reports\measured_latency.json >> "%LOG%" 2>&1

echo [6/8] continuous signing on real stitched clips >> "%LOG%"
if not exist "reports\continuous_real_100.json" "%PY%" -u scripts\continuous_real.py --subset 100 >> "%LOG%" 2>&1

echo [7/8] server self-test >> "%LOG%"
"%PY%" -c "import sys; sys.path.insert(0,'server'); sys.path.insert(0,'.'); import json; from fastapi.testclient import TestClient; import app as A; c=TestClient(A.app); print('index', c.get('/').status_code, 'meta', c.get('/models/sign_meta.json').status_code); ws=c.websocket_connect('/ws').__enter__(); ws.send_text(json.dumps({'type':'speak','glosses':['YESTERDAY','BUY','APPLE'],'emotion':'Happy'})); print(ws.receive_json()); r=ws.receive_json(); print(r['type'], r.get('server_ms'))" >> "%LOG%" 2>&1

echo [8/8] summary >> "%LOG%"
"%PY%" scripts\summarize.py > reports\summary.txt 2>&1
type reports\summary.txt >> "%LOG%"
echo ==== ALL DONE %DATE% %TIME% ==== >> "%LOG%"
echo Finished. Start the app with start_app.bat
pause
