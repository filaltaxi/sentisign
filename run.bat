@echo off
setlocal
cd /d "%~dp0"
if not exist logs mkdir logs
set "LOG=%CD%\logs\run_log.txt"
echo ==== SentiSign v2 run started %DATE% %TIME% ==== >> "%LOG%"
echo SentiSign v2 is running. Progress is written to logs\run_log.txt . Do not close this window.

rem ---- 1. Python 3.11 environment (conda) ----
rem conda.exe breaks when the user folder has spaces, so call conda through python -m conda or conda.bat
set "CBASE=%USERPROFILE%\miniconda3"
if not exist "%CBASE%\python.exe" set "CBASE=%USERPROFILE%\anaconda3"
if not exist "%CBASE%\python.exe" (echo ERROR: miniconda3 or anaconda3 not found in %USERPROFILE% >> "%LOG%" & goto :fail)
if not exist "%CD%\env\python.exe" (
  echo [1/6] creating Python 3.11 env >> "%LOG%"
  "%CBASE%\python.exe" -m conda create -y -p "%CD%\env" python=3.11 >> "%LOG%" 2>&1
)
if not exist "%CD%\env\python.exe" (
  echo   retrying with conda.bat >> "%LOG%"
  call "%CBASE%\condabin\conda.bat" create -y -p "%CD%\env" python=3.11 >> "%LOG%" 2>&1
)
set "PY=%CD%\env\python.exe"
if not exist "%PY%" (echo ERROR: env creation failed >> "%LOG%" & goto :fail)

rem ---- 2. packages ----
if not exist "%CD%\env\.pkgs_ok" (
  echo [2/6] installing packages >> "%LOG%"
  "%PY%" -m pip install --upgrade pip >> "%LOG%" 2>&1
  "%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu124 >> "%LOG%" 2>&1
  "%PY%" -m pip install --upgrade mediapipe opencv-python kaggle six scikit-learn onnx onnxruntime >> "%LOG%" 2>&1
  "%PY%" -c "import torch, mediapipe, cv2, sklearn, onnx" >> "%LOG%" 2>&1 && echo ok > "%CD%\env\.pkgs_ok"
)
if not exist "%CD%\env\.pkgs_ok" (echo ERROR: package install failed >> "%LOG%" & goto :fail)
"%PY%" -c "import torch; print('GPU check:', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')" >> "%LOG%" 2>&1

rem ---- 3. WLASL download ----
if not exist "data\wlasl\.done" (
  if exist "%USERPROFILE%\.kaggle\access_token" set /p KAGGLE_API_TOKEN=<"%USERPROFILE%\.kaggle\access_token"
  if not exist "%USERPROFILE%\.kaggle\access_token" if not exist "%USERPROFILE%\.kaggle\kaggle.json" (echo ERROR: no Kaggle credentials in %USERPROFILE%\.kaggle >> "%LOG%" & goto :fail)
  echo [3/6] downloading WLASL from Kaggle, about 5 GB >> "%LOG%"
  "%PY%" -m pip install six >> "%LOG%" 2>&1
  "%CD%\env\Scripts\kaggle.exe" datasets download -d risangbaskoro/wlasl-processed -p data\wlasl --unzip >> "%LOG%" 2>&1
  if not exist "data\wlasl\WLASL_v0.3.json" "%PY%" -c "from kaggle.cli import main; import sys; sys.argv=['kaggle','datasets','download','-d','risangbaskoro/wlasl-processed','-p','data/wlasl','--unzip']; main()" >> "%LOG%" 2>&1
  if exist "data\wlasl\WLASL_v0.3.json" (echo ok > "data\wlasl\.done") else (
    for /r "data\wlasl" %%F in (WLASL_v0.3.json) do if exist "%%F" copy "%%F" "data\wlasl\WLASL_v0.3.json" >nul
    if exist "data\wlasl\WLASL_v0.3.json" (echo ok > "data\wlasl\.done")
  )
)
if not exist "data\wlasl\.done" (echo ERROR: WLASL download failed or WLASL_v0.3.json missing >> "%LOG%" & goto :fail)

rem ---- 4. landmark extraction (resumable) ----
echo [4/6] extracting landmarks for WLASL100 >> "%LOG%"
"%PY%" -u scripts\extract_wlasl.py --root data\wlasl --subset 100 >> "%LOG%" 2>&1
if not exist "data\index_100.csv" (echo ERROR: extraction failed >> "%LOG%" & goto :fail)

rem ---- 5. training runs on GPU ----
echo [5/6] training >> "%LOG%"
for %%R in ("official 0 wlasl100_official_s0 ." "signer 0 wlasl100_signer_s0 ." "signer 1 wlasl100_signer_s1 ." "signer 2 wlasl100_signer_s2 ." "random 0 wlasl100_random_s0 ." "signer 0 wlasl100_signer_noloc_s0 loc+move") do call :train %%~R
echo [5b/6] WLASL300: extracting extra videos and training >> "%LOG%"
"%PY%" -u scripts\extract_wlasl.py --root data\wlasl --subset 300 >> "%LOG%" 2>&1
if not exist "data\index_300.csv" (echo ERROR: WLASL300 extraction failed >> "%LOG%" & goto :summary)
for %%R in ("official 0 wlasl300_official_s0 ." "signer 0 wlasl300_signer_s0 ." "signer 1 wlasl300_signer_s1 ." "signer 2 wlasl300_signer_s2 ." "signer 0 wlasl300_signer_noloc_s0 loc+move") do call :train300 %%~R
goto :summary

:train300
if exist "reports\%3.json" (echo   skip %3, already done >> "%LOG%" & goto :eof)
echo   run %3 >> "%LOG%"
if "%4"=="." ("%PY%" -u scripts\train_wlasl.py --subset 300 --split %1 --seed %2 --name %3 >> "%LOG%" 2>&1) else ("%PY%" -u scripts\train_wlasl.py --subset 300 --split %1 --seed %2 --name %3 --ablate %4 >> "%LOG%" 2>&1)
goto :eof

:train
if exist "reports\%3.json" (echo   skip %3, already done >> "%LOG%" & goto :eof)
echo   run %3 >> "%LOG%"
if "%4"=="." ("%PY%" -u scripts\train_wlasl.py --subset 100 --split %1 --seed %2 --name %3 >> "%LOG%" 2>&1) else ("%PY%" -u scripts\train_wlasl.py --subset 100 --split %1 --seed %2 --name %3 --ablate %4 >> "%LOG%" 2>&1)
goto :eof

:summary
echo [6/6] summary >> "%LOG%"
"%PY%" scripts\summarize.py > reports\summary.txt 2>&1
type reports\summary.txt >> "%LOG%"
echo ==== ALL DONE %DATE% %TIME% ==== >> "%LOG%"
echo Finished. You can close this window.
goto :end

:fail
echo ==== STOPPED WITH ERROR %DATE% %TIME% ==== >> "%LOG%"
echo Stopped with an error. See logs\run_log.txt

:end
pause
