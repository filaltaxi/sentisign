@echo off
setlocal
cd /d "%~dp0"
set "LOG=%CD%\logs\run_aslc_log.txt"
set "PY=%CD%\env\python.exe"
set "ZIP=%USERPROFILE%\Downloads\ASL_Citizen.zip"
echo ==== ASL Citizen stage started %DATE% %TIME% ==== >> "%LOG%"
echo SentiSign v2 ASL Citizen stage. Waiting for %ZIP% to finish downloading. Progress: logs\run_aslc_log.txt . Do not close this window.

rem ---- 1. wait until the browser has finished the download and the zip opens ----
:wait
if exist "%ZIP%" if not exist "%ZIP%.crdownload" if not exist "%ZIP%.part" (
  "%PY%" -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]); print('zip ok')" "%ZIP%" >> "%LOG%" 2>&1 && goto :ready
)
echo   %TIME% waiting for ASL_Citizen.zip download >> "%LOG%"
timeout /t 300 /nobreak >nul
goto :wait

:ready
echo [1/3] zip complete, extracting landmarks for clips overlapping WLASL300 >> "%LOG%"
"%PY%" -u scripts\aslc_extract.py --zip "%ZIP%" --subset 300 >> "%LOG%" 2>&1
if not exist "data\index_aslc_300.csv" (echo ERROR: ASL Citizen extraction failed >> "%LOG%" & goto :fail)

echo [2/3] training WLASL + ASL Citizen, same WLASL test sets as before >> "%LOG%"
for %%R in ("300 signer 0 wlasl300_aslc_signer_s0" "300 signer 1 wlasl300_aslc_signer_s1" "300 signer 2 wlasl300_aslc_signer_s2" "300 official 0 wlasl300_aslc_official_s0" "100 signer 0 wlasl100_aslc_signer_s0" "100 signer 1 wlasl100_aslc_signer_s1" "100 signer 2 wlasl100_aslc_signer_s2" "100 official 0 wlasl100_aslc_official_s0") do call :train %%~R
goto :summary

:train
if exist "reports\%4.json" (echo   skip %4 >> "%LOG%" & goto :eof)
echo   run %4 >> "%LOG%"
"%PY%" -u scripts\train_wlasl.py --subset %1 --split %2 --seed %3 --name %4 --extra data\index_aslc_300.csv >> "%LOG%" 2>&1
goto :eof

:summary
echo [3/3] summary >> "%LOG%"
"%PY%" scripts\summarize.py > reports\summary.txt 2>&1
type reports\summary.txt >> "%LOG%"
echo ==== ALL DONE %DATE% %TIME% ==== >> "%LOG%"
echo Finished. You can close this window.
goto :end

:fail
echo ==== STOPPED WITH ERROR %DATE% %TIME% ==== >> "%LOG%"
echo Stopped with an error. See logs\run_aslc_log.txt

:end
pause
