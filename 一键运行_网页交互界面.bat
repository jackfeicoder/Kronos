@echo off
title Kronos - Web UI Prediction Service
cd /d "%~dp0"
echo ===================================================
echo Activating base environment and starting Web UI...
echo Once started, please visit http://127.0.0.1:7070 in your browser.
echo ===================================================
call E:\LanuageEnvironment\Anaconda3\Scripts\activate.bat E:\LanuageEnvironment\Anaconda3
set PYTHONPATH=.
set HF_ENDPOINT=https://hf-mirror.com
python webui/app.py
pause
