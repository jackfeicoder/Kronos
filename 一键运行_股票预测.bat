@echo off
title Kronos - Stock Prediction Example
cd /d "%~dp0"
echo ===================================================
echo Activating base environment and running stock prediction...
echo ===================================================
call E:\LanuageEnvironment\Anaconda3\Scripts\activate.bat E:\LanuageEnvironment\Anaconda3
set PYTHONPATH=.
set HF_ENDPOINT=https://hf-mirror.com
python examples/prediction_akshare_2024-2025.py
pause
