@echo off
title Kronos - Regression Test
cd /d "%~dp0"
echo ===================================================
echo Activating base environment and running regression tests...
echo ===================================================
call E:\LanuageEnvironment\Anaconda3\Scripts\activate.bat E:\LanuageEnvironment\Anaconda3
echo Checking and ensuring pytest is installed...
pip install pytest -i https://pypi.tuna.tsinghua.edu.cn/simple
set PYTHONPATH=.
set HF_ENDPOINT=https://hf-mirror.com
pytest tests/test_kronos_regression.py
pause
