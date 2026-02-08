@echo off
echo ==========================================
echo Setting up GPU Environment for Disease Detector
echo ==========================================

REM Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.10 or 3.11 first.
    pause
    exit /b
)

REM Create Virtual Environment
if exist venv (
    echo Virtual environment 'venv' already exists. Skipping creation.
) else (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate and Install
echo Activating environment and installing dependencies...
call venv\Scripts\activate

echo Installing PyTorch with CUDA 12.4 support (for RTX 3050)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo Installing other requirements...
pip install fastapi uvicorn python-multipart pillow jinja2 numpy

echo.
echo ==========================================
echo Setup Complete!
echo.
echo To start training, run:
echo    venv\Scripts\activate
echo    python train.py
echo ==========================================
pause
