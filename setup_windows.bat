@echo off
REM ============================================================
REM setup_windows.bat — One-shot Windows setup for
REM VibeVoice Meeting Transcription App
REM
REM Prerequisites:
REM   - Python 3.10+ installed and on PATH
REM   - Git installed and on PATH
REM   - NVIDIA GPU driver installed (CUDA 12.x)
REM   - ffmpeg installed and on PATH
REM     (Download from https://ffmpeg.org/download.html and
REM      add the bin/ folder to your PATH)
REM
REM Run this script once from the repo root:
REM   setup_windows.bat
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo  VibeVoice Meeting Transcription App - Windows Setup
echo ============================================================
echo.

REM --- Check Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo         Install Python 3.10+ from https://python.org
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do echo [OK] Found %%i

REM --- Check Git ---
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found in PATH.
    echo         Install Git from https://git-scm.com
    exit /b 1
)
echo [OK] Git is available.

REM --- Check ffmpeg ---
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] ffmpeg not found in PATH.
    echo           Download from https://ffmpeg.org/download.html
    echo           Extract and add the bin\ folder to your PATH.
    echo           ffmpeg is required for audio splitting (files ^>25 min).
    echo.
) else (
    echo [OK] ffmpeg is available.
)

REM --- Check ffprobe (comes with ffmpeg) ---
ffprobe -version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] ffprobe not found. Install ffmpeg as above.
) else (
    echo [OK] ffprobe is available.
)

echo.
echo --- Creating virtual environment ---
if exist venv (
    echo [INFO] venv\ already exists, skipping creation.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
    echo [OK] Created venv\
)

echo.
echo --- Activating virtual environment ---
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    exit /b 1
)
echo [OK] Virtual environment activated.

echo.
echo --- Upgrading pip ---
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [ERROR] pip upgrade failed.
    exit /b 1
)

echo.
echo --- Installing PyTorch with CUDA 12.4 wheels ---
echo     (This downloads ~2.5 GB — please wait...)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed.
    echo         Ensure you have a working internet connection.
    exit /b 1
)
echo [OK] PyTorch with CUDA installed.

echo.
echo --- Verifying CUDA availability ---
python -c "import torch; print('[OK] CUDA available:', torch.cuda.is_available()); print('     Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo --- Installing VibeVoice package from GitHub ---
echo     (Cloning and installing Microsoft/VibeVoice...)
pip install git+https://github.com/microsoft/VibeVoice.git
if errorlevel 1 (
    echo [ERROR] VibeVoice installation failed.
    echo         Check your internet connection and Git configuration.
    exit /b 1
)
echo [OK] VibeVoice installed.

echo.
echo --- Installing remaining Python dependencies ---
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] requirements.txt installation failed.
    exit /b 1
)
echo [OK] Dependencies installed.

echo.
echo --- Attempting to install flash-attn (optional, may fail on Windows) ---
echo     If this fails, the app will use SDPA attention instead (recommended on Windows).
pip install flash-attn --no-build-isolation 2>nul
if errorlevel 1 (
    echo [INFO] flash-attn installation skipped (not available for your platform).
    echo        The app defaults to --attn-impl sdpa which works great on Windows.
) else (
    echo [OK] flash-attn installed (use --attn-impl flash_attention_2 for best performance).
)

echo.
echo --- Creating audio and output directories ---
if not exist audio mkdir audio
if not exist output mkdir output
echo [OK] Created audio\ and output\ directories.

echo.
echo ============================================================
echo  Setup complete!
echo ============================================================
echo.
echo  To start transcribing:
echo.
echo    1. Activate the venv (if not already):
echo       venv\Scripts\activate
echo.
echo    2. Place audio files in the audio\ folder.
echo.
echo    3. Transcribe a file:
echo       python transcribe.py --input audio\your_meeting.mp3
echo.
echo    4. Batch transcribe a folder:
echo       python transcribe.py --batch audio\ --output-dir output\
echo.
echo    5. Check VRAM usage:
echo       python transcribe.py --vram-check
echo.
echo    Note: First run downloads the ~7 GB model from HuggingFace.
echo          Subsequent runs use the local cache (~/.cache/huggingface/).
echo.

endlocal
