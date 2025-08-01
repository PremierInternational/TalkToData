@echo off
chcp 65001 >nul
title Premier NLP + Ollama Installer

echo ===========================================
echo Premier NLP + Ollama App Installer
echo ===========================================

echo -------------------------------
echo Creating virtual environment...
echo -------------------------------
python -m venv venv

echo -------------------------------
echo Activating virtual environment...
echo -------------------------------
call venv\Scripts\activate

echo -------------------------------
echo Installing Python dependencies...
echo -------------------------------
pip install -r requirements.txt

echo -------------------------------
echo Checking Ollama setup...
echo -------------------------------

:: Try to get Ollama version (redirect error to temp file)
ollama --version >nul 2>ollama_check.txt
findstr /C:"not recognized" ollama_check.txt >nul
if %errorlevel%==0 (
    echo 🧠 Ollama not found. Installing from ollamasetup.exe...
    if exist "ollamasetup.exe" (
        start /wait ollamasetup.exe
        echo ✅ Ollama installed successfully.
    ) else (
        echo ❌ ollamasetup.exe not found in folder.
        del ollama_check.txt
        exit /b 1
    )
) else (
    echo Ollama found. Version:
    ollama --version
)

del ollama_check.txt

echo -------------------------------
echo Pulling model manifest (tinyllama)...
echo -------------------------------
ollama pull tinyllama

echo -------------------------------
echo Running Streamlit App...
echo -------------------------------
streamlit run app.py

endlocal
pause
