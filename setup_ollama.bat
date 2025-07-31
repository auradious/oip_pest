@echo off
echo ========================================
echo   Ollama Setup for Pest Management AI
echo ========================================
echo.

REM Check if Ollama is already installed
where ollama >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo ✅ Ollama is already installed!
    goto :check_model
)

echo 📥 Downloading and installing Ollama...
echo.

REM Download Ollama installer
echo Downloading Ollama installer...
powershell -Command "Invoke-WebRequest -Uri 'https://ollama.com/download/windows' -OutFile 'ollama-installer.exe'"

if not exist "ollama-installer.exe" (
    echo ❌ Failed to download Ollama installer
    echo Please download manually from: https://ollama.com/download/windows
    pause
    exit /b 1
)

echo 🚀 Installing Ollama...
start /wait ollama-installer.exe

REM Clean up installer
del ollama-installer.exe

echo.
echo ⏳ Waiting for Ollama service to start...
timeout /t 10 /nobreak >nul

:check_model
echo.
echo 🤖 Checking for required AI model (gemma3:latest)...

REM Check if model exists
ollama list | findstr "gemma3" >nul
if %ERRORLEVEL% EQU 0 (
    echo ✅ gemma3:latest model is already available!
    goto :success
)

echo 📦 Pulling gemma3:latest model...
echo This may take several minutes depending on your internet connection...
echo.

ollama pull gemma3:latest

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to pull gemma3:latest model
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

:success
echo.
echo ========================================
echo ✅ Ollama Setup Complete!
echo ========================================
echo.
echo 🎉 Your system is now ready for AI-powered pest management!
echo.
echo Next steps:
echo 1. Run: py app.py
echo 2. Open: http://localhost:7860
echo 3. Upload pest images for identification
echo.
echo 📋 Installed components:
echo   • Ollama service
echo   • gemma3:latest AI model
echo.
echo 🔧 Troubleshooting:
echo   • If Ollama service doesn't start, restart your computer
echo   • Check Windows Defender/Antivirus settings if installation fails
echo   • Ensure you have administrator privileges
echo.
pause