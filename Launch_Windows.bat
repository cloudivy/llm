@echo off
setlocal enabledelayedexpansion
title Enterprise Offline Knowledge Assistant
cd /d "%~dp0"

echo === Enterprise Offline Knowledge Assistant ===
echo.

REM --- Check if Ollama is running, start it if not ---
curl -s http://127.0.0.1:11434 >nul 2>&1
if errorlevel 1 (
    echo Starting Ollama...
    start /min "" ollama serve
    timeout /t 5 /nobreak >nul
) else (
    echo Ollama already running.
)

REM --- Ensure models are pulled ---
ollama list | findstr /C:"llama3.1:8b" >nul
if errorlevel 1 (
    echo Pulling llama3.1:8b (one-time, ~4.9GB)...
    ollama pull llama3.1:8b
)

ollama list | findstr /C:"nomic-embed-text" >nul
if errorlevel 1 (
    echo Pulling nomic-embed-text (one-time)...
    ollama pull nomic-embed-text
)

REM --- Ensure Python packages are installed ---
echo Checking Python packages...
pip install --quiet -r requirements.txt

REM --- Knowledge base building now happens inside the app itself, per
REM selected knowledge base under KnowledgeBases\<name>\ — nothing to
REM pre-check here anymore.

REM --- Launch the app ---
echo.
echo Launching app in your browser...
streamlit run app.py

echo.
pause
