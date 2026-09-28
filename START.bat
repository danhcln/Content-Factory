@echo off
title AI CONTENT FACTORY - Local Server
cd /d "%~dp0"

echo ===================================================
echo           AI CONTENT FACTORY - LOCAL SYSTEM
echo ===================================================
echo.

:: Detect if port 8000 is already in use
netstat -ano | findstr LISTENING | findstr :8000 >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] AI Content Factory server is already running on http://localhost:8000!
    echo Opening browser window directly...
    start http://localhost:8000
    exit /b 0
)

:: Locate Python executable (.venv_vieneu_new preferred for VieNeu, .venv for standard app)
set PYTHON_EXE=
if exist ".venv_vieneu_new\Scripts\python.exe" (
    set "PYTHON_EXE=.venv_vieneu_new\Scripts\python.exe"
) else if exist ".venv_vieneu\Scripts\python.exe" (
    set "PYTHON_EXE=.venv_vieneu\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    where python >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON_EXE=python"
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python environment not found!
    echo Please make sure .venv or .venv_vieneu_new exists or Python is installed in PATH.
    echo.
    pause
    exit /b 1
)

echo [OK] Using Python environment: %PYTHON_EXE%
echo Initializing local environment and folders...

:: Ensure folders exist
if not exist "data" mkdir "data"
if not exist "downloads\original" mkdir "downloads\original"
if not exist "downloads\packages" mkdir "downloads\packages"
if not exist "downloads\final" mkdir "downloads\final"
if not exist "temp" mkdir "temp"
if not exist "logs" mkdir "logs"

echo [OK] Folders initialized.
echo Starting Web Server at http://localhost:8000 ...
echo.

:: Open browser after 2 seconds in background
start "" cmd /c "timeout /t 2 >nul & start http://localhost:8000"

:: Start Uvicorn
"%PYTHON_EXE%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Server terminated with an error code: %errorlevel%
    pause
)
