@echo off
title VieNeu Synthesis Test
cd /d "%~dp0"

echo ========================================================
echo       VIENEU REAL NEURAL SYNTHESIS - TEST RUN
echo ========================================================
echo.

if not exist "test_outputs" mkdir "test_outputs"

echo Running VieNeu Worker with isolated Python 3.11 (.venv_vieneu_new)...
echo Text: "Xin chao, day la bai kiem tra giong doc tieng Viet thuc te."
echo Voice: Truc Ly
echo.

.venv_vieneu_new\Scripts\python.exe app\services\tts\vieneu_worker.py --text "Xin chào, đây là bài kiểm tra giọng đọc tiếng Việt thực tế." --output "test_outputs\test_real_synthesis.wav" --voice "Trúc Ly"

set EXIT_CODE=%errorlevel%
echo.
echo ========================================================
echo Worker exit code: %EXIT_CODE%
echo ========================================================
echo.
pause
