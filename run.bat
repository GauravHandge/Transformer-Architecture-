@echo off
echo ============================================
echo   Transformer Explainer - Python Backend
echo ============================================
echo.
echo [1] Installing dependencies...
pip install -r requirements.txt

echo.
echo [2] Starting FastAPI server...
echo    API:      http://127.0.0.1:8000
echo    Frontend: http://127.0.0.1:8000
echo.
echo    First run will download GPT-2 (~500MB)
echo    Press Ctrl+C to stop
echo.
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
