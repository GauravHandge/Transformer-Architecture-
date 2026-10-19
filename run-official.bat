@echo off
echo ============================================================
echo   Official Transformer Explainer (Polo Club Georgia Tech)
echo ============================================================
echo.
echo Starting local server at: http://127.0.0.1:5173
echo.
cd /d "%~dp0official"
npx vite dev --host 127.0.0.1 --port 5173
