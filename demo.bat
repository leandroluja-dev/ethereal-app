@echo off
REM Abre o CandleSense com um mercado sintetico (nao precisa da IQ Option).
cd /d "%~dp0"
.venv\Scripts\python -m candlesense demo %*
if errorlevel 1 pause
