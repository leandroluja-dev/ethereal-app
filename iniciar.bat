@echo off
REM Abre o CandleSense lendo a IQ Option ao vivo.
cd /d "%~dp0"
.venv\Scripts\python -m candlesense %*
if errorlevel 1 pause
