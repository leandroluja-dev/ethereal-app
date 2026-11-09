@echo off
REM Roda os testes automatizados.
cd /d "%~dp0"
.venv\Scripts\python -m pytest -q
pause
