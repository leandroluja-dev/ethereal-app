@echo off
REM Instala o CandleSense num ambiente virtual (.venv). Precisa do Python 3.11+ instalado.
cd /d "%~dp0"
where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
if errorlevel 1 (
  echo Nao encontrei o Python. Instale o Python 3.11 em https://www.python.org/downloads/ marcando "Add python.exe to PATH".
  pause
  exit /b 1
)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements-dev.txt
if errorlevel 1 (
  echo A instalacao falhou. Veja as mensagens acima.
  pause
  exit /b 1
)
echo.
echo Instalado. Use iniciar.bat (ao vivo) ou demo.bat (demonstracao).
pause
