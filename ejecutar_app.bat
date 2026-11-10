@echo off
rem Doble clic para abrir el aplicativo en el navegador (http://localhost:8501).
rem Para cerrarlo, cierre esta ventana.
cd /d "%~dp0"
set PY=python
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
%PY% -m pip install -q -r requirements.txt
%PY% -m streamlit run app.py
pause
