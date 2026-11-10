@echo off
rem Doble clic para abrir el aplicativo en el navegador (http://localhost:8501).
rem Para cerrarlo, cierre esta ventana.
cd /d "%~dp0"
set PY=python
where py >nul 2>nul && set PY=py
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
echo Instalando librerias (solo tarda la primera vez)...
%PY% -m pip install --user -q -r requirements.txt
echo Abriendo el aplicativo en http://localhost:8501 ...
%PY% -m streamlit run app.py
pause
