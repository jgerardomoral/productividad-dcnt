@echo off
echo ================================================================================
echo Dashboard DCNT - Productividad Cientifica
echo ================================================================================
echo.
echo Verificando instalacion de dependencias...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Instalando dependencias...
    pip install -r requirements.txt
)
echo.
echo Iniciando dashboard...
streamlit run src/app.py
