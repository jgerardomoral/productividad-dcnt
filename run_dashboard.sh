#!/bin/bash
echo "================================================================================"
echo "Dashboard DCNT - Productividad Científica"
echo "================================================================================"
echo ""
echo "Verificando instalación de dependencias..."
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "Instalando dependencias..."
    pip3 install -r requirements.txt
fi
echo ""
echo "Iniciando dashboard..."
streamlit run src/app.py
