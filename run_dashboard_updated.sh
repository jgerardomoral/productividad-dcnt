#!/bin/bash
echo "🚀 Iniciando Dashboard de Productividad DCNT..."
echo ""

# Verificar si streamlit está instalado
if ! command -v streamlit &> /dev/null
then
    echo "⚠️  Streamlit no está instalado. Instalando dependencias..."
    pip install streamlit plotly networkx openpyxl pandas
    echo ""
fi

echo "📊 Abriendo dashboard en http://localhost:8501"
echo ""
echo "Presiona Ctrl+C para detener el dashboard"
echo ""

# Ejecutar streamlit
streamlit run src/app.py --server.port 8501 --server.address localhost
