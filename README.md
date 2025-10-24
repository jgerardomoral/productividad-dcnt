# Dashboard de Productividad Científica DCNT

Dashboard interactivo para visualizar y analizar la productividad científica del **Doctorado en Ciencias de la Nutrición Traslacional** (Universidad de Guadalajara, 2019-2025).

![Dashboard Preview](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)

## ✨ Características

- **226 publicaciones científicas** analizadas (2019-2025)
- **Clasificación por ODS**: Contribución a Objetivos de Desarrollo Sostenible
- **Clasificación por PRONACES**: Alineación con Programas Nacionales Estratégicos
- **Análisis temático**: 15+ temas de investigación identificados
- **Visualizaciones interactivas**:
  - Evolución temporal de publicaciones
  - Distribución por ODS y PRONACES
  - Treemap, Sunburst y matrices de co-ocurrencia de temas
  - Gráficas comparativas y métricas clave

## 🚀 Inicio Rápido

### Prerrequisitos

- Python 3.8 o superior
- pip

### Instalación

1. **Clonar el repositorio**

```bash
git clone https://github.com/tu-usuario/dashboard_DCNT_public.git
cd dashboard_DCNT_public
```

2. **Crear entorno virtual (recomendado)**

```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Instalar dependencias**

```bash
pip install -r requirements.txt
```

### Ejecutar el Dashboard

**Opción 1: Script de inicio**

```bash
# En Windows
run_dashboard.bat

# En Linux/Mac
bash run_dashboard.sh
```

**Opción 2: Comando directo**

```bash
streamlit run src/app.py
```

El dashboard se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📊 Secciones del Dashboard

### 1. 🎯 Panorama General
- Total de publicaciones por año
- Métricas clave (total publicaciones, revistas únicas)
- Gráfica de evolución temporal

### 2. 🌍 Contribución a ODS
- Distribución por Objetivos de Desarrollo Sostenible
- ODS principales: ODS 3 (Salud), ODS 2 (Hambre Cero), ODS 10 (Reducción de Desigualdades)
- Análisis de impacto

### 3. 🎯 Alineación PRONACES
- Distribución por Programas Nacionales Estratégicos
- Programas principales: Salud, Enfermedades Crónicas, Sistemas Alimentarios
- Matriz de calor PRONACES vs Años

### 4. 🔬 Análisis Temático
- **Distribución por tema**: Top 15 temas de investigación
- **Visualizaciones**:
  - Treemap jerárquico
  - Sunburst por categorías
  - Matriz de co-ocurrencia de temas
  - Top 5 conexiones entre temas

### 5. 📑 Publicaciones Detalladas
- Tabla completa de publicaciones
- Información: Título, Autores, Revista, Año, DOI

## 🏗️ Estructura del Proyecto

```
dashboard_DCNT_public/
├── data/
│   ├── publications_base.csv          # Base de datos de publicaciones
│   ├── ods_classification.json        # Clasificación ODS
│   ├── pronaces_classification.json   # Clasificación PRONACES
│   └── themes_classification.json     # Clasificación temática
├── src/
│   ├── app.py                         # Dashboard Streamlit
│   └── config_context.py              # Datos contextuales (ODS, PRONACES, epidemiología)
├── LOGO DCNT_1.png                    # Logo institucional
├── requirements.txt                    # Dependencias
├── run_dashboard.sh                   # Script de inicio (Linux/Mac)
├── run_dashboard.bat                  # Script de inicio (Windows)
└── README.md
```

## 📈 Datos

El dashboard incluye datos pre-procesados de 226 publicaciones científicas:

- **Periodo**: 2019-2025
- **Fuentes**: Publicaciones en revistas indexadas
- **Clasificaciones**: ODS, PRONACES, Temas de investigación

Los datos están almacenados en formato CSV y JSON, listos para visualización sin necesidad de extracción o procesamiento adicional.

## 🎨 Tecnologías

- **[Streamlit](https://streamlit.io/)**: Framework para el dashboard interactivo
- **[Plotly](https://plotly.com/python/)**: Visualizaciones interactivas
- **[NetworkX](https://networkx.org/)**: Análisis de redes (co-ocurrencia de temas)
- **Python**: Lenguaje de programación

## 🤝 Contribuciones

Este es un proyecto académico desarrollado para el Doctorado en Ciencias de la Nutrición Traslacional. Las contribuciones son bienvenidas:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

## 👥 Contacto

**Doctorado en Ciencias de la Nutrición Traslacional**
- Universidad de Guadalajara
- Centro Universitario de Ciencias de la Salud (CUCS)

## 🙏 Agradecimientos

Dashboard desarrollado para apoyar la evaluación del programa de doctorado ante SECIHTI.

**Desarrollado por:** José Gerardo Mora Almanza - Alumno del DCNT

---

**Desarrollado con ❤️ para el Doctorado en Ciencias de la Nutrición Traslacional**
