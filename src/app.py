"""
Dashboard de Productividad del Doctorado en Ciencias de la Nutrición Traslacional
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import networkx as nx
from collections import Counter
from config_context import (
    EPIDEMIOLOGIA_MEXICO,
    EPIDEMIOLOGIA_JALISCO,
    ODS_CONTEXTO,
    PRONACES_CONTEXTO,
    LINEAS_INVESTIGACION,
    REGION_OCCIDENTE,
    INFRAESTRUCTURA_UDG,
    INVESTIGACION_TRASLACIONAL
)

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Productividad DCNT",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar el estado del tema
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Función para obtener estilos CSS según el tema
def get_theme_css(theme):
    if theme == 'dark':
        return """
        <style>
            /* Tema Oscuro */
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                color: #4da6ff;
                text-align: center;
                padding: 1rem 0;
            }
            .section-header {
                font-size: 1.8rem;
                font-weight: bold;
                color: #e0e0e0;
                border-bottom: 3px solid #4da6ff;
                padding-bottom: 0.5rem;
                margin-top: 2rem;
            }
            .metric-card {
                background-color: #2b2b2b;
                padding: 1.5rem;
                border-radius: 10px;
                border-left: 5px solid #4da6ff;
            }
            .stMetric {
                background-color: #1e1e1e;
                padding: 1rem;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            /* Ajustar colores de texto en modo oscuro */
            .stMarkdown, p, h1, h2, h3, h4, h5, h6 {
                color: #e0e0e0 !important;
            }
            div[data-testid="stMetricValue"] {
                color: #4da6ff !important;
            }
        </style>
        """
    else:
        return """
        <style>
            /* Tema Claro */
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                color: #1f77b4;
                text-align: center;
                padding: 1rem 0;
            }
            .section-header {
                font-size: 1.8rem;
                font-weight: bold;
                color: #2c3e50;
                border-bottom: 3px solid #1f77b4;
                padding-bottom: 0.5rem;
                margin-top: 2rem;
            }
            .metric-card {
                background-color: #f8f9fa;
                padding: 1.5rem;
                border-radius: 10px;
                border-left: 5px solid #1f77b4;
            }
            .stMetric {
                background-color: #ffffff;
                padding: 1rem;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
        </style>
        """

# Aplicar estilos CSS según el tema seleccionado
st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Carga todos los datos necesarios"""
    base_dir = Path(__file__).parent.parent / "data"

    # Cargar publicaciones
    publications = pd.read_csv(base_dir / "publications_base.csv")

    # Intentar cargar clasificaciones (si existen)
    try:
        with open(base_dir / "ods_classification.json", 'r', encoding='utf-8') as f:
            ods_data = json.load(f)
    except FileNotFoundError:
        ods_data = []

    try:
        with open(base_dir / "pronaces_classification.json", 'r', encoding='utf-8') as f:
            pronaces_data = json.load(f)
    except FileNotFoundError:
        pronaces_data = []

    try:
        with open(base_dir / "themes_classification.json", 'r', encoding='utf-8') as f:
            themes_data = json.load(f)
    except FileNotFoundError:
        themes_data = []

    return publications, ods_data, pronaces_data, themes_data


def create_year_evolution_chart(df):
    """Gráfica de evolución anual de publicaciones"""
    publications_per_year = df.groupby('año').size().reset_index(name='Publicaciones')

    fig = px.bar(
        publications_per_year,
        x='año',
        y='Publicaciones',
        title='Evolución de la Productividad Científica (2019-2025)',
        labels={'año': 'Año', 'Publicaciones': 'Número de Publicaciones'},
        color='Publicaciones',
        color_continuous_scale='Blues'
    )

    fig.update_layout(
        xaxis=dict(tickmode='linear'),
        hovermode='x unified',
        height=400
    )

    return fig


def create_ods_distribution(ods_data):
    """Gráfica de distribución por ODS"""
    if not ods_data:
        return None

    # Contar ODS
    ods_counter = Counter()
    for pub in ods_data:
        for ods in pub.get('ods_principales', []):
            ods_name = f"ODS {ods.get('numero')}: {ods.get('nombre', '')}"
            ods_counter[ods_name] += 1

    if not ods_counter:
        return None

    # Crear DataFrame
    ods_df = pd.DataFrame(ods_counter.most_common(), columns=['ODS', 'Publicaciones'])

    # Gráfica de donut
    fig = px.pie(
        ods_df,
        values='Publicaciones',
        names='ODS',
        title='Distribución de Publicaciones por ODS',
        hole=0.4
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500)

    return fig


def create_pronaces_heatmap(pronaces_data, publications_df):
    """Matriz de calor PRONACES vs Años"""
    if not pronaces_data:
        return None

    # Crear matriz
    years = sorted(publications_df['año'].unique())
    pronaces_list = []
    matrix_data = []

    # Recopilar todos los PRONACES únicos
    all_pronaces = set()
    for pub in pronaces_data:
        for pron in pub.get('pronaces_principales', []):
            all_pronaces.add(pron.get('nombre', ''))

    pronaces_list = sorted(list(all_pronaces))

    # Crear matriz de conteo
    matrix = {year: {pron: 0 for pron in pronaces_list} for year in years}

    for pub in pronaces_data:
        year = pub.get('año')
        for pron in pub.get('pronaces_principales', []):
            pron_name = pron.get('nombre', '')
            if pron_name in pronaces_list and year in matrix:
                matrix[year][pron_name] += 1

    # Convertir a formato para heatmap
    z_data = [[matrix[year][pron] for year in years] for pron in pronaces_list]

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=years,
        y=pronaces_list,
        colorscale='YlOrRd',
        text=z_data,
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Publicaciones")
    ))

    fig.update_layout(
        title='Matriz de Calor: PRONACES vs Años',
        xaxis_title='Año',
        yaxis_title='PRONACES',
        height=500
    )

    return fig


def create_themes_treemap(themes_data):
    """Treemap jerárquico de temas de investigación"""
    if not themes_data:
        return None

    # Contar temas
    theme_counter = Counter()
    for pub in themes_data:
        for theme in pub.get('temas', []):
            theme_name = theme.get('nombre', '')
            if theme_name:
                theme_counter[theme_name] += 1

    if not theme_counter:
        return None

    # Preparar datos para treemap
    themes = list(theme_counter.keys())
    values = list(theme_counter.values())

    # Crear treemap
    fig = go.Figure(go.Treemap(
        labels=themes,
        parents=["Temas de Investigación"] * len(themes),
        values=values,
        textinfo="label+value+percent parent",
        textfont=dict(size=14),
        marker=dict(
            colorscale='Viridis',
            cmid=sum(values)/len(values),
            colorbar=dict(
                title="Publicaciones",
                thickness=20,
                len=0.7
            )
        ),
        hovertemplate='<b>%{label}</b><br>Publicaciones: %{value}<br>Porcentaje: %{percentParent}<extra></extra>'
    ))

    fig.update_layout(
        title='Distribución Jerárquica de Temas (Treemap)',
        height=600,
        margin=dict(l=10, r=10, t=50, b=10)
    )

    return fig


def create_themes_cooccurrence(themes_data):
    """Matriz de co-ocurrencia de temas"""
    if not themes_data:
        return None

    # Obtener todos los temas únicos
    all_themes = set()
    for pub in themes_data:
        for theme in pub.get('temas', []):
            theme_name = theme.get('nombre', '')
            if theme_name:
                all_themes.add(theme_name)

    all_themes = sorted(list(all_themes))

    if len(all_themes) == 0:
        return None

    # Crear matriz de co-ocurrencia
    matrix = [[0 for _ in all_themes] for _ in all_themes]

    for pub in themes_data:
        pub_themes = [t.get('nombre', '') for t in pub.get('temas', []) if t.get('nombre', '')]

        # Contar co-ocurrencias
        for i, theme1 in enumerate(all_themes):
            for j, theme2 in enumerate(all_themes):
                if theme1 in pub_themes and theme2 in pub_themes:
                    if i != j:  # No contar auto-correlación
                        matrix[i][j] += 1

    # Crear heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=all_themes,
        y=all_themes,
        colorscale='YlOrRd',
        text=matrix,
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{y} co-ocurre con %{x}<br>Veces: %{z}<extra></extra>',
        colorbar=dict(
            title="Co-ocurrencias",
            thickness=20,
            len=0.7
        )
    ))

    fig.update_layout(
        title='Matriz de Co-ocurrencia de Temas',
        xaxis_title='Temas',
        yaxis_title='Temas',
        height=700,
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0)
    )

    return fig


def create_themes_sunburst(themes_data):
    """Gráfica sunburst de temas por categoría"""
    if not themes_data:
        return None

    # Categorizar temas
    categorias = {
        'Enfermedades Metabólicas': ['OBESIDAD_SOBREPESO', 'DIABETES', 'LIPIDOS_COLESTEROL', 'INFLAMACION_METABOLICA'],
        'Enfermedades Inmunológicas': ['ENFERMEDADES_AUTOINMUNES', 'COVID19'],
        'Factores de Estilo de Vida': ['ACTIVIDAD_FISICA', 'MICROBIOTA_INTESTINAL'],
        'Genética y Nutrición': ['GENETICA_NUTRICION'],
        'Oncología': ['CANCER']
    }

    # Contar temas
    theme_counter = Counter()
    for pub in themes_data:
        for theme in pub.get('temas', []):
            theme_name = theme.get('nombre', '')
            if theme_name:
                theme_counter[theme_name] += 1

    if not theme_counter:
        return None

    # Preparar datos para sunburst
    labels = ['Todos los Temas']
    parents = ['']
    values = [0]  # Se calculará después

    # Agregar categorías
    for categoria in categorias.keys():
        labels.append(categoria)
        parents.append('Todos los Temas')
        values.append(0)  # Se calculará después

    # Agregar temas individuales
    total = 0
    for categoria, temas in categorias.items():
        cat_total = 0
        for tema in temas:
            if tema in theme_counter:
                count = theme_counter[tema]
                labels.append(tema)
                parents.append(categoria)
                values.append(count)
                cat_total += count
                total += count
        # Actualizar total de categoría
        cat_idx = labels.index(categoria)
        values[cat_idx] = cat_total

    values[0] = total  # Total general

    # Crear sunburst
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(
            colorscale='Viridis',
            cmid=sum([v for v in values if v > 0])/len([v for v in values if v > 0])
        ),
        hovertemplate='<b>%{label}</b><br>Publicaciones: %{value}<br>%{percentParent}<extra></extra>'
    ))

    fig.update_layout(
        title='Jerarquía de Temas por Categoría (Sunburst)',
        height=700,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    return fig


def create_themes_distribution(themes_data):
    """Gráfica de barras de distribución por tema"""
    if not themes_data:
        return None

    # Contar temas
    theme_counter = Counter()
    for pub in themes_data:
        for theme in pub.get('temas', []):
            theme_name = theme.get('nombre', '')
            if theme_name:
                theme_counter[theme_name] += 1

    if not theme_counter:
        return None

    # Crear DataFrame
    themes_df = pd.DataFrame(theme_counter.most_common(15), columns=['Tema', 'Publicaciones'])

    fig = px.bar(
        themes_df,
        x='Publicaciones',
        y='Tema',
        orientation='h',
        title='Top 15 Temas de Investigación',
        labels={'Publicaciones': 'Número de Publicaciones', 'Tema': 'Tema'},
        color='Publicaciones',
        color_continuous_scale='Teal'
    )

    fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})

    return fig


# Interfaz principal
def main():
    # Encabezado con logo
    logo_path = Path(__file__).parent.parent / "LOGO DCNT_1.png"

    col_logo, col_title = st.columns([1, 5])

    with col_logo:
        if logo_path.exists():
            st.image(str(logo_path), width=100)

    with col_title:
        st.markdown("""
        <div style='text-align: left;'>
            <h1 style='color: #1f77b4; margin-bottom: 0;'>Dashboard de Productividad Científica</h1>
            <h2 style='color: #2c3e50; margin-top: 0;'>Doctorado en Ciencias de la Nutrición Traslacional</h2>
            <p style='color: #666; font-size: 1.1rem;'>Universidad de Guadalajara • 2019-2025</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Sidebar - Toggle de tema
    with st.sidebar:
        st.markdown("### ⚙️ Configuración")

        # Toggle para cambiar entre modo claro y oscuro
        theme_option = st.radio(
            "Tema de visualización:",
            options=['☀️ Modo Claro', '🌙 Modo Oscuro'],
            index=0 if st.session_state.theme == 'light' else 1,
            key='theme_selector'
        )

        # Actualizar el tema si cambió
        new_theme = 'dark' if '🌙' in theme_option else 'light'
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()

        st.markdown("---")

    # Cargar datos
    publications_df, ods_data, pronaces_data, themes_data = load_data()

    # Usar todos los años sin filtros
    selected_years = sorted(publications_df['año'].unique())
    filtered_df = publications_df

    # SECCIÓN 0: Contexto y Pertinencia Estratégica
    st.markdown('<div class="section-header">🎯 Contexto y Pertinencia Estratégica</div>', unsafe_allow_html=True)

    st.markdown("""
    El **Doctorado en Ciencias de la Nutrición Traslacional (DCNT-UdeG)** responde directamente a las crisis
    nutricionales más urgentes de México. Este dashboard presenta la evidencia de productividad científica
    que demuestra la **contribución del programa a prioridades nacionales** (PRONACES), **compromisos internacionales** (ODS),
    y **problemáticas regionales críticas**.
    """)

    # Métricas clave de contexto
    st.markdown("### 🚨 Crisis Epidemiológica que Justifica el Programa")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Sobrepeso/Obesidad en Adultos",
            value=f"{EPIDEMIOLOGIA_MEXICO['sobrepeso_obesidad_adultos']['valor']}%",
            delta="75.2% de adultos (México)",
            delta_color="inverse"
        )
        st.caption("ENSANUT 2022")

    with col2:
        st.metric(
            label="Diabetes en Adultos",
            value=f"{EPIDEMIOLOGIA_MEXICO['diabetes_adultos']['valor']}%",
            delta="14.6 millones de personas",
            delta_color="inverse"
        )
        st.caption("ENSANUT 2022")

    with col3:
        st.metric(
            label="Muertes por Obesidad en Jalisco",
            value=f"{EPIDEMIOLOGIA_JALISCO['muertes_obesidad']['asociadas_obesidad']:,}",
            delta="30% de muertes anuales",
            delta_color="inverse"
        )
        st.caption("Secretaría de Salud Jalisco")

    with col4:
        st.metric(
            label="Desnutrición Infantil Jalisco",
            value="2do lugar",
            delta="+88% (2021-2023)",
            delta_color="inverse"
        )
        st.caption("Vigilancia Epidemiológica")

    # Alerta destacada
    st.error("""
    **⚠️ Situación Crítica en Jalisco:**
    - **6,284 casos** de desnutrición infantil en 2023 (incremento de 88% desde 2021)
    - **56.5%** de diabéticos hospitalizados **NO reciben atención nutricional**
    - **1,176,459 personas** con carencia de acceso a alimentación en Jalisco
    - Proyección nacional: **88% con sobrepeso/obesidad** para 2050 sin intervenciones efectivas
    """)

    # Pertinencia del enfoque traslacional
    st.info("""
    **💡 El Enfoque Traslacional del DCNT-UdeG:**

    El programa es único porque forma investigadores capaces de trabajar en todo el continuum de la investigación traslacional:

    - **T0 (Investigación Básica)**: Mecanismos moleculares, interacciones gen-dieta, biomarcadores
    - **T1-T2 (Traslación Clínica)**: Estudios en humanos, protocolos de atención nutricional basados en evidencia
    - **T3-T4 (Traslación Poblacional)**: Implementación en sistemas de salud, políticas públicas escalables

    Este es el paradigma científico del siglo XXI que México necesita para convertir conocimiento básico en **soluciones efectivas** contra la doble carga de malnutrición.
    """)

    st.markdown("---")

    # SECCIÓN 1: Panorama General
    st.markdown('<div class="section-header">📈 Panorama General de Productividad Científica</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total de Publicaciones",
            value=len(filtered_df),
            delta=f"{len(publications_df)} total"
        )

    with col2:
        st.metric(
            label="Años Analizados",
            value=f"{filtered_df['año'].min()} - {filtered_df['año'].max()}",
            delta=f"{len(filtered_df['año'].unique())} años"
        )

    with col3:
        st.metric(
            label="Revistas Únicas",
            value=filtered_df['revista'].nunique()
        )

    with col4:
        st.metric(
            label="Promedio Anual",
            value=f"{len(filtered_df) / len(selected_years):.1f}"
        )

    # Gráfica de evolución
    st.plotly_chart(create_year_evolution_chart(filtered_df), use_container_width=True)

    # SECCIÓN 2: Contribución a ODS
    st.markdown('<div class="section-header">🌍 Contribución a Objetivos de Desarrollo Sostenible</div>', unsafe_allow_html=True)

    if ods_data:
        col1, col2 = st.columns([1, 1])

        with col1:
            fig_ods = create_ods_distribution(ods_data)
            if fig_ods:
                st.plotly_chart(fig_ods, use_container_width=True)
            else:
                st.info("No hay datos de clasificación ODS disponibles")

        with col2:
            st.markdown("### 🎯 Alineación con Metas ODS 2030")

            tabs_ods = st.tabs(["ODS 2", "ODS 3", "ODS 10", "ODS 12"])

            with tabs_ods[0]:
                st.markdown(f"""
                **ODS 2: {ODS_CONTEXTO['ODS_2']['nombre']}**

                *Meta 2.2: {ODS_CONTEXTO['ODS_2']['meta_2_2']}*

                **Situación de México:**
                - 12.8% niños con baja talla (estancado desde 2012)
                - 27.4% en población indígena vs 13.9% promedio nacional
                - Anemia en mujeres: 15.8% (↑ desde 11.6% en 2012)

                **Contribución del DCNT-UdeG:**
                - Investigación en primeros 1000 días de vida
                - Epigenética y genómica nutricional
                - Intervenciones basadas en biomarcadores
                """)

            with tabs_ods[1]:
                st.markdown(f"""
                **ODS 3: {ODS_CONTEXTO['ODS_3']['nombre']}**

                *Meta 3.4: Reducir 1/3 mortalidad prematura por ENT para 2030*

                **Crisis en México:**
                - 18.3% adultos con diabetes (14.6 millones)
                - 75.2% con sobrepeso/obesidad
                - Proyección 2050: 88% con sobrepeso/obesidad

                **Contribución del DCNT-UdeG:**
                - Identificación de poblaciones alto riesgo
                - Intervenciones personalizadas (genómica)
                - Intervenciones poblacionales escalables
                """)

            with tabs_ods[2]:
                st.markdown(f"""
                **ODS 10: {ODS_CONTEXTO['ODS_10']['nombre']}**

                **Brechas Nutricionales Profundas:**
                - Baja talla: 27.4% indígena vs 13.9% promedio
                - Quintil bajo: 20.8% vs mucho menor en quintiles altos
                - Anemia vulnerable: 34.3% en mujeres de bajos recursos

                **Contribución del DCNT-UdeG:**
                - Intervenciones culturalmente pertinentes
                - 25+ años con comunidades Wixárikas
                - Reducción de inequidades en salud
                """)

            with tabs_ods[3]:
                st.markdown(f"""
                **ODS 12: {ODS_CONTEXTO['ODS_12']['nombre']}**

                **Situación de México:**
                - Desperdicio: 20.4 millones ton/año (34% producción)
                - Ultraprocesados: 46.6% del consumo total
                - 27% muertes diabetes relacionadas con bebidas azucaradas

                **Contribución del DCNT-UdeG:**
                - Evaluación alimentos tradicionales vs ultraprocesados
                - Alimentos funcionales (biodiversidad mexicana)
                - Sistemas alimentarios sostenibles
                """)
    else:
        st.warning("⚠️ Ejecuta primero el script de clasificación paralela para generar los datos de ODS")
        st.code("python src/classify_parallel.py", language="bash")

    # SECCIÓN 3: Alineación con PRONACES
    st.markdown('<div class="section-header">🇲🇽 Alineación con PRONACES - Prioridades Nacionales</div>', unsafe_allow_html=True)

    # Contexto PRONACES
    st.info(f"""
    **Programas Nacionales Estratégicos de Ciencia y Tecnología (PRONACES)**

    La política científica más importante de México para 2023-2025:
    - **Inversión:** {PRONACES_CONTEXTO['inversion_total']['monto']} en {PRONACES_CONTEXTO['inversion_total']['proyectos']} proyectos
    - **Alcance:** {PRONACES_CONTEXTO['inversion_total']['personas']} personas en {PRONACES_CONTEXTO['inversion_total']['instituciones']} instituciones
    - **Modelo:** Transdisciplinario que integra academia-gobierno-comunidad con acceso abierto
    """)

    if pronaces_data:
        col1, col2 = st.columns([2, 1])

        with col1:
            fig_pronaces = create_pronaces_heatmap(pronaces_data, filtered_df)
            if fig_pronaces:
                st.plotly_chart(fig_pronaces, use_container_width=True)

        with col2:
            st.markdown("### Alineación con PRONACES Prioritarios")

            with st.expander("🏥 PRONACE SALUD", expanded=True):
                st.markdown(f"""
                **Áreas Prioritarias:**
                - {PRONACES_CONTEXTO['PRONACE_SALUD']['areas_prioritarias'][0]}
                - {PRONACES_CONTEXTO['PRONACE_SALUD']['areas_prioritarias'][1]}
                - {PRONACES_CONTEXTO['PRONACE_SALUD']['areas_prioritarias'][2]}

                **Financiamiento:** {PRONACES_CONTEXTO['PRONACE_SALUD']['financiamiento']}

                **Líneas DCNT-UdeG Alineadas:**
                - Genómica Nutricional → Medicina de sistemas
                - Salud Pública → Alimentación comunitaria
                - Alimentación y Nutrición → Patrones alimentarios
                """)

            with st.expander("🌾 PRONACE SOBERANÍA ALIMENTARIA"):
                st.markdown(f"""
                **Alcance Actual:**
                - {PRONACES_CONTEXTO['PRONACE_SOBERANIA_ALIMENTARIA']['pronaii_activos']} PRONAII activos
                - {PRONACES_CONTEXTO['PRONACE_SOBERANIA_ALIMENTARIA']['localidades']} localidades
                - {PRONACES_CONTEXTO['PRONACE_SOBERANIA_ALIMENTARIA']['organizaciones_comunitarias']} organizaciones comunitarias

                **Demandas Prioritarias:**
                - Alimentación saludable y culturalmente adecuada
                - Alimentos funcionales
                - Calidad nutrimental maíz-tortilla

                **Líneas DCNT-UdeG Alineadas:**
                - Salud Pública → Educación alimentaria
                - Alimentación y Nutrición → Ciencias de alimentos
                """)

        st.success("""
        ✅ **Alta Pertinencia Demostrada:** El DCNT-UdeG forma recursos humanos especializados
        para liderar o participar en futuros PRONAII, con competencias específicas en investigación
        traslacional, vinculación con gobierno/comunidad, y trabajo multidisciplinario.
        """)

    else:
        st.warning("⚠️ Ejecuta primero el script de clasificación para generar los datos de PRONACES")

    # SECCIÓN 4: Análisis Temático
    st.markdown('<div class="section-header">🔬 Análisis Temático de la Investigación</div>', unsafe_allow_html=True)

    if themes_data:
        st.markdown("""
        Visualizaciones interactivas que muestran la **distribución de temas de investigación**,
        su **jerarquía** y las **relaciones entre temas** en las publicaciones del doctorado.
        """)

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Distribución",
            "🎯 Jerarquía (Treemap)",
            "🌅 Categorías (Sunburst)",
            "🔥 Co-ocurrencia"
        ])

        with tab1:
            st.markdown("### Distribución de Publicaciones por Tema")
            fig_themes_dist = create_themes_distribution(themes_data)
            if fig_themes_dist:
                st.plotly_chart(fig_themes_dist, use_container_width=True)
                st.caption("💡 **Interpretación:** Los temas más frecuentes indican las áreas de mayor investigación del programa.")

        with tab2:
            st.markdown("### Treemap - Vista Jerárquica")
            fig_treemap = create_themes_treemap(themes_data)
            if fig_treemap:
                st.plotly_chart(fig_treemap, use_container_width=True)
                st.caption("""
                💡 **Interpretación:** El tamaño de cada bloque representa el número de publicaciones en ese tema.
                Los colores indican la intensidad de investigación. Haz clic en un bloque para enfocarte en él.
                """)

        with tab3:
            st.markdown("### Sunburst - Categorización de Temas")
            fig_sunburst = create_themes_sunburst(themes_data)
            if fig_sunburst:
                st.plotly_chart(fig_sunburst, use_container_width=True)
                st.caption("""
                💡 **Interpretación:** Los temas están organizados en categorías (anillo interior).
                Haz clic en una categoría para ver sus temas específicos. El tamaño de cada segmento
                indica el número de publicaciones.
                """)

        with tab4:
            st.markdown("### Matriz de Co-ocurrencia - Relaciones entre Temas")
            fig_cooccurrence = create_themes_cooccurrence(themes_data)
            if fig_cooccurrence:
                st.plotly_chart(fig_cooccurrence, use_container_width=True)
                st.caption("""
                💡 **Interpretación:** Esta matriz muestra cuántas veces dos temas aparecen juntos en la misma publicación.
                Valores altos (colores más oscuros) indican temas frecuentemente investigados en conjunto,
                revelando las conexiones interdisciplinarias del programa.
                """)

                # Análisis adicional de conexiones fuertes
                st.markdown("#### 🔗 Conexiones Interdisciplinarias Destacadas")

                # Encontrar las 5 co-ocurrencias más fuertes
                all_themes = set()
                for pub in themes_data:
                    for theme in pub.get('temas', []):
                        theme_name = theme.get('nombre', '')
                        if theme_name:
                            all_themes.add(theme_name)

                all_themes = sorted(list(all_themes))
                connections = []

                for pub in themes_data:
                    pub_themes = [t.get('nombre', '') for t in pub.get('temas', []) if t.get('nombre', '')]
                    for i, theme1 in enumerate(pub_themes):
                        for theme2 in pub_themes[i+1:]:
                            connections.append((theme1, theme2))

                from collections import Counter
                top_connections = Counter(connections).most_common(5)

                if top_connections:
                    st.markdown("**Top 5 combinaciones de temas más frecuentes:**")
                    for (t1, t2), count in top_connections:
                        st.write(f"- **{t1}** ↔ **{t2}**: {count} publicaciones")

                    st.info("""
                    Estas conexiones muestran el **enfoque interdisciplinario** del programa, donde se integran
                    múltiples áreas de conocimiento para abordar problemas complejos de nutrición y salud.
                    """)
    else:
        st.warning("⚠️ Ejecuta primero el script de clasificación para generar los datos temáticos")

    # SECCIÓN 4.5: Líneas de Investigación del Doctorado
    st.markdown('<div class="section-header">🎓 Líneas de Investigación del DCNT-UdeG</div>', unsafe_allow_html=True)

    st.markdown("""
    El DCNT-UdeG opera con **tres líneas de investigación complementarias** que cubren todo el espectro
    de la investigación traslacional en nutrición: desde mecanismos moleculares hasta intervenciones poblacionales.
    """)

    tab1, tab2, tab3 = st.tabs([
        "🧬 Línea 1: Genómica Nutricional",
        "👥 Línea 2: Salud Pública",
        "🍽️ Línea 3: Alimentación y Nutrición"
    ])

    with tab1:
        linea1 = LINEAS_INVESTIGACION['linea_1']
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown(f"### {linea1['nombre']}")
            st.markdown(f"**{linea1['descripcion']}**")

            st.markdown("**Áreas de Investigación:**")
            for area in linea1['areas_investigacion']:
                st.markdown(f"- {area}")

        with col2:
            st.info(f"""
            **Fase Traslacional:** T0-T1 (Básica a Clínica)

            **Aplicaciones:**
            - {linea1['aplicaciones'][0]}
            - {linea1['aplicaciones'][1]}
            - {linea1['aplicaciones'][2]}

            **Campo Laboral:**
            - {linea1['campo_laboral'][0]}
            - {linea1['campo_laboral'][1]}
            - {linea1['campo_laboral'][2]}
            """)

        st.success("""
        **Relevancia Única para México:** La población mexicana tiene alta diversidad genética por su composición
        mestiza e indígena única. La investigación nutrigenómica específica en población local NO es extrapolable
        de estudios europeos o asiáticos.
        """)

    with tab2:
        linea2 = LINEAS_INVESTIGACION['linea_2']
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown(f"### {linea2['nombre']}")
            st.markdown(f"**{linea2['descripcion']}**")

            st.markdown("**Áreas de Investigación:**")
            for area in linea2['areas_investigacion']:
                st.markdown(f"- {area}")

        with col2:
            st.info(f"""
            **Fase Traslacional:** T3-T4 (Práctica a Población)

            **Aplicaciones:**
            - {linea2['aplicaciones'][0]}
            - {linea2['aplicaciones'][1]}
            - {linea2['aplicaciones'][3]}

            **Campo Laboral:**
            - {linea2['campo_laboral'][0]}
            - {linea2['campo_laboral'][1]}
            - {linea2['campo_laboral'][2]}
            """)

        st.success("""
        **Impacto en Salud Pública:** Forma investigadores con competencias para diseñar, implementar y
        evaluar intervenciones nutricionales poblacionales, contribuyendo a la generación de evidencia
        científica que pueda informar políticas públicas basadas en el contexto mexicano.
        """)

    with tab3:
        linea3 = LINEAS_INVESTIGACION['linea_3']
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown(f"### {linea3['nombre']}")
            st.markdown(f"**{linea3['descripcion']}**")

            st.markdown("**Áreas de Investigación:**")
            for area in linea3['areas_investigacion']:
                st.markdown(f"- {area}")

        with col2:
            st.info(f"""
            **Fase Traslacional:** T1-T2 (Clínica a Pacientes)

            **Aplicaciones:**
            - {linea3['aplicaciones'][0]}
            - {linea3['aplicaciones'][1]}
            - {linea3['aplicaciones'][2]}

            **Campo Laboral:**
            - {linea3['campo_laboral'][0]}
            - {linea3['campo_laboral'][1]}
            - {linea3['campo_laboral'][3]}
            """)

        st.success("""
        **Biodiversidad Mexicana:** Investigación puede desarrollar alimentos funcionales aprovechando
        biodiversidad única (nopal, amaranto, chía, quelites, aguamiel) reduciendo dependencia de importaciones
        y promoviendo sistemas alimentarios sostenibles.
        """)

    st.markdown("---")

    # # SECCIÓN 5: Impacto en Problemas Alimentario-Nutricios (DESACTIVADA)
    # st.markdown('<div class="section-header">🍎 Impacto en Problemáticas Alimentario-Nutricias de México</div>', unsafe_allow_html=True)

    # st.markdown("""
    # La investigación del DCNT-UdeG aborda las **4 problemáticas alimentario-nutricias críticas** identificadas
    # en el análisis epidemiológico nacional actualizado (ENSANUT 2022-2023, Sistema de Vigilancia 2024).
    # """)

    #     # # Problema 1: Epidemia de obesidad
    #     # with st.expander("🚨 1. EPIDEMIA DE OBESIDAD Y ENFERMEDADES METABÓLICAS SIN CONTROL", expanded=True):
    #         col1, col2, col3 = st.columns(3)
    # 
    #         with col1:
    #             st.metric("Sobrepeso/Obesidad Adultos", "75.2%", "38.3% sobrepeso + 36.9% obesidad")
    #             st.caption("ENSANUT 2022")
    # 
    #         with col2:
    #             st.metric("Diabetes Adultos", "18.3%", "14.6 millones de personas")
    #             st.caption("ENSANUT 2022")
    # 
    #         with col3:
    #             st.metric("Sin Atención Nutricional", "56.5%", "Diabéticos hospitalizados")
    #             st.caption("Vigilancia Hospitalaria 2023")
    # 
    #         st.markdown(f"""
    #         **Situación:**
    #         - **Obesidad abdominal:** {EPIDEMIOLOGIA_MEXICO['obesidad_abdominal']['valor']}% de adultos
    #         - **Mortalidad:** {EPIDEMIOLOGIA_MEXICO['mortalidad_diabetes']['valor']:,} muertes por diabetes en 2023
    #         - **Actividad física inadecuada:** Solo 15.7% de diabéticos hace actividad física diaria adecuada
    #         - **Proyección:** 88% con sobrepeso/obesidad para 2050 sin intervenciones efectivas
    # 
    #         **Contribución del DCNT-UdeG:**
    #         - **Línea 1 (Genómica):** Identificar variantes genéticas que predisponen a diabetes/obesidad en población mexicana
    #         - **Línea 2 (Salud Pública):** Diseñar intervenciones escalables para prevención primaria
    #         - **Línea 3 (Alimentación):** Desarrollar protocolos efectivos de atención nutricional para los 580 centros de salud de Jalisco
    #         """)
    # 
    #     # Problema 2: Desnutrición infantil
    #     with st.expander("📉 2. DESNUTRICIÓN INFANTIL ESTANCADA CON INCREMENTOS ALARMANTES"):
    #         col1, col2, col3 = st.columns(3)
    # 
    #         with col1:
    #             st.metric("Baja Talla Infantil", "12.8%", "Estancado desde 2012", delta_color="inverse")
    # 
    #         with col2:
    #             st.metric("Anemia Mujeres", "15.8%", "↑ de 11.6% en 2012", delta_color="inverse")
    # 
    #         with col3:
    #             st.metric("Jalisco Desnutrición", "+88%", "Incremento 2021-2023", delta_color="inverse")
    # 
    #         st.markdown(f"""
    #         **Situación:**
    #         - **Nacional:** {EPIDEMIOLOGIA_MEXICO['desnutricion_infantil']['valor']}% niños menores de 5 años con baja talla (sin avances desde 2012)
    #         - **Población indígena:** 27.4% baja talla vs 13.9% promedio nacional
    #         - **Jalisco:** {EPIDEMIOLOGIA_JALISCO['desnutricion_casos']['2023']:,} casos en 2023 (vs {EPIDEMIOLOGIA_JALISCO['desnutricion_casos']['2021']:,} en 2021) = **{EPIDEMIOLOGIA_JALISCO['desnutricion_casos']['incremento_porcentual']}% incremento**
    #         - **Meta global:** México proyectado a cumplir solo 1 de 6 metas nutricionales para 2025/2030
    # 
    #         **Contribución del DCNT-UdeG:**
    #         - **Investigación primeros 1000 días:** Genómica nutricional y epigenética para identificar ventanas críticas de intervención
    #         - **Suplementación guiada por biomarcadores:** Más efectiva que suplementación universal
    #         - **Evaluación rigurosa:** Programas DIF Jalisco "Primeros 1000 Días" para mejorarlos y escalarlos
    #         """)
    # 
    #     # Problema 3: Transición nutricional
    #     with st.expander("🍔 3. TRANSICIÓN NUTRICIONAL CON CAMBIOS DIETÉTICOS PERJUDICIALES"):
    #         col1, col2 = st.columns(2)
    # 
    #         with col1:
    #             st.metric("Consumo Ultraprocesados", "46.6%", "+7.1 puntos en 20 años", delta_color="inverse")
    # 
    #         with col2:
    #             st.metric("Muertes Diabetes por Bebidas", "27%", "Relacionadas con bebidas azucaradas", delta_color="inverse")
    # 
    #         st.markdown(f"""
    #         **Situación:**
    #         - **Ultraprocesados:** {EPIDEMIOLOGIA_MEXICO['ultraprocesados']['valor']}% del consumo total (incremento de 7.1 puntos 2000-2020)
    #         - **Bebidas azucaradas:** 163 litros per cápita/año, 27% de muertes por diabetes relacionadas
    #         - **Frutas y verduras:** Menos del 50% consume regularmente
    #         - **Niños pequeños:** 42% consume alimentos no saludables (6-23 meses), 87% preescolares consume bebidas endulzadas
    # 
    #         **Contribución del DCNT-UdeG:**
    #         - **Estudios de aceptabilidad:** Alimentos tradicionales saludables vs ultraprocesados
    #         - **Evaluación etiquetado frontal:** Implementado en México desde 2020
    #         - **Alimentos funcionales:** Basados en biodiversidad mexicana (nopal, amaranto, chía, aguamiel)
    #         - **Educación nutricional:** Estrategias culturalmente apropiadas
    #         """)
    # 
    #     # Problema 4: Inseguridad alimentaria
    #     with st.expander("🍞 4. INSEGURIDAD ALIMENTARIA PERSISTENTE CON BRECHAS SOCIOECONÓMICAS"):
    #         col1, col2 = st.columns(2)
    # 
    #         with col1:
    #             st.metric("Carencia Alimentaria Nacional", "18.2%", "23.4 millones mexicanos", delta_color="inverse")
    # 
    #         with col2:
    #             st.metric("Carencia Jalisco", "1,176,459", "Personas sin acceso", delta_color="inverse")
    # 
    #         st.markdown(f"""
    #         **Situación:**
    #         - **Nacional:** {EPIDEMIOLOGIA_MEXICO['inseguridad_alimentaria']['valor']}% población ({EPIDEMIOLOGIA_MEXICO['inseguridad_alimentaria']['personas']})
    #         - **Inseguridad severa:** 8.2 millones de mexicanos
    #         - **Desperdicio alimentario:** 20.4 millones ton/año (34% producción) mientras hay inseguridad alimentaria
    #         - **Anemia en vulnerables:** 34.3% en mujeres con menores capacidades económicas
    # 
    #         **Contribución del DCNT-UdeG:**
    #         - **Fuentes bajo costo:** Identificar alimentos de bajo costo y alto valor nutricional (biodiversidad local)
    #         - **Evaluación programas sociales:** Sembrando Vida (441,466 beneficiarios), Jóvenes Construyendo el Futuro
    #         - **Intervenciones costo-efectivas:** Viables en contextos de pobreza
    #         - **Reducción inequidades:** Investigación culturalmente pertinente (25+ años con Wixárikas)
    #         """)
    # 
    st.markdown("---")

    # SECCIÓN 6: Pertinencia Regional
    st.markdown('<div class="section-header">🗺️ Pertinencia Regional: Jalisco como Epicentro de Crisis Nutricional</div>', unsafe_allow_html=True)

    st.markdown("""
    El DCNT-UdeG no es un programa académico abstracto sino una **respuesta institucional urgente a crisis
    de salud pública regional**. Jalisco y la región Occidente presentan problemáticas específicas que justifican
    este programa doctoral único en la región.
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🚨 Crisis en Jalisco")

        st.error(f"""
        **Jalisco: {EPIDEMIOLOGIA_JALISCO['desnutricion_casos']['ranking']} en desnutrición infantil**

        - **{EPIDEMIOLOGIA_JALISCO['desnutricion_casos']['2023']:,} casos** en 2023
        - **+{EPIDEMIOLOGIA_JALISCO['desnutricion_casos']['incremento_porcentual']}%** incremento desde 2021
        - **{EPIDEMIOLOGIA_JALISCO['muertes_obesidad']['asociadas_obesidad']:,} muertes anuales** asociadas a obesidad ({EPIDEMIOLOGIA_JALISCO['muertes_obesidad']['porcentaje']}% del total)
        """)

        st.warning(f"""
        **Brechas Críticas en Atención:**

        - **{EPIDEMIOLOGIA_JALISCO['sin_atencion_nutricional']['valor']}%** de diabéticos hospitalizados **NO reciben atención nutricional**
        - **{EPIDEMIOLOGIA_JALISCO['carencia_alimentaria']['personas']:,} personas** con carencia de acceso a alimentación
        - **{EPIDEMIOLOGIA_JALISCO['diabetes_casos']['ranking']}** junto con CDMX
        """)

    with col2:
        st.markdown("### 🌎 Alcance Regional")

        st.info(f"""
        **Región Occidente:**

        - **Población:** {REGION_OCCIDENTE['poblacion']}
        - **PIB Nacional:** {REGION_OCCIDENTE['pib_nacional']}
        - **Estados:** {', '.join(REGION_OCCIDENTE['estados'])}

        **Jalisco como Líder:**
        - {REGION_OCCIDENTE['lider_regional']['economia']}
        - {REGION_OCCIDENTE['lider_regional']['pib_regional']} del PIB regional
        - IPS: {REGION_OCCIDENTE['lider_regional']['ips']} (mejor de la región)
        """)

        st.success(f"""
        **Vacío de Formación Doctoral:**

        {REGION_OCCIDENTE['deficit_formacion']}

        Candidatos debían migrar a CDMX o extranjero, con baja tasa de retorno.
        """)

    #     # Infraestructura robusta
    #     st.markdown("### 🏛️ Infraestructura Institucional Robusta Lista para Potenciar el Programa")
    # 
    #     col1, col2, col3 = st.columns(3)
    # 
    #     with col1:
    #         st.markdown("**CUCS - Universidad de Guadalajara**")
    #         st.metric("Investigadores SNI", INFRAESTRUCTURA_UDG['cucs']['investigadores_sni'])
    #         st.metric("Profesores Tiempo Completo", INFRAESTRUCTURA_UDG['cucs']['profesores_tiempo_completo'])
    #         st.metric("Artículos Anuales", INFRAESTRUCTURA_UDG['cucs']['articulos_anuales'])
    # 
    #     with col2:
    #         st.markdown("**Hospital Civil de Guadalajara**")
    #         st.metric("Investigadores SNI", INFRAESTRUCTURA_UDG['hospital_civil']['investigadores_sni'])
    #         st.metric("Publicaciones Anuales", INFRAESTRUCTURA_UDG['hospital_civil']['publicaciones_anuales'])
    #         st.metric("Proyectos Activos", INFRAESTRUCTURA_UDG['hospital_civil']['proyectos_investigacion'])
    # 
    #     with col3:
    #         st.markdown("**Red Asistencial Jalisco**")
    #         st.metric("Centros de Salud", INFRAESTRUCTURA_UDG['ss_jalisco']['centros_salud'])
    #         st.metric("Hospitales", INFRAESTRUCTURA_UDG['ss_jalisco']['hospitales'])
    #         st.caption("Sistema urgencias mejor de Latinoamérica")
    # 
    #     st.success(f"""
    #     **INHU (Instituto de Nutrición Humana):**
    #     - **{INFRAESTRUCTURA_UDG['inhu']['años_operacion']}** de experiencia en investigación materno-infantil
    #     - Maestría en Nutrición Humana **{INFRAESTRUCTURA_UDG['inhu']['maestria_pnpc']}** en PNPC-CONAHCYT
    #     - **{INFRAESTRUCTURA_UDG['inhu']['generaciones_formadas']} generaciones** de egresados formados
    # 
    #     **CMNO-IMSS (Centro Médico Nacional de Occidente):**
    #     - **{INFRAESTRUCTURA_UDG['cmno_imss']['usuarios_potenciales']}** (30% de derechohabientes IMSS nacional)
    #     - {INFRAESTRUCTURA_UDG['cmno_imss']['infraestructura']}
    # 
    #     Esta infraestructura representa la base sobre la cual el DCNT-UdeG puede realizar **investigación traslacional
    #     de impacto inmediato** escalable a nivel nacional.
    #     """)

    st.markdown("---")

    # SECCIÓN 7: Tabla de Publicaciones
    st.markdown('<div class="section-header">📚 Publicaciones Detalladas</div>', unsafe_allow_html=True)

    # Selector de año
    year_filter = st.selectbox("Filtrar por año:", ["Todos"] + sorted(filtered_df['año'].unique().tolist(), reverse=True))

    if year_filter != "Todos":
        display_df = filtered_df[filtered_df['año'] == year_filter]
    else:
        display_df = filtered_df

    # Mostrar tabla
    st.dataframe(
        display_df[['año', 'numero', 'titulo', 'revista', 'doi']].style.set_properties(**{
            'text-align': 'left'
        }),
        use_container_width=True,
        height=400
    )

    # Footer
    st.markdown("---")

    # Footer con logo al lado del texto
    footer_col1, footer_col_logo, footer_col_text, footer_col2 = st.columns([1, 1, 3, 1])

    with footer_col1:
        st.write("")  # Espacio

    with footer_col_logo:
        if logo_path.exists():
            st.image(str(logo_path), width=100)

    with footer_col_text:
        st.markdown("""
        <div style='color: #666; padding: 1rem 0;'>
            <p style='margin: 0.3rem 0;'><strong style='color: #1f77b4; font-size: 1.2rem;'>Doctorado en Ciencias de la Nutrición Traslacional</strong></p>
            <p style='margin: 0.2rem 0;'>Universidad de Guadalajara</p>
            <p style='margin: 0.2rem 0;'>Centro Universitario de Ciencias de la Salud (CUCS)</p>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.75rem; color: #999;'>Desarrollado por: José Gerardo Mora Almanza - Alumno del DCNT</p>
        </div>
        """, unsafe_allow_html=True)

    with footer_col2:
        st.write("")  # Espacio


if __name__ == "__main__":
    main()
