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
            /* =========================== TEMA OSCURO =========================== */

            /* Fondo principal de la app */
            .stApp {
                background-color: #0e1117 !important;
            }

            /* Fondo del contenido principal */
            .main .block-container {
                background-color: #0e1117 !important;
            }

            /* Sidebar */
            [data-testid="stSidebar"] {
                background-color: #262730 !important;
            }

            [data-testid="stSidebar"] * {
                color: #fafafa !important;
            }

            /* Headers personalizados */
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                color: #58a6ff !important;
                text-align: center;
                padding: 1rem 0;
            }

            .section-header {
                font-size: 1.8rem;
                font-weight: bold;
                color: #e6edf3 !important;
                border-bottom: 3px solid #58a6ff;
                padding-bottom: 0.5rem;
                margin-top: 2rem;
            }

            /* Texto general */
            .stMarkdown, p, span, label, div {
                color: #c9d1d9 !important;
            }

            h1, h2, h3, h4, h5, h6 {
                color: #e6edf3 !important;
            }

            /* Métricas */
            [data-testid="stMetric"] {
                background-color: #161b22 !important;
                padding: 1rem;
                border-radius: 8px;
                border: 1px solid #30363d;
            }

            [data-testid="stMetricValue"] {
                color: #58a6ff !important;
            }

            [data-testid="stMetricLabel"] {
                color: #8b949e !important;
            }

            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                background-color: #161b22 !important;
            }

            .stTabs [data-baseweb="tab"] {
                color: #8b949e !important;
                background-color: #0d1117 !important;
            }

            .stTabs [aria-selected="true"] {
                color: #58a6ff !important;
                background-color: #161b22 !important;
                border-bottom-color: #58a6ff !important;
            }

            /* Radio buttons */
            [data-testid="stRadio"] label {
                color: #c9d1d9 !important;
            }

            /* Dividers */
            hr {
                border-color: #30363d !important;
            }

            /* Success/Info boxes */
            .stAlert {
                background-color: #161b22 !important;
                color: #c9d1d9 !important;
                border: 1px solid #30363d;
            }
        </style>
        """
    else:
        return """
        <style>
            /* =========================== TEMA CLARO =========================== */

            /* Fondo principal de la app */
            .stApp {
                background-color: #ffffff !important;
            }

            /* Fondo del contenido principal */
            .main .block-container {
                background-color: #ffffff !important;
            }

            /* Sidebar */
            [data-testid="stSidebar"] {
                background-color: #f0f2f6 !important;
            }

            [data-testid="stSidebar"] * {
                color: #31333F !important;
            }

            /* Headers personalizados */
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                color: #1f77b4 !important;
                text-align: center;
                padding: 1rem 0;
            }

            .section-header {
                font-size: 1.8rem;
                font-weight: bold;
                color: #2c3e50 !important;
                border-bottom: 3px solid #1f77b4;
                padding-bottom: 0.5rem;
                margin-top: 2rem;
            }

            /* Texto general */
            .stMarkdown, p, span, label, div {
                color: #31333F !important;
            }

            h1, h2, h3, h4, h5, h6 {
                color: #262730 !important;
            }

            /* Métricas */
            [data-testid="stMetric"] {
                background-color: #f8f9fa !important;
                padding: 1rem;
                border-radius: 8px;
                border: 1px solid #e6e9ef;
            }

            [data-testid="stMetricValue"] {
                color: #1f77b4 !important;
            }

            [data-testid="stMetricLabel"] {
                color: #666666 !important;
            }

            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                background-color: #f0f2f6 !important;
            }

            .stTabs [data-baseweb="tab"] {
                color: #666666 !important;
                background-color: #ffffff !important;
            }

            .stTabs [aria-selected="true"] {
                color: #1f77b4 !important;
                background-color: #ffffff !important;
                border-bottom-color: #1f77b4 !important;
            }

            /* Radio buttons */
            [data-testid="stRadio"] label {
                color: #31333F !important;
            }

            /* Dividers */
            hr {
                border-color: #e6e9ef !important;
            }

            /* Success/Info boxes */
            .stAlert {
                background-color: #f8f9fa !important;
                color: #31333F !important;
                border: 1px solid #e6e9ef;
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
        with open(base_dir / "classifications" / "ods_classification_embeddings.json", 'r', encoding='utf-8') as f:
            ods_full = json.load(f)
            # Extraer solo la lista de artículos
            ods_data = ods_full.get('articulos', [])
    except FileNotFoundError:
        ods_data = []

    # Cargar clasificación PRONACES (embeddings)
    try:
        with open(base_dir / "classifications" / "pronaces_classification_embeddings.json", 'r', encoding='utf-8') as f:
            pronaces_full = json.load(f)
            # Extraer solo la lista de artículos
            pronaces_data = pronaces_full.get('articulos', [])
    except FileNotFoundError:
        pronaces_data = []

    try:
        with open(base_dir / "classifications" / "themes_classification.json", 'r', encoding='utf-8') as f:
            themes_data = json.load(f)
    except FileNotFoundError:
        themes_data = []

    try:
        with open(base_dir / "classifications" / "lineas_classification.json", 'r', encoding='utf-8') as f:
            lineas_data = json.load(f)
    except FileNotFoundError:
        lineas_data = None

    return publications, ods_data, pronaces_data, themes_data, lineas_data


@st.cache_data
def load_pubmed_metadata():
    """Carga metadata enriquecida de PubMed con citaciones, MeSH terms, keywords, etc."""
    base_dir = Path(__file__).parent.parent / "data" / "pubmed_extracted"

    try:
        with open(base_dir / "metadata_updated_20251024_043156.json", 'r', encoding='utf-8') as f:
            pubmed_data = json.load(f)
        return pubmed_data
    except FileNotFoundError:
        return []


def create_year_evolution_chart(df):
    """Gráfica de evolución anual de publicaciones con tema DCNT"""
    publications_per_year = df.groupby('año').size().reset_index(name='Publicaciones')

    # Colores inspirados en el logo del DCNT (verde-azul institucional)
    # Con gradiente que mejora la visibilidad de todos los años
    dcnt_colors = {
        2019: '#004C6D',  # Azul oscuro
        2020: '#005F89',  # Azul medio-oscuro
        2021: '#0072A5',  # Azul medio
        2022: '#0086C1',  # Azul claro (más visible)
        2023: '#009ADD',  # Azul-cyan
        2024: '#00AEF9',  # Cyan brillante
        2025: '#17C3FF'   # Cyan muy brillante
    }

    # Asignar colores específicos a cada año
    publications_per_year['color'] = publications_per_year['año'].map(dcnt_colors)

    fig = px.bar(
        publications_per_year,
        x='año',
        y='Publicaciones',
        title='Evolución de la Productividad Científica DCNT-UdeG (2019-2025)',
        labels={'año': 'Año', 'Publicaciones': 'Número de Publicaciones'},
        color='año',
        color_discrete_map=dcnt_colors,
        text='Publicaciones'
    )

    # Agregar valores en las barras para mejor legibilidad
    fig.update_traces(
        texttemplate='%{text}',
        textposition='outside',
        textfont_size=12,
        marker_line_color='rgba(0,0,0,0.3)',
        marker_line_width=1.5
    )

    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            title_font=dict(size=14, family='Arial, sans-serif', color='#2c3e50'),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title_font=dict(size=14, family='Arial, sans-serif', color='#2c3e50'),
            tickfont=dict(size=12),
            gridcolor='rgba(0,0,0,0.1)'
        ),
        title=dict(
            font=dict(size=16, family='Arial, sans-serif', color='#2c3e50'),
            x=0.5,
            xanchor='center'
        ),
        hovermode='x unified',
        height=450,
        showlegend=False,
        plot_bgcolor='rgba(240,248,255,0.3)',  # Fondo muy suave azul
        paper_bgcolor='white'
    )

    # Agregar anotación para destacar el año 2022 si tiene pocas publicaciones
    year_2022_data = publications_per_year[publications_per_year['año'] == 2022]
    if not year_2022_data.empty and year_2022_data.iloc[0]['Publicaciones'] < 10:
        fig.add_annotation(
            x=2022,
            y=year_2022_data.iloc[0]['Publicaciones'] + 2,
            text="↓",
            showarrow=False,
            font=dict(size=20, color='#0086C1')
        )

    return fig


def create_ods_distribution(ods_data):
    """Gráfica de distribución por ODS con barras horizontales para mejor visibilidad"""
    if not ods_data:
        return None

    # Contar ODS
    ods_counter = Counter()
    total_articulos = len(ods_data)

    for pub in ods_data:
        for ods in pub.get('ods_principales', []):
            ods_name = f"ODS {ods.get('numero')}: {ods.get('nombre', '')}"
            ods_counter[ods_name] += 1

    if not ods_counter:
        return None

    # Crear DataFrame y ordenar por número de ODS
    ods_list = []
    for ods_name, count in ods_counter.items():
        ods_num = int(ods_name.split(':')[0].replace('ODS ', ''))
        percentage = (count / total_articulos) * 100
        ods_list.append({
            'ODS': ods_name,
            'Publicaciones': count,
            'Porcentaje': percentage,
            'ODS_Num': ods_num,
            'Texto': f"{count} ({percentage:.1f}%)"
        })

    ods_df = pd.DataFrame(ods_list)
    ods_df = ods_df.sort_values('ODS_Num')

    # Definir colores por ODS (colores institucionales y temáticos)
    colors_map = {
        'ODS 1: Fin de la Pobreza': '#E5243B',
        'ODS 2: Hambre Cero': '#DDA83A',
        'ODS 3: Salud y Bienestar': '#4C9F38',
        'ODS 5: Igualdad de Género': '#FF3A21',
        'ODS 10: Reducir Desigualdades': '#DD1367',
        'ODS 12: Producción y Consumo': '#BF8B2E',
        'ODS 15: Vida de Ecosistemas': '#56C02B',
        'ODS 17: Alianzas': '#00689D'
    }

    # Gráfica de barras horizontales
    fig = px.bar(
        ods_df,
        x='Publicaciones',
        y='ODS',
        orientation='h',
        title='Distribución de Publicaciones por Objetivos de Desarrollo Sostenible',
        text='Texto',
        color='ODS',
        color_discrete_map=colors_map,
        labels={'Publicaciones': 'Número de Publicaciones'}
    )

    # Personalizar el gráfico
    fig.update_traces(
        textposition='outside',
        textfont_size=12,
        hovertemplate='<b>%{y}</b><br>Publicaciones: %{x}<br><extra></extra>'
    )

    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis=dict(
            title='Número de Publicaciones',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title='',
            automargin=True
        ),
        plot_bgcolor='rgba(240,248,255,0.3)',
        paper_bgcolor='white',
        font=dict(size=11)
    )

    return fig


def filter_articulos_by_ods(ods_data, ods_num):
    """
    Filtra artículos por número de ODS

    Args:
        ods_data: Lista de artículos con clasificación ODS
        ods_num: Número de ODS (2, 3, 10, 12, etc.)

    Returns:
        DataFrame con artículos filtrados
    """
    articulos_filtrados = []

    for art in ods_data:
        for ods in art.get('ods_principales', []):
            if ods.get('numero') == ods_num:
                articulos_filtrados.append({
                    'Año': art.get('año', ''),
                    'Título': art.get('titulo', ''),
                    'Justificación': art.get('justificacion', 'N/A')[:100] + '...' if len(art.get('justificacion', '')) > 100 else art.get('justificacion', 'N/A')
                })
                break

    if not articulos_filtrados:
        return None

    df = pd.DataFrame(articulos_filtrados)
    df = df.sort_values('Año', ascending=False)
    return df


def get_ods_stats(ods_data):
    """
    Obtiene estadísticas por ODS

    Returns:
        dict con conteo de artículos por ODS
    """
    stats = {}
    for pub in ods_data:
        for ods in pub.get('ods_principales', []):
            num = ods.get('numero')
            if num:
                stats[num] = stats.get(num, 0) + 1
    return stats


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


# ============================ FUNCIONES ANÁLISIS MeSH ============================

def create_mesh_distribution(pubmed_data, top_n=20):
    """Distribución de los términos MeSH más frecuentes"""
    if not pubmed_data:
        return None

    # Contar términos MeSH
    mesh_counter = Counter()
    for article in pubmed_data:
        mesh_terms = article.get('mesh_terms', [])
        mesh_counter.update(mesh_terms)

    if not mesh_counter:
        return None

    # Top N términos
    top_mesh = mesh_counter.most_common(top_n)
    mesh_df = pd.DataFrame(top_mesh, columns=['Término MeSH', 'Frecuencia'])

    # Crear gráfico de barras
    fig = px.bar(
        mesh_df,
        y='Término MeSH',
        x='Frecuencia',
        orientation='h',
        title=f'Top {top_n} Términos MeSH más Frecuentes',
        labels={'Término MeSH': 'Término MeSH', 'Frecuencia': 'Número de Artículos'},
        color='Frecuencia',
        color_continuous_scale='Blues'
    )

    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )

    return fig


def create_mesh_treemap(pubmed_data, top_n=30):
    """Treemap de términos MeSH"""
    if not pubmed_data:
        return None

    # Contar términos MeSH
    mesh_counter = Counter()
    for article in pubmed_data:
        mesh_terms = article.get('mesh_terms', [])
        mesh_counter.update(mesh_terms)

    if not mesh_counter:
        return None

    # Top N términos
    top_mesh = mesh_counter.most_common(top_n)

    # Preparar datos para treemap
    labels = ['Términos MeSH'] + [term for term, _ in top_mesh]
    parents = [''] + ['Términos MeSH'] * len(top_mesh)
    values = [0] + [count for _, count in top_mesh]

    # Calcular total
    values[0] = sum(values[1:])

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        textinfo="label+value+percent parent",
        marker=dict(colorscale='Blues', cmid=sum(values[1:]) / len(values[1:])),
        hovertemplate='<b>%{label}</b><br>Artículos: %{value}<br>%{percentParent}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Treemap de Top {top_n} Términos MeSH',
        height=600,
        margin=dict(l=10, r=10, t=50, b=10)
    )

    return fig


def create_mesh_cooccurrence(pubmed_data, top_n=15):
    """Matriz de co-ocurrencia de términos MeSH"""
    if not pubmed_data:
        return None

    # Obtener los términos MeSH más frecuentes
    mesh_counter = Counter()
    for article in pubmed_data:
        mesh_terms = article.get('mesh_terms', [])
        mesh_counter.update(mesh_terms)

    if not mesh_counter:
        return None

    # Top N términos
    top_mesh_terms = [term for term, _ in mesh_counter.most_common(top_n)]

    # Crear matriz de co-ocurrencia
    matrix = [[0 for _ in top_mesh_terms] for _ in top_mesh_terms]

    for article in pubmed_data:
        article_mesh = article.get('mesh_terms', [])
        # Solo considerar términos que están en el top N
        article_mesh_filtered = [term for term in article_mesh if term in top_mesh_terms]

        # Contar co-ocurrencias
        for i, term1 in enumerate(top_mesh_terms):
            for j, term2 in enumerate(top_mesh_terms):
                if term1 in article_mesh_filtered and term2 in article_mesh_filtered:
                    if i != j:  # No contar auto-correlación
                        matrix[i][j] += 1

    # Crear heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=top_mesh_terms,
        y=top_mesh_terms,
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
        title=f'Matriz de Co-ocurrencia de Top {top_n} Términos MeSH',
        xaxis_title='Términos MeSH',
        yaxis_title='Términos MeSH',
        height=700,
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0)
    )

    return fig


def get_top_mesh_connections(pubmed_data, top_n=5):
    """Obtiene las co-ocurrencias más fuertes entre términos MeSH"""
    if not pubmed_data:
        return []

    connections = []

    for article in pubmed_data:
        mesh_terms = article.get('mesh_terms', [])
        # Crear pares de términos
        for i, term1 in enumerate(mesh_terms):
            for term2 in mesh_terms[i+1:]:
                connections.append((term1, term2))

    # Contar co-ocurrencias
    connection_counter = Counter(connections)

    return connection_counter.most_common(top_n)


# ============================ FUNCIONES ANÁLISIS KEYWORDS ============================

def create_keywords_distribution(pubmed_data, top_n=20):
    """Distribución de keywords (palabras clave de autores)"""
    if not pubmed_data:
        return None

    # Contar keywords
    keyword_counter = Counter()
    for article in pubmed_data:
        keywords = article.get('keywords', [])
        if keywords and isinstance(keywords, list):
            keyword_counter.update(keywords)

    if not keyword_counter:
        return None

    # Top N keywords
    top_keywords = keyword_counter.most_common(top_n)
    keywords_df = pd.DataFrame(top_keywords, columns=['Keyword', 'Frecuencia'])

    # Crear gráfico de barras
    fig = px.bar(
        keywords_df,
        y='Keyword',
        x='Frecuencia',
        orientation='h',
        title=f'Top {top_n} Keywords más Frecuentes (Palabras Clave de Autores)',
        labels={'Keyword': 'Keyword', 'Frecuencia': 'Número de Artículos'},
        color='Frecuencia',
        color_continuous_scale='Greens'
    )

    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )

    return fig


def create_combined_terms_distribution(pubmed_data, top_n=25):
    """Distribución combinada de términos MeSH y keywords"""
    if not pubmed_data:
        return None

    # Contar MeSH y keywords por separado
    mesh_counter = Counter()
    keyword_counter = Counter()

    for article in pubmed_data:
        mesh_terms = article.get('mesh_terms', [])
        keywords = article.get('keywords', [])

        mesh_counter.update(mesh_terms)
        if keywords and isinstance(keywords, list):
            keyword_counter.update(keywords)

    # Combinar ambos contadores
    combined_counter = Counter()

    # Agregar MeSH terms con etiqueta
    for term, count in mesh_counter.items():
        combined_counter[term] = combined_counter.get(term, 0) + count

    # Agregar keywords con etiqueta
    for kw, count in keyword_counter.items():
        combined_counter[kw] = combined_counter.get(kw, 0) + count

    if not combined_counter:
        return None

    # Top N términos combinados
    top_combined = combined_counter.most_common(top_n)

    # Identificar fuente de cada término
    sources = []
    for term, count in top_combined:
        if term in mesh_counter and term in keyword_counter:
            sources.append('Ambos')
        elif term in mesh_counter:
            sources.append('MeSH')
        else:
            sources.append('Keywords')

    combined_df = pd.DataFrame({
        'Término': [term for term, _ in top_combined],
        'Frecuencia': [count for _, count in top_combined],
        'Fuente': sources
    })

    # Crear gráfico de barras con colores por fuente
    fig = px.bar(
        combined_df,
        y='Término',
        x='Frecuencia',
        orientation='h',
        color='Fuente',
        title=f'Top {top_n} Términos Combinados (MeSH + Keywords)',
        labels={'Término': 'Término', 'Frecuencia': 'Número de Artículos'},
        color_discrete_map={'MeSH': '#2E86AB', 'Keywords': '#A23B72', 'Ambos': '#F18F01'}
    )

    fig.update_layout(
        height=700,
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig


def create_mesh_vs_keywords_comparison(pubmed_data):
    """Comparación de cobertura entre MeSH y Keywords"""
    if not pubmed_data:
        return None

    articles_with_mesh = 0
    articles_with_keywords = 0
    articles_with_both = 0
    articles_with_none = 0

    for article in pubmed_data:
        has_mesh = bool(article.get('mesh_terms', []))
        has_keywords = bool(article.get('keywords', []))

        if has_mesh and has_keywords:
            articles_with_both += 1
        elif has_mesh:
            articles_with_mesh += 1
        elif has_keywords:
            articles_with_keywords += 1
        else:
            articles_with_none += 1

    # Crear gráfico de pastel
    labels = ['Solo MeSH', 'Solo Keywords', 'Ambos', 'Ninguno']
    values = [articles_with_mesh, articles_with_keywords, articles_with_both, articles_with_none]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#CCCCCC']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors),
        textinfo='label+value+percent',
        hovertemplate='<b>%{label}</b><br>Artículos: %{value}<br>Porcentaje: %{percent}<extra></extra>'
    )])

    fig.update_layout(
        title='Cobertura de Términos MeSH vs Keywords en Artículos',
        height=500
    )

    return fig


# ============================ FUNCIONES LÍNEAS DE INVESTIGACIÓN ============================

def create_lineas_distribution_chart(lineas_data):
    """Gráfica de distribución de artículos por línea de investigación"""
    if not lineas_data or 'estadisticas' not in lineas_data:
        return None

    stats = lineas_data['estadisticas']
    por_linea = stats.get('por_linea', {})

    # Nombres de líneas
    lineas_nombres = {
        '1': 'Línea 1: Bases Moleculares y Genómica Nutricional',
        '2': 'Línea 2: Epidemiología Clínica y Factores de Riesgo',
        '3': 'Línea 3: Salud Poblacional y Políticas Públicas'
    }

    # Preparar datos
    data = []
    for linea_num in ['1', '2', '3']:
        count = por_linea.get(linea_num, 0)
        data.append({
            'Línea': lineas_nombres[linea_num],
            'Artículos': count,
            'Porcentaje': f"{count/stats['total_articulos']*100:.1f}%"
        })

    # Crear gráfica de barras horizontal
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=[d['Línea'] for d in data],
        x=[d['Artículos'] for d in data],
        text=[f"{d['Artículos']} ({d['Porcentaje']})" for d in data],
        textposition='outside',
        orientation='h',
        marker=dict(
            color=['#1f77b4', '#ff7f0e', '#2ca02c'],
            line=dict(color='#000', width=1)
        )
    ))

    fig.update_layout(
        title='Distribución de Artículos por Línea de Investigación',
        xaxis_title='Número de Artículos (totales: líneas principales + secundarias)',
        yaxis_title='',
        height=300,
        showlegend=False,
        margin=dict(l=50, r=150, t=50, b=50)
    )

    return fig


def create_upset_plot(lineas_data):
    """UpSet Plot mostrando intersecciones de líneas de investigación"""
    if not lineas_data or 'articulos' not in lineas_data:
        return None

    # Nombres de líneas
    lineas_nombres = {
        1: 'L1: Molecular y Genómica',
        2: 'L2: Clínica y Epidemiología',
        3: 'L3: Poblacional y Políticas'
    }

    # Contar todos los artículos por línea (para barras laterales)
    totales_por_linea = {1: 0, 2: 0, 3: 0}

    # Contar intersecciones
    intersecciones = {}

    for art in lineas_data['articulos']:
        lineas = art['clasificacion'].get('lineas_principales', [])
        lineas_nums = sorted(set([l['linea'] for l in lineas]))

        # Actualizar totales
        for num in lineas_nums:
            totales_por_linea[num] += 1

        # Crear clave para la intersección
        key = tuple(lineas_nums)
        intersecciones[key] = intersecciones.get(key, 0) + 1

    # Ordenar intersecciones por tamaño (descendente)
    intersecciones_sorted = sorted(intersecciones.items(), key=lambda x: x[1], reverse=True)

    if not intersecciones_sorted:
        return None

    # Preparar datos para el plot
    from plotly.subplots import make_subplots

    # Crear subplots: barras arriba, matriz abajo
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.02,
        subplot_titles=('Tamaño de Intersecciones', 'Combinaciones de Líneas')
    )

    # Preparar datos para barras superiores (intersecciones)
    x_labels = []
    y_values = []
    hover_texts = []

    for idx, (combo, count) in enumerate(intersecciones_sorted):
        # Crear etiqueta
        if len(combo) == 1:
            label = f"Solo L{combo[0]}"
        else:
            label = " ∩ ".join([f"L{n}" for n in combo])

        x_labels.append(label)
        y_values.append(count)

        # Crear hover text detallado
        lineas_str = " + ".join([lineas_nombres[n] for n in combo])
        hover_texts.append(f"{count} artículos<br>{lineas_str}")

    # Agregar barras superiores
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=y_values,
            text=y_values,
            textposition='outside',
            marker=dict(color='#1f77b4'),
            hovertext=hover_texts,
            hoverinfo='text',
            showlegend=False
        ),
        row=1, col=1
    )

    # Preparar matriz de dots para panel inferior
    # Crear una matriz donde cada fila es una línea, cada columna es una intersección
    for linea_num in [3, 2, 1]:  # Invertido para que L1 quede arriba
        y_dots = []

        for combo, _ in intersecciones_sorted:
            if linea_num in combo:
                y_dots.append(linea_num)
            else:
                y_dots.append(None)

        # Agregar dots
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=y_dots,
                mode='markers',
                marker=dict(
                    size=12,
                    color='#2ca02c',
                    symbol='circle'
                ),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=2, col=1
        )

        # Agregar líneas verticales conectando dots en cada intersección
        for idx, (combo, _) in enumerate(intersecciones_sorted):
            if len(combo) > 1 and linea_num in combo:
                # Encontrar las otras líneas en esta combo
                lineas_en_combo = [l for l in combo if l in [1, 2, 3]]
                if len(lineas_en_combo) > 1:
                    y_min = min(lineas_en_combo)
                    y_max = max(lineas_en_combo)

                    fig.add_trace(
                        go.Scatter(
                            x=[x_labels[idx], x_labels[idx]],
                            y=[y_min, y_max],
                            mode='lines',
                            line=dict(color='#2ca02c', width=3),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=2, col=1
                    )

    # Actualizar layout
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="Artículos", row=1, col=1)

    fig.update_xaxes(title_text="", showticklabels=False, row=2, col=1)
    fig.update_yaxes(
        title_text="Líneas",
        tickmode='array',
        tickvals=[1, 2, 3],
        ticktext=['L1', 'L2', 'L3'],
        range=[0.5, 3.5],
        row=2, col=1
    )

    total_articulos = len(lineas_data['articulos'])

    fig.update_layout(
        title=f'UpSet Plot: Intersecciones de Líneas de Investigación ({total_articulos} artículos)',
        height=600,
        showlegend=False,
        font=dict(size=11)
    )

    return fig


def create_lineas_cooccurrence_matrix(lineas_data):
    """Matriz de co-ocurrencia entre líneas de investigación"""
    if not lineas_data or 'articulos' not in lineas_data:
        return None

    # Inicializar matriz 3x3
    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    lineas_nombres = ['L1: Genómica', 'L2: Salud Pública', 'L3: Alimentación']

    # Contar co-ocurrencias
    for art in lineas_data['articulos']:
        lineas_principales = [l['linea'] for l in art['clasificacion']['lineas_principales']]

        # Para cada par de líneas
        for i in range(1, 4):
            for j in range(1, 4):
                if i != j and i in lineas_principales and j in lineas_principales:
                    matrix[i-1][j-1] += 1

    # Crear heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=lineas_nombres,
        y=lineas_nombres,
        colorscale='Blues',
        text=matrix,
        texttemplate='%{text} artículos',
        textfont={"size": 12},
        hovertemplate='%{y} + %{x}<br>Co-ocurrencias: %{z}<extra></extra>',
        colorbar=dict(
            title="Artículos",
            thickness=15,
            len=0.7
        )
    ))

    fig.update_layout(
        title='Matriz de Co-ocurrencia entre Líneas',
        xaxis_title='',
        yaxis_title='',
        height=400,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )

    return fig


# ============================================================================
# NUEVAS FUNCIONES PARA ANÁLISIS ENRIQUECIDO CON METADATA DE PUBMED
# ============================================================================

def create_mesh_terms_chart(pubmed_data):
    """Gráfica de barras horizontal con los términos MeSH más frecuentes"""
    if not pubmed_data:
        return None

    # Términos demográficos a excluir
    demographic_terms = {
        'Humans', 'Female', 'Male', 'Adult', 'Middle Aged', 'Aged',
        'Animals', 'Mice', 'Rats', 'Young Adult', 'Aged, 80 and over',
        'Child', 'Adolescent', 'Infant'
    }

    # Extraer todos los MeSH terms
    all_mesh = []
    for article in pubmed_data:
        if article.get('mesh_terms'):
            for term in article['mesh_terms']:
                if term not in demographic_terms:
                    all_mesh.append(term)

    if not all_mesh:
        return None

    # Contar frecuencias
    from collections import Counter
    mesh_counts = Counter(all_mesh).most_common(30)

    # Crear dataframe
    df_mesh = pd.DataFrame(mesh_counts, columns=['Term', 'Count'])

    # Crear gráfica de barras horizontal
    fig = px.bar(
        df_mesh,
        y='Term',
        x='Count',
        orientation='h',
        title='Top 30 Términos MeSH - Vocabulario Biomédico Internacional',
        labels={'Term': 'Término MeSH', 'Count': 'Número de Artículos'},
        color='Count',
        color_continuous_scale='Blues'
    )

    fig.update_layout(
        height=800,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Número de Artículos',
        yaxis_title='',
        showlegend=False
    )

    return fig


def create_citations_metrics_and_chart(pubmed_data):
    """Crea métricas de impacto y gráfica de distribución de citaciones"""
    if not pubmed_data:
        return None, None

    # Extraer citaciones
    citations = [art.get('cited_by_count', 0) for art in pubmed_data if art.get('cited_by_count')]

    if not citations:
        return None, None

    # Calcular métricas
    total_citations = sum(citations)
    avg_citations = sum(citations) / len(citations)
    max_citations = max(citations)

    # Calcular h-index simplificado
    sorted_citations = sorted(citations, reverse=True)
    h_index = 0
    for i, cites in enumerate(sorted_citations, 1):
        if cites >= i:
            h_index = i
        else:
            break

    metrics = {
        'total': total_citations,
        'average': avg_citations,
        'max': max_citations,
        'h_index': h_index,
        'count': len(citations)
    }

    # Crear histograma
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=citations,
        nbinsx=20,
        marker_color='#1f77b4',
        hovertemplate='Citaciones: %{x}<br>Artículos: %{y}<extra></extra>'
    ))

    fig.update_layout(
        title='Distribución de Citaciones por Artículo',
        xaxis_title='Número de Citaciones',
        yaxis_title='Número de Artículos',
        height=400,
        showlegend=False
    )

    return metrics, fig


def create_top_cited_articles(pubmed_data, top_n=10):
    """Retorna los artículos más citados"""
    if not pubmed_data:
        return []

    # Filtrar artículos con citaciones
    cited_articles = [
        {
            'pmid': art.get('pmid', 'N/A'),
            'title': art.get('title', art.get('original_title', 'Sin título'))[:150],
            'year': art.get('original_year', 'N/A'),
            'citations': art.get('cited_by_count', 0),
            'journal': art.get('journal', art.get('original_journal', 'N/A'))
        }
        for art in pubmed_data
        if art.get('cited_by_count', 0) > 0
    ]

    # Ordenar por citaciones
    cited_articles.sort(key=lambda x: x['citations'], reverse=True)

    return cited_articles[:top_n]


def create_evidence_pyramid_chart(pubmed_data):
    """Crea visualización de pirámide de evidencia basada en tipos de publicación"""
    if not pubmed_data:
        return None

    # Extraer tipos de publicación
    pub_types_count = {}
    for article in pubmed_data:
        for pub_type in article.get('pub_types', []):
            pub_types_count[pub_type] = pub_types_count.get(pub_type, 0) + 1

    # Categorizar en niveles de evidencia
    evidence_levels = {
        'Meta-Análisis': pub_types_count.get('Meta-Analysis', 0),
        'Revisiones Sistemáticas': pub_types_count.get('Systematic Review', 0) + pub_types_count.get('Scoping Review', 0),
        'RCTs': pub_types_count.get('Randomized Controlled Trial', 0) + pub_types_count.get('Clinical Trial', 0),
        'Estudios Observacionales': pub_types_count.get('Observational Study', 0) + pub_types_count.get('Case-Control Studies', 0),
        'Revisiones': pub_types_count.get('Review', 0),
        'Artículos de Investigación': pub_types_count.get('Journal Article', 0)
    }

    # Filtrar solo los que tienen datos
    evidence_levels = {k: v for k, v in evidence_levels.items() if v > 0}

    # Crear gráfica de barras horizontal
    df_evidence = pd.DataFrame(list(evidence_levels.items()), columns=['Tipo', 'Cantidad'])

    # Ordenar por nivel de evidencia (de mayor a menor calidad)
    order_map = {
        'Meta-Análisis': 6,
        'Revisiones Sistemáticas': 5,
        'RCTs': 4,
        'Estudios Observacionales': 3,
        'Revisiones': 2,
        'Artículos de Investigación': 1
    }
    df_evidence['order'] = df_evidence['Tipo'].map(order_map)
    df_evidence = df_evidence.sort_values('order', ascending=False)

    fig = px.bar(
        df_evidence,
        y='Tipo',
        x='Cantidad',
        orientation='h',
        title='Pirámide de Evidencia Científica',
        labels={'Tipo': 'Nivel de Evidencia', 'Cantidad': 'Número de Artículos'},
        color='Cantidad',
        color_continuous_scale='YlOrRd'
    )

    fig.update_layout(
        height=400,
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )

    return fig


def create_funding_analysis(pubmed_data):
    """Analiza información de financiamiento"""
    if not pubmed_data:
        return None, None

    # Contar artículos con financiamiento
    funded_articles = [art for art in pubmed_data if art.get('grants') and len(art['grants']) > 0]
    funded_count = len(funded_articles)
    total_count = len(pubmed_data)

    # Extraer agencias de financiamiento
    agencies = []
    for article in funded_articles:
        for grant in article.get('grants', []):
            agency = grant.get('agency', 'Unknown')
            if agency:
                agencies.append(agency)

    # Contar frecuencias de agencias (top 10)
    from collections import Counter
    agency_counts = Counter(agencies).most_common(10)

    metrics = {
        'funded': funded_count,
        'total': total_count,
        'percentage': (funded_count / total_count * 100) if total_count > 0 else 0
    }

    if not agency_counts:
        return metrics, None

    # Crear gráfica de agencias
    df_agencies = pd.DataFrame(agency_counts, columns=['Agencia', 'Artículos'])

    fig = px.bar(
        df_agencies,
        x='Agencia',
        y='Artículos',
        title='Top 10 Agencias Financiadoras',
        labels={'Agencia': 'Agencia de Financiamiento', 'Artículos': 'Número de Artículos'},
        color='Artículos',
        color_continuous_scale='Greens'
    )

    fig.update_layout(
        height=400,
        xaxis_tickangle=-45,
        showlegend=False
    )

    return metrics, fig


def create_collaboration_network_data(pubmed_data):
    """Extrae datos de red de colaboración institucional"""
    if not pubmed_data:
        return None

    # Extraer instituciones de las afiliaciones
    institutions = []

    for article in pubmed_data:
        article_institutions = set()

        for affiliation in article.get('affiliations', []):
            # Extraer nombre de institución (simplificado)
            if affiliation:
                # Buscar palabras clave de instituciones
                if 'Universidad de Guadalajara' in affiliation or 'University of Guadalajara' in affiliation:
                    article_institutions.add('Universidad de Guadalajara')
                elif 'IMSS' in affiliation or 'Mexican Social Security' in affiliation:
                    article_institutions.add('IMSS')
                elif 'UNAM' in affiliation:
                    article_institutions.add('UNAM')
                elif 'IPN' in affiliation or 'Politécnico Nacional' in affiliation:
                    article_institutions.add('IPN')
                elif 'CIATEJ' in affiliation:
                    article_institutions.add('CIATEJ')
                elif 'Universidad Autónoma' in affiliation:
                    article_institutions.add('Universidades Autónomas')
                elif 'Hospital' in affiliation:
                    article_institutions.add('Hospitales')
                # Agregar más instituciones según sea necesario

        if len(article_institutions) > 0:
            institutions.append(list(article_institutions))

    # Contar colaboraciones
    from collections import Counter
    institution_counts = Counter()

    for inst_list in institutions:
        institution_counts.update(inst_list)

    # Top instituciones
    top_institutions = institution_counts.most_common(10)

    return top_institutions


def create_collaboration_map(pubmed_data):
    """Crea mapa mundial de colaboraciones internacionales basado en afiliaciones"""
    if not pubmed_data:
        return None

    # Diccionario de países a buscar en afiliaciones (con códigos ISO)
    country_keywords = {
        'Mexico': {'keywords': ['Mexico', 'México', 'Guadalajara', 'Jalisco', 'CDMX'], 'code': 'MEX'},
        'USA': {'keywords': ['USA', 'United States', 'U.S.A', 'California', 'Texas', 'Florida'], 'code': 'USA'},
        'Canada': {'keywords': ['Canada', 'Toronto', 'Vancouver', 'Montreal'], 'code': 'CAN'},
        'Spain': {'keywords': ['Spain', 'España', 'Madrid', 'Barcelona', 'Sevilla'], 'code': 'ESP'},
        'Brazil': {'keywords': ['Brazil', 'Brasil', 'São Paulo', 'Rio de Janeiro'], 'code': 'BRA'},
        'Argentina': {'keywords': ['Argentina', 'Buenos Aires'], 'code': 'ARG'},
        'Colombia': {'keywords': ['Colombia', 'Bogotá', 'Medellín'], 'code': 'COL'},
        'Chile': {'keywords': ['Chile', 'Santiago'], 'code': 'CHL'},
        'Peru': {'keywords': ['Peru', 'Perú', 'Lima'], 'code': 'PER'},
        'UK': {'keywords': ['United Kingdom', 'England', 'London', 'Scotland', 'UK'], 'code': 'GBR'},
        'Germany': {'keywords': ['Germany', 'Alemania', 'Berlin', 'Munich'], 'code': 'DEU'},
        'France': {'keywords': ['France', 'Francia', 'Paris'], 'code': 'FRA'},
        'Italy': {'keywords': ['Italy', 'Italia', 'Rome', 'Milan'], 'code': 'ITA'},
        'Netherlands': {'keywords': ['Netherlands', 'Amsterdam', 'Países Bajos'], 'code': 'NLD'},
        'Sweden': {'keywords': ['Sweden', 'Suecia', 'Stockholm'], 'code': 'SWE'},
        'Switzerland': {'keywords': ['Switzerland', 'Suiza', 'Geneva', 'Zürich'], 'code': 'CHE'},
        'China': {'keywords': ['China', 'Beijing', 'Shanghai'], 'code': 'CHN'},
        'Japan': {'keywords': ['Japan', 'Japón', 'Tokyo'], 'code': 'JPN'},
        'South Korea': {'keywords': ['South Korea', 'Korea', 'Seoul'], 'code': 'KOR'},
        'India': {'keywords': ['India', 'New Delhi', 'Mumbai'], 'code': 'IND'},
        'Australia': {'keywords': ['Australia', 'Sydney', 'Melbourne'], 'code': 'AUS'},
    }

    # Contar colaboraciones por país
    country_counts = {}

    for article in pubmed_data:
        countries_in_article = set()

        for affiliation in article.get('affiliations', []):
            if affiliation:
                for country, data in country_keywords.items():
                    for keyword in data['keywords']:
                        if keyword in affiliation:
                            countries_in_article.add(country)
                            break

        # Contar cada país presente en el artículo
        for country in countries_in_article:
            country_counts[country] = country_counts.get(country, 0) + 1

    if not country_counts:
        return None

    # Excluir México del mapa (es la sede del DCNT)
    if 'Mexico' in country_counts:
        del country_counts['Mexico']

    if not country_counts:
        return None

    # Preparar datos para el mapa
    countries = []
    codes = []
    counts = []

    for country, count in country_counts.items():
        countries.append(country)
        codes.append(country_keywords[country]['code'])
        counts.append(count)

    # Crear DataFrame
    df_map = pd.DataFrame({
        'Country': countries,
        'Code': codes,
        'Articles': counts
    })

    # Crear mapa choropleth
    fig = go.Figure(data=go.Choropleth(
        locations=df_map['Code'],
        z=df_map['Articles'],
        text=df_map['Country'],
        colorscale='Blues',
        autocolorscale=False,
        reversescale=False,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title="Artículos",
        hovertemplate='<b>%{text}</b><br>Artículos: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title_text='Mapa de Colaboración Internacional del DCNT-UdeG (excluye México - sede)',
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        ),
        height=500
    )

    return fig, df_map


def filter_articulos_by_linea(lineas_data, linea_num):
    """Filtra artículos que pertenecen a una línea específica (principal o secundaria)"""
    if not lineas_data or 'articulos' not in lineas_data:
        return pd.DataFrame()

    articulos_filtrados = []

    for art in lineas_data['articulos']:
        # En el nuevo formato, todas las líneas están en 'lineas_principales'
        # La primera es la principal, las demás con confianza='secundaria' son secundarias
        lineas = art['clasificacion'].get('lineas_principales', [])

        # Verificar si el artículo pertenece a la línea
        linea_info = None
        tipo_clasificacion = None

        for idx, l in enumerate(lineas):
            if l['linea'] == linea_num:
                linea_info = l
                # La primera línea es siempre principal, las demás son secundarias
                tipo_clasificacion = 'Principal' if idx == 0 else 'Secundaria'
                break

        if linea_info:
            # Obtener similitud (nuevo formato) o score_ml (formato antiguo)
            score = linea_info.get('similitud', linea_info.get('score_ml', linea_info.get('score_final', 0)))
            # Convertir similitud (0-1) a porcentaje si es necesario
            if score < 1.0:
                score = score * 100

            articulos_filtrados.append({
                'PMID': art['pmid'],
                'Año': art['año'],
                'Título': art['titulo'][:100] + '...' if len(art['titulo']) > 100 else art['titulo'],
                'Score ML': round(score, 1),
                'Confianza': linea_info['confianza'].capitalize(),
                'Tipo': tipo_clasificacion
            })

    return pd.DataFrame(articulos_filtrados)


# Interfaz principal
def main():
    # Encabezado con logo
    logo_path = Path(__file__).parent.parent / "assets" / "logo_dcnt.png"

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
    publications_df, ods_data, pronaces_data, themes_data, lineas_data = load_data()
    pubmed_data = load_pubmed_metadata()

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

    *Fuente: Surkis A, et al. (2016). "Classifying publications from the clinical and translational science award program along the translational research spectrum: a machine learning approach". Journal of Translational Medicine, 14:235. [PMID: 27492440](https://pubmed.ncbi.nlm.nih.gov/27492440/) | DOI: [10.1186/s12967-016-0992-8](https://doi.org/10.1186/s12967-016-0992-8)*
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
        # Obtener estadísticas
        ods_stats = get_ods_stats(ods_data)
        total_ods_articles = len(ods_data)

        # Resumen visual en métricas
        st.markdown("### 📊 Alineación con Agenda 2030")

        # Primera fila de ODS principales
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            count_2 = ods_stats.get(2, 0)
            st.metric("🌾 ODS 2: Hambre Cero", f"{count_2} artículos", f"{count_2/total_ods_articles*100:.1f}%")

        with col2:
            count_3 = ods_stats.get(3, 0)
            st.metric("❤️ ODS 3: Salud y Bienestar", f"{count_3} artículos", f"{count_3/total_ods_articles*100:.1f}%")

        with col3:
            count_10 = ods_stats.get(10, 0)
            st.metric("⚖️ ODS 10: Reducir Desigualdades", f"{count_10} artículos", f"{count_10/total_ods_articles*100:.1f}%")

        with col4:
            count_12 = ods_stats.get(12, 0)
            st.metric("♻️ ODS 12: Producción y Consumo", f"{count_12} artículos", f"{count_12/total_ods_articles*100:.1f}%")

        # Segunda fila de ODS adicionales - ahora con 3 columnas
        col5, col6, col7 = st.columns(3)

        with col5:
            count_1 = ods_stats.get(1, 0)
            st.metric("🏚️ ODS 1: Fin de la Pobreza", f"{count_1} artículos", f"{count_1/total_ods_articles*100:.1f}%" if count_1 > 0 else "0%")

        with col6:
            count_5 = ods_stats.get(5, 0)
            st.metric("👥 ODS 5: Igualdad de Género", f"{count_5} artículos", f"{count_5/total_ods_articles*100:.1f}%" if count_5 > 0 else "0%")

        with col7:
            # Calcular el total de ODS abordados
            num_ods = len([v for v in ods_stats.values() if v > 0])
            st.metric("🎯 Total ODS Abordados", f"{num_ods} ODS", f"{num_ods/17*100:.0f}% de la Agenda")

        st.markdown("---")

        # Nota explicativa sobre la metodología
        with st.expander("ℹ️ Metodología de Clasificación de ODS", expanded=False):
            st.markdown("""
            ### 🤖 Clasificación Automática con Embeddings

            Los artículos fueron clasificados automáticamente usando **sentence-transformers** con el modelo `all-MiniLM-L6-v2`.

            **Metadata utilizada para clasificación:**
            - ✅ **Título completo** del artículo
            - ✅ **Abstract** (resumen científico completo)
            - ✅ **Términos MeSH** (vocabulario controlado de PubMed)
            - ✅ **Keywords** (palabras clave de autores)

            **Proceso:**
            1. Se generan embeddings (representaciones vectoriales) de cada artículo usando toda su metadata
            2. Se generan embeddings de las descripciones detalladas de cada ODS
            3. Se calcula la **similitud de coseno** entre cada artículo y cada ODS
            4. Se asignan ODS principales (similitud ≥ 0.45) y secundarios (similitud ≥ 0.35)

            **ODS clasificados:** 7 ODS relevantes para investigación en nutrición traslacional (ODS 1, 2, 3, 5, 10, 12, 13)
            """)

        st.markdown("---")

        # Gráfica de distribución
        col_graph, col_info = st.columns([1, 1])

        with col_graph:
            fig_ods = create_ods_distribution(ods_data)
            if fig_ods:
                st.plotly_chart(fig_ods, use_container_width=True)

        with col_info:
            st.info("""
            **Meta 2030 de la Agenda Global:**

            Los 17 Objetivos de Desarrollo Sostenible (ODS) son el plan maestro de la ONU para un futuro sostenible.

            El DCNT-UdeG contribuye directamente a múltiples ODS prioritarios, abordando desafíos interconectados de nutrición, salud, educación, equidad y desarrollo sostenible.
            """)

        st.markdown("---")

        # Explorador interactivo de artículos por ODS
        st.markdown("### 📚 Explorador de Artículos por ODS")

        st.markdown("""
        Selecciona un ODS para ver todos los artículos clasificados (principales y secundarios).
        """)

        # Selector de ODS
        ods_options = {
            1: "ODS 1: Fin de la Pobreza",
            2: "ODS 2: Hambre Cero",
            3: "ODS 3: Salud y Bienestar",
            5: "ODS 5: Igualdad de Género",
            10: "ODS 10: Reducción de Desigualdades",
            12: "ODS 12: Producción y Consumo Responsables",
            13: "ODS 13: Acción por el Clima"
        }

        selected_ods = st.selectbox(
            "Selecciona un ODS:",
            options=list(ods_options.keys()),
            format_func=lambda x: ods_options[x],
            index=2  # ODS 3 por defecto (el más frecuente)
        )

        if selected_ods:
            # Filtrar artículos del ODS seleccionado
            articulos_principales = []
            articulos_secundarios = []

            for articulo in ods_data:
                # Verificar ODS principales
                for ods_p in articulo.get('ods_principales', []):
                    if ods_p.get('numero') == selected_ods:
                        articulos_principales.append({
                            'título': articulo.get('titulo', ''),
                            'año': articulo.get('año', 0),
                            'revista': articulo.get('revista', ''),
                            'doi': articulo.get('doi', ''),
                            'similitud': ods_p.get('similitud', 0),
                            'confianza': ods_p.get('confianza', ''),
                            'tipo': 'Principal'
                        })
                        break

                # Verificar ODS secundarios
                for ods_s in articulo.get('ods_secundarios', []):
                    if ods_s.get('numero') == selected_ods:
                        articulos_secundarios.append({
                            'título': articulo.get('titulo', ''),
                            'año': articulo.get('año', 0),
                            'revista': articulo.get('revista', ''),
                            'doi': articulo.get('doi', ''),
                            'similitud': ods_s.get('similitud', 0),
                            'confianza': ods_s.get('confianza', ''),
                            'tipo': 'Secundario'
                        })
                        break

            total_articulos = len(articulos_principales) + len(articulos_secundarios)

            # Mostrar estadísticas
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("📊 Total de artículos", total_articulos)
            with col_stat2:
                st.metric("🎯 ODS Principal", len(articulos_principales))
            with col_stat3:
                st.metric("🔗 ODS Secundario", len(articulos_secundarios))

            # Mostrar artículos
            if total_articulos > 0:
                st.markdown(f"#### Artículos clasificados en {ods_options[selected_ods]}")

                # Combinar y crear DataFrame
                todos_articulos = articulos_principales + articulos_secundarios
                df_ods_articulos = pd.DataFrame(todos_articulos)

                # Ordenar por tipo (principales primero) y luego por similitud
                df_ods_articulos['tipo_orden'] = df_ods_articulos['tipo'].map({'Principal': 0, 'Secundario': 1})
                df_ods_articulos = df_ods_articulos.sort_values(['tipo_orden', 'similitud'], ascending=[True, False])
                df_ods_articulos = df_ods_articulos.drop('tipo_orden', axis=1)

                # Renombrar columnas para mejor visualización
                df_ods_articulos = df_ods_articulos.rename(columns={
                    'título': 'Título',
                    'año': 'Año',
                    'revista': 'Revista',
                    'doi': 'DOI',
                    'similitud': 'Similitud',
                    'confianza': 'Confianza',
                    'tipo': 'Clasificación'
                })

                # Mostrar tabla
                st.dataframe(
                    df_ods_articulos,
                    use_container_width=True,
                    height=400,
                    column_config={
                        "Similitud": st.column_config.ProgressColumn(
                            "Similitud",
                            help="Similitud con el ODS (0-1)",
                            format="%.3f",
                            min_value=0,
                            max_value=1,
                        ),
                        "Confianza": st.column_config.TextColumn(
                            "Confianza",
                            help="Nivel de confianza de la clasificación",
                        ),
                        "Clasificación": st.column_config.TextColumn(
                            "Clasificación",
                            help="Principal o Secundario",
                        )
                    }
                )

                # Botón de descarga
                csv = df_ods_articulos.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"⬇️ Descargar artículos de {ods_options[selected_ods]} (CSV)",
                    data=csv,
                    file_name=f"articulos_ods_{selected_ods}.csv",
                    mime="text/csv",
                )

                # Distribución de confianza
                st.markdown("##### Distribución de Confianza")
                confianza_counts = df_ods_articulos['Confianza'].value_counts()

                col_conf1, col_conf2, col_conf3, col_conf4 = st.columns(4)
                with col_conf1:
                    st.metric("🟢 Alta", confianza_counts.get('alta', 0))
                with col_conf2:
                    st.metric("🟡 Media", confianza_counts.get('media', 0))
                with col_conf3:
                    st.metric("🟠 Baja", confianza_counts.get('baja', 0))
                with col_conf4:
                    st.metric("🔴 Tentativa", confianza_counts.get('tentativa', 0))

            else:
                st.info(f"No hay artículos clasificados en {ods_options[selected_ods]}")


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
        # Calcular estadísticas de PRONACES
        pronace_counts = {
            "SALUD": 0,
            "SOBERANIA_ALIMENTARIA": 0,
            "SISTEMAS_ALIMENTARIOS": 0
        }

        for article in pronaces_data:
            for pron in article.get('pronaces_principales', []):
                codigo = pron.get('codigo', '')
                if codigo in pronace_counts:
                    pronace_counts[codigo] += 1

        total_clasificados = sum(pronace_counts.values())

        # Layout mejorado
        st.markdown("### 📊 Distribución de Publicaciones por PRONACE")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Total Clasificado",
                total_clasificados,
                delta=f"{(total_clasificados/len(publications_df)*100):.1f}% del total"
            )
        with col2:
            st.metric(
                "🏥 Salud",
                pronace_counts["SALUD"],
                delta=f"{(pronace_counts['SALUD']/total_clasificados*100):.1f}%"
            )
        with col3:
            st.metric(
                "🌾 Soberanía Alimentaria",
                pronace_counts["SOBERANIA_ALIMENTARIA"],
                delta=f"{(pronace_counts['SOBERANIA_ALIMENTARIA']/total_clasificados*100):.1f}%"
            )
        with col4:
            st.metric(
                "♻️ Sistemas Alimentarios",
                pronace_counts["SISTEMAS_ALIMENTARIOS"],
                delta=f"{(pronace_counts['SISTEMAS_ALIMENTARIOS']/total_clasificados*100):.1f}%"
            )

        st.markdown("---")

        # Gráfico de distribución
        col_left, col_right = st.columns([1, 1])

        with col_left:
            # Gráfico de barras horizontales
            pronace_names = {
                "SALUD": "🏥 PRONACE Salud",
                "SOBERANIA_ALIMENTARIA": "🌾 PRONACE Soberanía Alimentaria",
                "SISTEMAS_ALIMENTARIOS": "♻️ Sistemas Alimentarios Sostenibles"
            }

            fig_dist = go.Figure(data=[
                go.Bar(
                    y=[pronace_names[k] for k in pronace_counts.keys()],
                    x=list(pronace_counts.values()),
                    orientation='h',
                    text=list(pronace_counts.values()),
                    textposition='auto',
                    marker=dict(
                        color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                        line=dict(color='white', width=2)
                    )
                )
            ])

            fig_dist.update_layout(
                title='Publicaciones del DCNT por PRONACE',
                xaxis_title='Número de Publicaciones',
                yaxis_title='',
                height=300,
                showlegend=False
            )

            st.plotly_chart(fig_dist, use_container_width=True)

        with col_right:
            # Heatmap temporal
            fig_pronaces = create_pronaces_heatmap(pronaces_data, filtered_df)
            if fig_pronaces:
                st.plotly_chart(fig_pronaces, use_container_width=True)

        st.markdown("---")

        # Sección mejorada: Relevancia para el DCNT
        st.markdown("### 🎯 Relevancia de los PRONACES para el DCNT-UdeG")

        # PRONACE SALUD
        with st.expander("🏥 PRONACE SALUD - Cobertura Principal del DCNT", expanded=True):
            col_a, col_b = st.columns([1, 1])

            with col_a:
                st.markdown(f"""
                **📊 Alineación del DCNT:**
                - **{pronace_counts['SALUD']} publicaciones** ({(pronace_counts['SALUD']/total_clasificados*100):.1f}% del total)
                - Principal área de contribución del programa
                - Aborda crisis nacional de ENT

                **🎯 Áreas Prioritarias PRONACE:**
                - Enfermedades Crónicas no Transmisibles
                - Alimentación y Salud Integral Comunitaria
                - Medicina de Sistemas y Determinantes Moleculares
                - Ciencia de Datos Aplicada a Salud
                """)

            with col_b:
                st.markdown(f"""
                **💡 Contribución Específica del DCNT:**

                **Línea 1 (Genómica Nutricional):**
                - Medicina de sistemas y biomarcadores
                - Determinantes moleculares de ENT
                - Nutrigenética y nutrigenómica

                **Línea 2 (Salud Pública):**
                - Intervenciones comunitarias escalables
                - Evaluación de políticas de salud
                - Investigación traslacional poblacional

                **Línea 3 (Alimentación y Nutrición):**
                - Terapia nutricional en ENT
                - Patrones alimentarios saludables
                - Prevención de obesidad y diabetes
                """)

            st.info(f"""
            💰 **Financiamiento:** {PRONACES_CONTEXTO['PRONACE_SALUD']['financiamiento']}

            🔬 **Competencia del DCNT:** Los graduados están preparados para liderar proyectos PRONAII en
            prevención y manejo de ENT, con expertise en genómica nutricional, intervenciones poblacionales
            y medicina de precisión aplicada a nutrición.
            """)

        # PRONACE SOBERANÍA ALIMENTARIA
        with st.expander("🌾 PRONACE SOBERANÍA ALIMENTARIA - Segunda Área de Impacto"):
            col_a, col_b = st.columns([1, 1])

            with col_a:
                st.markdown(f"""
                **📊 Alineación del DCNT:**
                - **{pronace_counts['SOBERANIA_ALIMENTARIA']} publicaciones** ({(pronace_counts['SOBERANIA_ALIMENTARIA']/total_clasificados*100):.1f}% del total)
                - Enfoque en malnutrición y seguridad alimentaria
                - Alimentos funcionales y tradicionales

                **🎯 Demandas Prioritarias PRONACE:**
                - Alimentación saludable y culturalmente adecuada
                - Alimentos funcionales
                - Calidad nutrimental maíz-tortilla
                - Educación para alimentación saludable
                - Circuitos regionales de alimentos
                """)

            with col_b:
                st.markdown(f"""
                **💡 Contribución Específica del DCNT:**

                **Línea 2 (Salud Pública):**
                - Educación nutricional comunitaria
                - Evaluación de programas alimentarios
                - Intervenciones en poblaciones vulnerables

                **Línea 3 (Alimentación y Nutrición):**
                - Ciencias de alimentos y alimentos funcionales
                - Calidad nutrimental de alimentos tradicionales
                - Desarrollo de productos nutricionales
                - Sistemas alimentarios locales
                """)

            st.success(f"""
            🌱 **Alcance Nacional:** {PRONACES_CONTEXTO['PRONACE_SOBERANIA_ALIMENTARIA']['pronaii_activos']} PRONAII activos en
            {PRONACES_CONTEXTO['PRONACE_SOBERANIA_ALIMENTARIA']['localidades']} localidades con
            {PRONACES_CONTEXTO['PRONACE_SOBERANIA_ALIMENTARIA']['organizaciones_comunitarias']} organizaciones comunitarias.

            🎓 **Formación DCNT:** Los estudiantes desarrollan competencias en investigación participativa,
            valoración de alimentos tradicionales, y diseño de intervenciones nutricionales culturalmente apropiadas.
            """)

        # SISTEMAS ALIMENTARIOS
        with st.expander("♻️ SISTEMAS ALIMENTARIOS SOSTENIBLES - Área Emergente"):
            col_a, col_b = st.columns([1, 1])

            with col_a:
                st.markdown(f"""
                **📊 Alineación del DCNT:**
                - **{pronace_counts['SISTEMAS_ALIMENTARIOS']} publicaciones** ({(pronace_counts['SISTEMAS_ALIMENTARIOS']/total_clasificados*100):.1f}% del total)
                - Consumo responsable y sostenibilidad
                - Impacto de alimentos ultraprocesados

                **🎯 Temas de Investigación:**
                - Alimentos ultraprocesados y bebidas azucaradas
                - Patrones dietéticos sostenibles
                - Etiquetado frontal de alimentos
                - Ambientes alimentarios saludables
                - Transición nutricional
                """)

            with col_b:
                st.markdown(f"""
                **💡 Contribución Específica del DCNT:**

                **Enfoque Transdisciplinario:**
                - Análisis de sistemas alimentarios complejos
                - Impacto ambiental de patrones dietéticos
                - Políticas públicas de alimentación
                - Innovación en alimentos funcionales

                **Investigación Traslacional:**
                - Evaluación de intervenciones de etiquetado
                - Estudios de consumo alimentario
                - Análisis de ambientes obesogénicos
                """)

            st.warning("""
            ⚠️ **Área en Crecimiento:** Aunque actualmente representa {:.1f}% de las publicaciones,
            es un área estratégica emergente que vincula salud pública, sostenibilidad ambiental y
            políticas alimentarias - competencias clave para el futuro de la nutrición en México.
            """.format(pronace_counts['SISTEMAS_ALIMENTARIOS']/total_clasificados*100))

        st.markdown("---")

        # Nota de metodología
        with st.expander("ℹ️ Metodología de Clasificación", expanded=False):
            st.markdown("""
            **Clasificación Automática con Embeddings:**

            Los artículos del DCNT fueron clasificados en PRONACES utilizando **embeddings semánticos**
            (modelo all-MiniLM-L6-v2) con similitud de coseno.

            **Proceso:**
            1. Se generan embeddings (representaciones vectoriales) de cada artículo usando toda su metadata
            2. Se generan embeddings de las descripciones detalladas de cada PRONACE
            3. Se calcula la **similitud de coseno** entre cada artículo y cada PRONACE
            4. Se asignan PRONACES principales (similitud ≥ 0.40) y secundarios (similitud ≥ 0.30)

            **PRONACES clasificados:** 3 programas más relevantes para investigación en nutrición traslacional:
            - PRONACE Salud
            - PRONACE Soberanía Alimentaria
            - Sistemas Alimentarios Sostenibles
            """)

        st.markdown("---")

        # Explorador interactivo de PRONACES
        st.markdown("### 🔍 Explorador de Artículos por PRONACE")

        # Preparar opciones
        pronace_options = {
            "SALUD": "🏥 PRONACE Salud",
            "SOBERANIA_ALIMENTARIA": "🌾 PRONACE Soberanía Alimentaria",
            "SISTEMAS_ALIMENTARIOS": "♻️ Sistemas Alimentarios Sostenibles"
        }

        selected_pronace = st.selectbox(
            "Selecciona un PRONACE:",
            options=list(pronace_options.keys()),
            format_func=lambda x: pronace_options[x],
            index=0  # SALUD por defecto
        )

        # Filtrar artículos del PRONACE seleccionado
        pronace_articles = []
        for article in pronaces_data:
            # Verificar si está en principales
            for pron in article.get('pronaces_principales', []):
                if pron['codigo'] == selected_pronace:
                    pronace_articles.append({
                        'tipo': 'Principal',
                        **article
                    })
                    break
            else:
                # Verificar si está en secundarios
                for pron in article.get('pronaces_secundarios', []):
                    if pron['codigo'] == selected_pronace:
                        pronace_articles.append({
                            'tipo': 'Secundario',
                            **article
                        })
                        break

        # Mostrar métricas
        st.markdown(f"#### Artículos clasificados en {pronace_options[selected_pronace]}")

        principales_count = sum(1 for a in pronace_articles if a.get('tipo') == 'Principal')
        secundarios_count = sum(1 for a in pronace_articles if a.get('tipo') == 'Secundario')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", len(pronace_articles))
        with col2:
            st.metric("Principales", principales_count)
        with col3:
            st.metric("Secundarios", secundarios_count)

        if pronace_articles:
            # Preparar dataframe
            df_pronace = pd.DataFrame([
                {
                    'Año': art['año'],
                    'Título': art['titulo'],
                    'Revista': art['revista'],
                    'Clasificación': art['tipo'],
                    'Similitud': next((p['similitud'] for p in art.get('pronaces_principales', []) if p['codigo'] == selected_pronace),
                                    next((p['similitud'] for p in art.get('pronaces_secundarios', []) if p['codigo'] == selected_pronace), 0)),
                    'Confianza': next((p['confianza'] for p in art.get('pronaces_principales', []) if p['codigo'] == selected_pronace),
                                    next((p['confianza'] for p in art.get('pronaces_secundarios', []) if p['codigo'] == selected_pronace), '')),
                    'DOI': art.get('doi', ''),
                    'PMID': art.get('pmid', '')
                }
                for art in pronace_articles
            ])

            # Ordenar por similitud
            df_pronace = df_pronace.sort_values('Similitud', ascending=False)

            st.dataframe(
                df_pronace,
                use_container_width=True,
                height=400
            )

            # Botón de descarga
            csv = df_pronace.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"⬇️ Descargar artículos de {pronace_options[selected_pronace]} (CSV)",
                data=csv,
                file_name=f'pronace_{selected_pronace.lower()}_articulos.csv',
                mime='text/csv'
            )

        else:
            st.info(f"No hay artículos clasificados en {pronace_options[selected_pronace]}")

        st.markdown("---")

        st.success("""
        ✅ **Alta Pertinencia Demostrada:** El DCNT-UdeG forma recursos humanos especializados
        para liderar o participar en futuros PRONAII, con competencias específicas en investigación
        traslacional, vinculación con gobierno/comunidad, y trabajo multidisciplinario.
        """)

    else:
        st.warning("⚠️ Ejecuta primero el script de clasificación para generar los datos de PRONACES")

    # SECCIÓN 4: Análisis Temático con Términos MeSH y Keywords
    st.markdown('<div class="section-header">🔬 Análisis Temático de la Investigación (MeSH + Keywords)</div>', unsafe_allow_html=True)

    if pubmed_data:
        st.markdown("""
        Análisis temático enriquecido combinando:
        - **Términos MeSH**: Vocabulario controlado asignado por expertos del NLM (National Library of Medicine)
        - **Keywords**: Palabras clave específicas proporcionadas por los autores de cada artículo

        Esta combinación proporciona una visión más completa: los términos MeSH garantizan estandarización,
        mientras que las keywords capturan la especificidad y terminología actual de cada investigación.
        """)

        # Estadísticas generales
        total_articles = len(pubmed_data)
        articles_with_mesh = sum(1 for a in pubmed_data if a.get('mesh_terms', []))
        articles_with_keywords = sum(1 for a in pubmed_data if a.get('keywords', []))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📚 Total de Artículos", total_articles)
        with col2:
            st.metric("🏷️ Con Términos MeSH", f"{articles_with_mesh} ({articles_with_mesh/total_articles*100:.1f}%)")
        with col3:
            st.metric("🔑 Con Keywords", f"{articles_with_keywords} ({articles_with_keywords/total_articles*100:.1f}%)")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Términos Combinados",
            "📊 MeSH",
            "🔑 Keywords",
            "⚖️ Comparación",
            "🔥 Co-ocurrencia"
        ])

        with tab1:
            st.markdown("### Términos Más Frecuentes (MeSH + Keywords)")
            st.info("""
            Esta visualización combina términos MeSH y keywords, mostrando la fuente de cada término con colores:
            - 🔵 **Azul**: Solo MeSH (estandarizado)
            - 🔴 **Rosa**: Solo Keywords (específico de autores)
            - 🟠 **Naranja**: Aparece en ambos (validación cruzada)
            """)
            fig_combined = create_combined_terms_distribution(pubmed_data, top_n=30)
            if fig_combined:
                st.plotly_chart(fig_combined, use_container_width=True)
                st.caption("""
                💡 **Interpretación:** Los términos en naranja (aparecen en ambas fuentes) indican temas
                bien establecidos y ampliamente reconocidos. Los términos únicos de keywords revelan
                terminología emergente o específica del campo.
                """)

        with tab2:
            st.markdown("### Distribución de Términos MeSH")
            st.info("""
            **Términos MeSH (Medical Subject Headings)**:
            - Vocabulario controlado de biomedicina
            - Asignados por indexadores expertos de PubMed
            - Garantizan consistencia y comparabilidad internacional
            """)
            fig_mesh_dist = create_mesh_distribution(pubmed_data, top_n=25)
            if fig_mesh_dist:
                st.plotly_chart(fig_mesh_dist, use_container_width=True)
                st.caption("""
                💡 **Interpretación:** Términos estandarizados que permiten comparar la investigación
                del DCNT con otros programas a nivel internacional.
                """)

        with tab3:
            st.markdown("### Distribución de Keywords (Palabras Clave de Autores)")
            st.info("""
            **Keywords de Autores**:
            - Términos específicos elegidos por los investigadores
            - Reflejan la terminología actual del campo
            - Capturan conceptos emergentes y especificidad metodológica
            """)
            fig_keywords_dist = create_keywords_distribution(pubmed_data, top_n=25)
            if fig_keywords_dist:
                st.plotly_chart(fig_keywords_dist, use_container_width=True)
                st.caption("""
                💡 **Interpretación:** Las keywords revelan los términos específicos que los investigadores
                del DCNT consideran más representativos de su trabajo, incluyendo terminología técnica
                y conceptos emergentes no siempre capturados por MeSH.
                """)

        with tab4:
            st.markdown("### Cobertura: MeSH vs Keywords")
            st.info("""
            Comparación de la cobertura de términos en los artículos del DCNT.
            Idealmente, los artículos deben tener tanto términos MeSH (estandarización) como keywords (especificidad).
            """)
            fig_comparison = create_mesh_vs_keywords_comparison(pubmed_data)
            if fig_comparison:
                st.plotly_chart(fig_comparison, use_container_width=True)
                st.caption("""
                💡 **Interpretación:** Los artículos con ambos tipos de términos tienen la mejor
                visibilidad y permiten análisis más completos. Los artículos sin términos pueden
                requerir actualización de metadata.
                """)

        with tab5:
            st.markdown("### Matriz de Co-ocurrencia de Términos MeSH")
            fig_mesh_cooccurrence = create_mesh_cooccurrence(pubmed_data, top_n=15)
            if fig_mesh_cooccurrence:
                st.plotly_chart(fig_mesh_cooccurrence, use_container_width=True)
                st.caption("""
                💡 **Interpretación:** Esta matriz muestra cuántas veces dos términos MeSH aparecen juntos.
                Valores altos revelan las intersecciones temáticas más frecuentes en la investigación del DCNT.
                """)

                # Análisis adicional de conexiones fuertes
                st.markdown("#### 🔗 Conexiones Interdisciplinarias Destacadas")

                top_mesh_connections = get_top_mesh_connections(pubmed_data, top_n=10)

                if top_mesh_connections:
                    st.markdown("**Top 10 combinaciones de términos MeSH más frecuentes:**")
                    for (term1, term2), count in top_mesh_connections:
                        st.write(f"- **{term1}** ↔ **{term2}**: {count} publicaciones")

                    st.info("""
                    Estas conexiones muestran el **enfoque interdisciplinario** del programa, donde se integran
                    múltiples áreas de conocimiento para abordar problemas complejos de nutrición y salud.
                    """)
    else:
        st.warning("⚠️ No hay datos de PubMed disponibles para análisis temático")

    # ========================================================================
    # NUEVAS SUBSECCIONES: ANÁLISIS ENRIQUECIDO CON METADATA DE PUBMED
    # ========================================================================

    if pubmed_data:
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Análisis Enriquecido con Metadata de PubMed</div>', unsafe_allow_html=True)

        st.markdown("""
        Análisis enriquecido con metadata completa de **PubMed/MEDLINE**, incluyendo términos MeSH,
        citaciones, tipos de evidencia, financiamiento y colaboraciones internacionales.
        """)

        # Crear tabs para organizar las visualizaciones
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏷️ MeSH Terms",
            "📈 Impacto",
            "🔬 Evidencia",
            "💰 Financiamiento",
            "🤝 Colaboración",
            "🌍 Mapa Mundial"
        ])

        # TAB 1: Términos MeSH
        with tab1:
            st.markdown("### Vocabulario Biomédico Internacional")
            st.markdown("""
            Los **términos MeSH** (Medical Subject Headings) son el vocabulario controlado de la
            **Biblioteca Nacional de Medicina de EE.UU.** (NLM/NIH), permitiendo comparabilidad internacional.
            """)

            col1, col2 = st.columns([2, 1])

            with col1:
                fig_mesh = create_mesh_terms_chart(pubmed_data)
                if fig_mesh:
                    st.plotly_chart(fig_mesh, use_container_width=True)

            with col2:
                st.markdown("#### 🔍 Términos Destacados")
                st.info("""
                **Top áreas:**
                - **Mexico** (51): Contexto regional
                - **Obesity** (24): Área central
                - **Lupus/Arthritis** (31): Autoinmunes
                - **COVID-19** (13): Respuesta a crisis
                - **Biomarkers** (13): Medicina personalizada
                """)

                st.success("""
                ✅ **Ventaja**: Indexación internacional en PubMed aumenta visibilidad global.
                """)

        # TAB 2: Impacto Científico
        with tab2:
            st.markdown("### Impacto Científico y Citaciones")
            st.markdown("""
            Las **citaciones** reflejan cuántas veces otros investigadores han referenciado el trabajo del programa.
            """)

            # Métricas de impacto
            metrics, fig_citations = create_citations_metrics_and_chart(pubmed_data)

            if metrics:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        label="Total de Citaciones",
                        value=f"{metrics['total']:,}",
                        delta="Impacto Acumulado"
                    )

                with col2:
                    st.metric(
                        label="Promedio por Artículo",
                        value=f"{metrics['average']:.1f}",
                        delta=f"{metrics['count']} artículos citados"
                    )

                with col3:
                    st.metric(
                        label="Artículo Más Citado",
                        value=f"{metrics['max']} citas"
                    )

                with col4:
                    st.metric(
                        label="h-index del Programa",
                        value=metrics['h_index'],
                        delta="Indicador de productividad"
                    )

                st.markdown("")

                # Gráfica de distribución
                if fig_citations:
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.plotly_chart(fig_citations, use_container_width=True)

                    with col2:
                        st.markdown("#### 📊 Interpretación")
                        st.info(f"""
                        **h-index {metrics['h_index']}**: {metrics['h_index']} artículos con ≥{metrics['h_index']} citas.

                        **Contexto:**
                        - Programa joven (2019-2025)
                        - {metrics['average']:.1f} citas/artículo **competitivo**
                        - {metrics['total']:,} citas = reconocimiento internacional
                        """)

                # Top artículos citados
                st.markdown("#### 🏆 Top 10 Artículos Más Citados")

                top_cited = create_top_cited_articles(pubmed_data, top_n=10)

                if top_cited:
                    for i, article in enumerate(top_cited, 1):
                        with st.expander(f"#{i} - {article['citations']} citas - {article['title'][:80]}..."):
                            st.markdown(f"""
                            **PMID:** [{article['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{article['pmid']}/)
                            **Año:** {article['year']} | **Revista:** {article['journal']}
                            **Citaciones:** {article['citations']}

                            **Título:** {article['title']}
                            """)

        # TAB 3: Pirámide de Evidencia
        with tab3:
            st.markdown("### Pirámide de Evidencia Científica")
            st.markdown("""
            Clasificación por **rigor metodológico**. La capacidad de producir diferentes tipos
            de evidencia demuestra **versatilidad científica**.
            """)

            fig_pyramid = create_evidence_pyramid_chart(pubmed_data)

            if fig_pyramid:
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.plotly_chart(fig_pyramid, use_container_width=True)

                with col2:
                    st.markdown("#### 🎯 Calidad Metodológica")
                    st.success("""
                    **Alto Nivel:**
                    - 2 Meta-Análisis
                    - 7 Rev. Sistemáticas
                    - 3 RCTs

                    **12 estudios** de máxima calidad.
                    """)

                    st.info("""
                    **36 Revisiones** = liderazgo en síntesis de conocimiento.
                    """)

        # TAB 4: Financiamiento
        with tab4:
            st.markdown("### Financiamiento Competitivo")
            st.markdown("""
            Indicador de **calidad científica**: requiere evaluación por pares y demostrar pertinencia.
            """)

            funding_metrics, fig_funding = create_funding_analysis(pubmed_data)

            if funding_metrics:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        label="Artículos Financiados",
                        value=funding_metrics['funded'],
                        delta=f"{funding_metrics['percentage']:.1f}%"
                    )

                with col2:
                    st.metric(
                        label="Total Artículos",
                        value=funding_metrics['total']
                    )

                with col3:
                    if funding_metrics['percentage'] >= 25:
                        st.success("✅ **Excelente**")
                    else:
                        st.info("📊 **Bueno**")

                if fig_funding:
                    st.plotly_chart(fig_funding, use_container_width=True)

                    st.info("""
                    **CONACYT/SEP** lideran el financiamiento:
                    - Alineación con prioridades nacionales
                    - Competitividad en convocatorias federales
                    - Reconocimiento de calidad científica
                    """)

        # TAB 5: Red de Colaboración Nacional
        with tab5:
            st.markdown("### Red de Colaboración Institucional")
            st.markdown("""
            **Inserción** en ecosistema científico nacional. Colaboraciones multi-institucionales
            generan sinergias.
            """)

            top_collab = create_collaboration_network_data(pubmed_data)

            if top_collab:
                st.markdown("#### 🌐 Top 10 Instituciones")

                col1, col2 = st.columns(2)

                for i, (institution, count) in enumerate(top_collab):
                    if i < 5:
                        with col1:
                            st.metric(
                                label=institution,
                                value=f"{count} artículos",
                                delta="Co-autoría"
                            )
                    else:
                        with col2:
                            st.metric(
                                label=institution,
                                value=f"{count} artículos",
                                delta="Co-autoría"
                            )

                st.success("""
                ✅ **Red Consolidada**: Colaboraciones con IMSS, hospitales y universidades:
                - Trabajo multi-institucional
                - Acceso a infraestructura diversa
                - Transferencia al sector salud
                - Modelo academia-servicios
                """)

        # TAB 6: Mapa Mundial
        with tab6:
            st.markdown("### Mapa de Colaboración Internacional")
            st.markdown("""
            Visualización geográfica de colaboraciones científicas del DCNT-UdeG con instituciones
            de otros países, demostrando alcance global.
            """)

            map_result = create_collaboration_map(pubmed_data)

            if map_result:
                fig_map, df_map = map_result
                st.plotly_chart(fig_map, use_container_width=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 🌎 Países Colaboradores")
                    df_map_sorted = df_map.sort_values('Articles', ascending=False)
                    st.dataframe(df_map_sorted, use_container_width=True, hide_index=True)

                with col2:
                    st.markdown("#### 🌐 Alcance Global")
                    st.metric("Total de Países", len(df_map))
                    st.metric("País Principal", df_map_sorted.iloc[0]['Country'])
                    st.metric("Artículos más citados", df_map_sorted.iloc[0]['Articles'])

                st.info("""
                **Colaboración Internacional**:
                - Mayor concentración en México (sede del programa)
                - Colaboraciones con USA, España y Latinoamérica
                - Presencia en Europa y Asia
                - Red global demuestra competitividad internacional
                """)

    else:
        st.warning("⚠️ No se encontraron datos de PubMed. Verifica que exista el archivo metadata_updated_20251024_043156.json")

    st.markdown("---")

    # SECCIÓN 4.5: Líneas de Investigación del Doctorado
    st.markdown('<div class="section-header">🎓 Líneas de Investigación del DCNT-UdeG</div>', unsafe_allow_html=True)

    st.markdown("""
    El DCNT-UdeG opera con **tres líneas de investigación complementarias** que cubren todo el espectro
    de la investigación traslacional en nutrición: desde mecanismos moleculares hasta intervenciones poblacionales.
    """)

    # Verificar si hay datos de clasificación
    if lineas_data and 'estadisticas' in lineas_data:
        stats = lineas_data['estadisticas']

        # Métricas principales
        st.markdown("### 📊 Clasificación de Artículos por Línea de Investigación")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Clasificados",
                stats['total_articulos'],
                delta="100% Cobertura"
            )

        with col2:
            st.metric(
                "Multi-Línea",
                stats['multi_linea'],
                delta=f"{stats['multi_linea']/stats['total_articulos']*100:.1f}%"
            )

        with col3:
            alta_confianza = stats['por_confianza']['alta']
            st.metric(
                "Alta Confianza",
                alta_confianza,
                delta=f"{alta_confianza/stats['total_articulos']*100:.1f}%"
            )

        with col4:
            st.metric(
                "Método",
                "Embeddings ML",
                delta="Similitud Coseno"
            )

        st.markdown("")

        # Gráfica de distribución
        fig_dist = create_lineas_distribution_chart(lineas_data)
        if fig_dist:
            st.plotly_chart(fig_dist, use_container_width=True)

        # Información sobre la metodología
        with st.expander("ℹ️ Metodología de Clasificación (Detalles Técnicos)"):
            metadata = lineas_data.get('metadata', {})
            umbrales = metadata.get('umbrales', {})

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
                **Método de Clasificación:**
                - **Embeddings + Similitud Coseno**
                - Modelo: `paraphrase-multilingual-MiniLM-L12-v2`
                - Clasificación basada en similitud semántica real
                - Multilingüe (español + inglés)

                **Umbrales de Similitud:**
                - Línea Principal: Similitud ≥ 0.35 (35%)
                - Línea Secundaria: Similitud ≥ 0.30 (30%)
                - Multi-línea: Artículos con múltiples líneas ≥ umbral secundario

                **Datos Utilizados:**
                - Título completo del artículo
                - Abstract (92.5% disponibles)
                - MeSH terms (vocabulario controlado)
                - Keywords del autor
                """)

            with col2:
                st.markdown(f"""
                **Niveles de Confianza:**
                - 🟢 **Alta** (≥50%): {stats['por_confianza'].get('alta', 0)} artículos
                - 🟡 **Media** (40-50%): {stats['por_confianza'].get('media', 0)} artículos
                - 🟠 **Baja** (35-40%): {stats['por_confianza'].get('baja', 0)} artículos
                - 🔴 **Tentativa** (<35%): {stats['por_confianza'].get('tentativa', 0)} artículos

                **Características del Modelo:**
                - Tamaño: ~420 MB
                - Arquitectura: Sentence Transformers
                - Embeddings: 384 dimensiones
                - Velocidad: ~30 segundos (226 artículos)
                - Distribución equilibrada sin sesgos
                """)

        # Análisis de artículos multi-línea
        if stats['multi_linea'] > 0:
            st.markdown("### 🔗 Análisis de Artículos Multi-Línea")

            st.info(f"""
            **{stats['multi_linea']} artículos ({stats['multi_linea']/stats['total_articulos']*100:.1f}%)**
            pertenecen a múltiples líneas de investigación, demostrando el carácter **interdisciplinario**
            de la investigación del DCNT-UdeG.
            """)

            col1, col2 = st.columns(2)

            with col1:
                fig_upset = create_upset_plot(lineas_data)
                if fig_upset:
                    st.plotly_chart(fig_upset, use_container_width=True)

            with col2:
                fig_matrix = create_lineas_cooccurrence_matrix(lineas_data)
                if fig_matrix:
                    st.plotly_chart(fig_matrix, use_container_width=True)

        st.markdown("---")

    else:
        st.warning("""
        ⚠️ **Datos de clasificación de líneas no disponibles**

        Los datos de clasificación por líneas de investigación no están disponibles.
        Para regenerar estos datos, ejecuta:
        1. `python src/embeddings_classifier.py` - Clasificación con Embeddings
        2. `python src/convert_embeddings_to_dashboard.py` - Conversión a formato dashboard
        """)

    # Tabs por línea
    st.markdown("### 📚 Descripción y Artículos por Línea")

    tab1, tab2, tab3 = st.tabs([
        "🧬 Línea 1: Bases Moleculares y Genómica Nutricional",
        "🏥 Línea 2: Epidemiología Clínica y Factores de Riesgo",
        "👥 Línea 3: Salud Poblacional y Políticas Públicas"
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

        # Tabla de artículos clasificados en Línea 1
        if lineas_data:
            st.markdown("---")
            st.markdown("#### 📄 Artículos Clasificados en esta Línea")

            df_linea1 = filter_articulos_by_linea(lineas_data, 1)

            if not df_linea1.empty:
                # Métricas de la línea
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Total Artículos", len(df_linea1))
                with col_m2:
                    principales = len(df_linea1[df_linea1['Tipo'] == 'Principal'])
                    st.metric("Línea Principal", principales)
                with col_m3:
                    alta_conf = len(df_linea1[df_linea1['Confianza'] == 'Alta'])
                    st.metric("Alta Confianza", alta_conf)

                # Filtros
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    años_disponibles = sorted(df_linea1['Año'].unique())
                    año_filtro = st.multiselect(
                        "Filtrar por Año",
                        options=años_disponibles,
                        default=años_disponibles,
                        key="año_l1"
                    )
                with col_f2:
                    confianza_filtro = st.multiselect(
                        "Filtrar por Confianza",
                        options=['Alta', 'Media', 'Baja', 'Tentativa'],
                        default=['Alta', 'Media', 'Baja', 'Tentativa'],
                        key="conf_l1"
                    )

                # Aplicar filtros
                df_filtrado = df_linea1[
                    (df_linea1['Año'].isin(año_filtro)) &
                    (df_linea1['Confianza'].isin(confianza_filtro))
                ]

                st.dataframe(
                    df_filtrado,
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )

                # Botón de descarga
                csv = df_filtrado.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar tabla como CSV",
                    data=csv,
                    file_name='linea1_genomica_nutricional.csv',
                    mime='text/csv',
                    key="download_l1"
                )
            else:
                st.info("No hay artículos clasificados en esta línea.")

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

        # Tabla de artículos clasificados en Línea 2
        if lineas_data:
            st.markdown("---")
            st.markdown("#### 📄 Artículos Clasificados en esta Línea")

            df_linea2 = filter_articulos_by_linea(lineas_data, 2)

            if not df_linea2.empty:
                # Métricas de la línea
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Total Artículos", len(df_linea2))
                with col_m2:
                    principales = len(df_linea2[df_linea2['Tipo'] == 'Principal'])
                    st.metric("Línea Principal", principales)
                with col_m3:
                    alta_conf = len(df_linea2[df_linea2['Confianza'] == 'Alta'])
                    st.metric("Alta Confianza", alta_conf)

                # Filtros
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    años_disponibles = sorted(df_linea2['Año'].unique())
                    año_filtro = st.multiselect(
                        "Filtrar por Año",
                        options=años_disponibles,
                        default=años_disponibles,
                        key="año_l2"
                    )
                with col_f2:
                    confianza_filtro = st.multiselect(
                        "Filtrar por Confianza",
                        options=['Alta', 'Media', 'Baja', 'Tentativa'],
                        default=['Alta', 'Media', 'Baja', 'Tentativa'],
                        key="conf_l2"
                    )

                # Aplicar filtros
                df_filtrado = df_linea2[
                    (df_linea2['Año'].isin(año_filtro)) &
                    (df_linea2['Confianza'].isin(confianza_filtro))
                ]

                st.dataframe(
                    df_filtrado,
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )

                # Botón de descarga
                csv = df_filtrado.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar tabla como CSV",
                    data=csv,
                    file_name='linea2_salud_publica.csv',
                    mime='text/csv',
                    key="download_l2"
                )
            else:
                st.info("No hay artículos clasificados en esta línea.")

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

        # Tabla de artículos clasificados en Línea 3
        if lineas_data:
            st.markdown("---")
            st.markdown("#### 📄 Artículos Clasificados en esta Línea")

            df_linea3 = filter_articulos_by_linea(lineas_data, 3)

            if not df_linea3.empty:
                # Métricas de la línea
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Total Artículos", len(df_linea3))
                with col_m2:
                    principales = len(df_linea3[df_linea3['Tipo'] == 'Principal'])
                    st.metric("Línea Principal", principales)
                with col_m3:
                    alta_conf = len(df_linea3[df_linea3['Confianza'] == 'Alta'])
                    st.metric("Alta Confianza", alta_conf)

                # Filtros
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    años_disponibles = sorted(df_linea3['Año'].unique())
                    año_filtro = st.multiselect(
                        "Filtrar por Año",
                        options=años_disponibles,
                        default=años_disponibles,
                        key="año_l3"
                    )
                with col_f2:
                    confianza_filtro = st.multiselect(
                        "Filtrar por Confianza",
                        options=['Alta', 'Media', 'Baja', 'Tentativa'],
                        default=['Alta', 'Media', 'Baja', 'Tentativa'],
                        key="conf_l3"
                    )

                # Aplicar filtros
                df_filtrado = df_linea3[
                    (df_linea3['Año'].isin(año_filtro)) &
                    (df_linea3['Confianza'].isin(confianza_filtro))
                ]

                st.dataframe(
                    df_filtrado,
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )

                # Botón de descarga
                csv = df_filtrado.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar tabla como CSV",
                    data=csv,
                    file_name='linea3_alimentacion_nutricion.csv',
                    mime='text/csv',
                    key="download_l3"
                )
            else:
                st.info("No hay artículos clasificados en esta línea.")

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
