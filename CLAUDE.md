# CLAUDE.md

This file provides technical documentation and development guidance for working with code in this repository.

## Project Overview

This is a **Streamlit-based interactive dashboard** for visualizing scientific productivity of the **Doctorado en Ciencias de la Nutrición Traslacional** (DCNT) at Universidad de Guadalajara (2019-2025). The dashboard analyzes 226 scientific publications and their alignment with:
- **ODS (Sustainable Development Goals)** - International commitments
- **PRONACES** (National Strategic Programs) - Mexican national priorities
- **Research themes** - 15+ identified research topics

The dashboard demonstrates the program's strategic relevance to address Mexico's nutritional crises (obesity epidemic, child malnutrition, food insecurity) and its contribution to national/international priorities.

## Commands

### Running the Dashboard
```bash
# Primary method - Direct Streamlit execution
streamlit run src/app.py

# Alternative - Using shell script
bash run_dashboard.sh

# Windows alternative
run_dashboard.bat
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies:**
- `streamlit==1.29.0` - Dashboard framework
- `plotly==5.18.0` - Interactive visualizations
- `networkx==3.2.1` - Theme co-occurrence analysis
- Standard libraries: `pandas`, `json`, `pathlib`, `collections`

### No Testing/Linting
This project does not have automated tests or linting configured. It's a data visualization tool for academic reporting.

## Architecture & Key Design Patterns

### Two-Module Structure

1. **`src/app.py`** (Main Dashboard - 1369 lines)
   - Streamlit page configuration and UI rendering
   - All visualization functions (Plotly charts, metrics, tables)
   - Theme switching system (light/dark mode via CSS injection)
   - Data loading with `@st.cache_data` decorator
   - Seven main sections (Context, Productivity, ODS, PRONACES, Themes, Regional Relevance, Publications Table)

2. **`src/config_context.py`** (Contextual Data - 410 lines)
   - Static reference data as Python dictionaries
   - Epidemiological data (Mexico & Jalisco)
   - ODS context and alignment
   - PRONACES programs details
   - DCNT research lines descriptions
   - Regional infrastructure information

**Why this separation?** The contextual data is extensive and changes infrequently, while app.py handles all dynamic rendering logic. This keeps the main app file focused on visualization logic.

### Data Flow Architecture

```
data/
├── publications_base.csv          # Base publications data (año, titulo, revista, doi, pmid)
├── ods_classification.json        # Publications classified by ODS (generated externally)
├── pronaces_classification.json   # Publications classified by PRONACES (generated externally)
└── themes_classification.json     # Publications classified by research themes (generated externally)
```

**Critical data loading pattern** in `load_data()` (src/app.py:244-271):
- CSV loaded with pandas
- JSON files loaded with try/except (returns `[]` if missing)
- All cached with `@st.cache_data` for performance
- Returns tuple: `(publications_df, ods_data, pronaces_data, themes_data)`

### Theme System Architecture

**Dual-mode CSS injection** (src/app.py:32-242):
- Session state tracks current theme: `st.session_state.theme` ('light' or 'dark')
- `get_theme_css(theme)` returns complete CSS string with aggressive `!important` overrides
- CSS targets Streamlit internal elements via `data-testid` selectors
- Applied globally: `st.markdown(get_theme_css(...), unsafe_allow_html=True)`
- Radio button in sidebar triggers `st.rerun()` to re-render with new theme

**Why aggressive CSS?** Streamlit's theming system has limited customization. The `!important` flags ensure dark mode overrides all default Streamlit styles for consistent appearance.

### Visualization Functions Pattern

All chart functions follow this pattern:
```python
def create_<chart_name>(data):
    """Docstring explaining chart purpose"""
    if not data:
        return None

    # Data processing/transformation
    # ...

    # Create Plotly figure (px.bar, go.Figure, etc.)
    fig = px.bar(...)

    # Customize layout
    fig.update_layout(...)

    return fig
```

Key visualization functions:
- `create_year_evolution_chart()` - Bar chart of publications over time
- `create_ods_distribution()` - Donut chart of ODS alignment
- `create_pronaces_heatmap()` - Heatmap of PRONACES vs years
- `create_themes_treemap()` - Hierarchical theme visualization
- `create_themes_sunburst()` - Categorized theme hierarchy
- `create_themes_cooccurrence()` - Theme relationship matrix using NetworkX concepts

### Why Streamlit?

The dashboard serves a **non-technical academic audience** (doctorate program evaluators). Streamlit enables:
- Zero web development complexity (no HTML/CSS/JS knowledge required)
- Built-in responsive layout with `st.columns()`
- Native support for scientific visualizations (Plotly integration)
- Session state for interactivity (theme switching, year filtering)
- Fast prototyping for academic reporting deadlines

## Critical Implementation Details

### Data Classification Files
The JSON classification files (`ods_classification.json`, `pronaces_classification.json`, `themes_classification.json`) are **generated externally** (likely by an AI classification script referenced in comments like "Ejecuta primero el script de clasificación").

**Structure example** (from code context):
```json
[
  {
    "año": 2019,
    "titulo": "...",
    "ods_principales": [{"numero": 3, "nombre": "Salud y Bienestar"}],
    "pronaces_principales": [{"nombre": "PRONACE Salud"}],
    "temas": [{"nombre": "DIABETES"}]
  }
]
```

If these files are missing, sections show warnings asking users to run classification scripts (src/app.py:841, 907, 997).

### Theme Categorization Logic
The `create_themes_sunburst()` function (src/app.py:490-563) uses hardcoded category mappings:
```python
categorias = {
    'Enfermedades Metabólicas': ['OBESIDAD_SOBREPESO', 'DIABETES', ...],
    'Enfermedades Inmunológicas': ['ENFERMEDADES_AUTOINMUNES', 'COVID19'],
    # etc.
}
```

**Important:** These categories must align with theme names in `themes_classification.json`. If adding new themes, update this mapping.

### Co-occurrence Matrix Algorithm
`create_themes_cooccurrence()` (src/app.py:430-487):
1. Extracts all unique themes from publications
2. Creates N×N matrix (where N = number of themes)
3. For each publication with multiple themes, increments matrix[i][j] for all theme pairs
4. Diagonal (self-correlation) explicitly set to 0
5. Rendered as heatmap showing interdisciplinary connections

This reveals research interdisciplinarity - high values indicate themes frequently studied together.

### Performance Considerations
- **Caching is essential:** `@st.cache_data` on `load_data()` prevents re-reading files on every interaction
- **No database:** All data loaded into memory (acceptable for 226 publications, ~300KB JSON files)
- **CSS reapplication:** Theme CSS re-injected on every page load (necessary for Streamlit's DOM structure)

## File Structure Conventions

```
productividad-dcnt/
├── src/
│   ├── app.py              # Main dashboard (DO NOT split further - Streamlit convention)
│   └── config_context.py   # Static reference data only
├── data/                   # CSV and JSON data files (not tracked in git per .gitignore)
├── .streamlit/
│   └── config.toml         # Base Streamlit config (overridden by CSS in app.py)
├── LOGO DCNT_1.png         # Institution logo (loaded in header/footer)
├── run_dashboard.sh        # Linux/Mac launcher
└── run_dashboard.bat       # Windows launcher
```

**Convention:** Streamlit apps typically live in a single file (`app.py`). Do not split into multiple page files unless converting to multi-page app (currently unnecessary).

## Common Modification Scenarios

### Adding a New Chart Section
1. Create visualization function in `src/app.py` following the pattern above
2. Add section header: `st.markdown('<div class="section-header">Title</div>', unsafe_allow_html=True)`
3. Call visualization function and render: `st.plotly_chart(fig, use_container_width=True)`
4. Add contextual explanation with `st.markdown()` or `st.info()`

### Modifying Contextual Data
Edit `src/config_context.py` dictionaries directly. These are reference data from academic documents (ENSANUT 2022, PRONACES docs, etc.). Sources are documented in comments.

### Changing Theme Colors
Modify CSS in `get_theme_css()` function (src/app.py:38-239):
- Dark mode: Search for `theme == 'dark'` section
- Light mode: Search for `else:` section (line 140)
- Key color variables: `#58a6ff` (dark primary), `#1f77b4` (light primary)

### Adding New Research Themes
1. Ensure theme name appears in `themes_classification.json`
2. Add to appropriate category in `create_themes_sunburst()` (line 496)
3. Theme will automatically appear in other visualizations (distribution, treemap, co-occurrence)

## Known Limitations

- **No user authentication:** Dashboard is public (appropriate for academic reporting)
- **No data editing:** All data management happens externally (CSV/JSON files)
- **Hardcoded text:** Most Spanish text is hardcoded (not i18n ready)
- **No responsive mobile optimization:** Designed for desktop viewing (academic presentation context)
- **Single-instance deployment:** No multi-user session handling needed

## Development Workflow

1. Modify `src/app.py` or `src/config_context.py`
2. Save changes
3. Streamlit auto-reloads (if watching) or rerun manually: `streamlit run src/app.py`
4. Test in browser at `http://localhost:8501`
5. Commit changes to git

**No build step required** - Streamlit is a runtime framework.
