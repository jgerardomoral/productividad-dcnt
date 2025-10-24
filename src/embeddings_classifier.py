#!/usr/bin/env python3
"""
Clasificador de artículos usando Embeddings y Similitud Coseno
Método más equilibrado que Zero-Shot para clasificación multi-label
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DESCRIPCIONES DE LÍNEAS DE INVESTIGACIÓN
# ============================================================================

LINEAS_DESCRIPCIONES = {
    "linea_1": {
        "nombre": "Bases Moleculares y Genómica Nutricional",
        "descripcion": """
        Investigación en mecanismos moleculares, genética, epigenética, y genómica nutricional.
        Incluye estudios de:
        - Regulación génica y expresión de genes
        - Polimorfismos genéticos (SNPs) y variantes genéticas
        - Epigenética y modificaciones epigenéticas (metilación, acetilación)
        - Vías de señalización celular y metabolismo molecular
        - Receptores celulares, proteínas, enzimas
        - Estudios in vitro y cultivos celulares
        - Modelos animales (ratones, ratas) para estudios moleculares
        - Biomarcadores moleculares y metabolitos
        - Microbioma y microbiota intestinal
        - Inflamación a nivel molecular (citocinas, mediadores)
        - Estrés oxidativo y especies reactivas de oxígeno
        - Nutrigenómica y nutrigenética
        - Interacciones gen-nutriente
        """,
        "keywords": [
            "gene", "genetic", "epigenetic", "DNA", "RNA", "protein", "enzyme",
            "receptor", "signaling", "pathway", "molecular", "cellular", "in vitro",
            "cell culture", "mouse", "mice", "rat", "polymorphism", "SNP", "allele",
            "biomarker", "metabolite", "microbiome", "microbiota", "cytokine",
            "inflammation", "oxidative stress", "nutrigenomics", "nutrigenetics",
            "methylation", "acetylation", "transcription", "expression"
        ]
    },

    "linea_2": {
        "nombre": "Epidemiología Clínica y Factores de Riesgo",
        "descripcion": """
        Estudios epidemiológicos, clínicos, observacionales y de intervención en humanos.
        Incluye:
        - Ensayos clínicos y estudios de intervención
        - Estudios de cohorte, caso-control, transversales
        - Factores de riesgo y factores protectores
        - Incidencia y prevalencia de enfermedades
        - Asociaciones entre dieta y enfermedad en poblaciones
        - Evaluación de intervenciones nutricionales
        - Biomarcadores clínicos y pruebas diagnósticas
        - Outcomes clínicos y pronóstico
        - Estudios en pacientes y poblaciones clínicas
        - Validación de instrumentos de evaluación
        - Análisis de datos clínicos y epidemiológicos
        - Medicina de precisión y personalizada
        """,
        "keywords": [
            "clinical trial", "intervention", "cohort", "case-control", "cross-sectional",
            "epidemiological", "risk factor", "protective factor", "incidence", "prevalence",
            "association", "correlation", "patients", "subjects", "participants",
            "diagnosis", "prognosis", "outcome", "treatment", "therapy",
            "randomized", "placebo", "double-blind", "clinical study",
            "human study", "observational", "prospective", "retrospective"
        ]
    },

    "linea_3": {
        "nombre": "Salud Poblacional y Políticas Públicas",
        "descripcion": """
        Salud pública, nutrición poblacional, políticas de salud y determinantes sociales.
        Incluye:
        - Estudios poblacionales y comunitarios
        - Encuestas nacionales de salud y nutrición
        - Desigualdades en salud y determinantes sociales
        - Acceso a alimentos y seguridad alimentaria
        - Políticas públicas de salud y nutrición
        - Programas de intervención comunitaria
        - Educación nutricional y promoción de la salud
        - Epidemiología nutricional poblacional
        - Transición nutricional y epidemiológica
        - Malnutrición: desnutrición, sobrepeso, obesidad infantil
        - Enfermedades crónicas no transmisibles a nivel poblacional
        - Sistemas de salud y servicios de salud
        - Indicadores de salud poblacional
        - Costos y economía de la salud
        """,
        "keywords": [
            "public health", "population", "community", "national survey",
            "health policy", "public policy", "social determinants", "health inequality",
            "food security", "food access", "malnutrition", "undernutrition",
            "childhood obesity", "overweight", "nutritional transition",
            "health promotion", "nutrition education", "health program",
            "community intervention", "health system", "health services",
            "health economics", "cost-effectiveness", "prevalence rate",
            "population-based", "nationwide"
        ]
    }
}

# ============================================================================
# FUNCIONES PRINCIPALES
# ============================================================================

def load_articles_with_metadata() -> pd.DataFrame:
    """Carga artículos con metadata completa de PubMed"""
    print("📂 Paso 1: Cargando artículos con metadata de PubMed...")

    # Cargar base de artículos
    df = pd.read_csv('data/publications_base.csv')
    print(f"   ✓ {len(df)} artículos cargados")

    # Cargar metadata de PubMed si existe
    pubmed_file = Path('data/pubmed_extracted/metadata_updated_20251024_043156.json')
    if pubmed_file.exists():
        with open(pubmed_file, 'r', encoding='utf-8') as f:
            pubmed_data = json.load(f)

        # Crear lookup por PMID
        pubmed_lookup = {art['pmid']: art for art in pubmed_data}

        # Agregar metadata
        df['abstract'] = df['pmid'].apply(
            lambda x: pubmed_lookup.get(str(x), {}).get('abstract', '')
        )
        df['mesh_terms'] = df['pmid'].apply(
            lambda x: pubmed_lookup.get(str(x), {}).get('mesh_terms', [])
        )
        df['keywords'] = df['pmid'].apply(
            lambda x: pubmed_lookup.get(str(x), {}).get('keywords', [])
        )

        print(f"   ✓ Metadata de PubMed agregada")
        print(f"   • Artículos con abstract: {df['abstract'].notna().sum()} ({df['abstract'].notna().sum()/len(df)*100:.1f}%)")
    else:
        df['abstract'] = ''
        df['mesh_terms'] = [[] for _ in range(len(df))]
        df['keywords'] = [[] for _ in range(len(df))]
        print("   ⚠️  No se encontró metadata de PubMed")

    return df

def create_text_for_embedding(row: pd.Series) -> str:
    """Crea texto completo para embedding"""
    parts = []

    # Título (siempre presente)
    if pd.notna(row['titulo']):
        parts.append(row['titulo'])

    # Abstract (si disponible)
    if pd.notna(row['abstract']) and row['abstract']:
        parts.append(row['abstract'])

    # MeSH terms (vocabulario controlado)
    if isinstance(row['mesh_terms'], list) and row['mesh_terms']:
        mesh_text = " ".join(row['mesh_terms'])
        parts.append(f"Medical terms: {mesh_text}")

    # Keywords
    if isinstance(row['keywords'], list) and row['keywords']:
        kw_text = " ".join(row['keywords'])
        parts.append(f"Keywords: {kw_text}")

    return " ".join(parts)

def classify_with_embeddings(
    df: pd.DataFrame,
    model: SentenceTransformer,
    threshold_primary: float = 0.35,
    threshold_secondary: float = 0.30
) -> List[Dict]:
    """
    Clasifica artículos usando embeddings y similitud coseno

    Args:
        df: DataFrame con artículos
        model: Modelo de SentenceTransformer
        threshold_primary: Umbral para línea principal
        threshold_secondary: Umbral para líneas secundarias
    """
    print("\n🔄 Paso 3: Clasificando artículos...")
    print(f"   • Umbrales: primaria={threshold_primary}, secundaria={threshold_secondary}")

    # Crear textos para embeddings
    print("   Generando textos...")
    article_texts = df.apply(create_text_for_embedding, axis=1).tolist()

    # Generar embeddings de las descripciones de líneas
    print("   Generando embeddings de líneas...")
    linea_texts = [
        f"{data['nombre']}: {data['descripcion']}"
        for data in LINEAS_DESCRIPCIONES.values()
    ]
    linea_embeddings = model.encode(linea_texts, show_progress_bar=False)

    # Generar embeddings de artículos
    print("   Generando embeddings de artículos...")
    article_embeddings = model.encode(
        article_texts,
        show_progress_bar=True,
        batch_size=32
    )

    # Calcular similitudes
    print("   Calculando similitudes...")
    similarities = cosine_similarity(article_embeddings, linea_embeddings)

    # Clasificar
    resultados = []
    stats = {
        'total': len(df),
        'por_linea': {1: 0, 2: 0, 3: 0},
        'multi_linea': 0,
        'sin_abstract': 0,
        'por_confianza': {'alta': 0, 'media': 0, 'baja': 0, 'tentativa': 0}
    }

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Clasificando"):
        sims = similarities[idx]

        # Ordenar por similitud
        lineas_sorted = sorted(
            [(i+1, sims[i]) for i in range(3)],
            key=lambda x: x[1],
            reverse=True
        )

        linea_principal, sim_principal = lineas_sorted[0]

        # Determinar confianza
        if sim_principal >= 0.5:
            confianza = 'alta'
        elif sim_principal >= 0.40:
            confianza = 'media'
        elif sim_principal >= threshold_primary:
            confianza = 'baja'
        else:
            confianza = 'tentativa'

        stats['por_confianza'][confianza] += 1
        stats['por_linea'][linea_principal] += 1

        # Líneas secundarias
        lineas_secundarias = []
        for linea, sim in lineas_sorted[1:]:
            if sim >= threshold_secondary:
                lineas_secundarias.append({
                    'linea': linea,
                    'nombre': LINEAS_DESCRIPCIONES[f'linea_{linea}']['nombre'],
                    'similitud': float(sim)
                })

        if lineas_secundarias:
            stats['multi_linea'] += 1

        if not row['abstract']:
            stats['sin_abstract'] += 1

        resultado = {
            'pmid': str(row['pmid']),
            'titulo': row['titulo'],
            'año': int(row['año']),
            'similitudes': {
                f'linea_{i+1}': float(sims[i])
                for i in range(3)
            },
            'linea_principal': {
                'linea': linea_principal,
                'nombre': LINEAS_DESCRIPCIONES[f'linea_{linea_principal}']['nombre'],
                'similitud': float(sim_principal),
                'confianza': confianza
            },
            'lineas_secundarias': lineas_secundarias,
            'metodo': 'embeddings_cosine_similarity',
            'tiene_abstract': bool(row['abstract'])
        }

        resultados.append(resultado)

    return resultados, stats

def save_results(resultados: List[Dict], stats: Dict, model_name: str):
    """Guarda resultados en formato JSON"""
    output_file = Path('data/lineas_classification/embeddings_results.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'metadata': {
            'fecha_generacion': datetime.now().isoformat(),
            'modelo': model_name,
            'metodo': 'embeddings_cosine_similarity',
            'total_articulos': stats['total']
        },
        'estadisticas': stats,
        'clasificaciones': resultados
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Resultados guardados: {output_file}")
    print(f"   Tamaño: {output_file.stat().st_size / 1024:.1f} KB")

def print_summary(stats: Dict):
    """Imprime resumen de resultados"""
    print("\n" + "=" * 80)
    print("📊 RESUMEN DE CLASIFICACIÓN CON EMBEDDINGS")
    print("=" * 80)
    print()

    print(f"✅ Total clasificados: {stats['total']} artículos (100%)")
    print()

    print("📈 Distribución por línea (principales):")
    for linea, count in sorted(stats['por_linea'].items()):
        pct = count / stats['total'] * 100
        print(f"   Línea {linea}: {count} artículos ({pct:.1f}%)")
    print()

    print("🎯 Distribución por confianza:")
    icons = {'alta': '🟢', 'media': '🟡', 'baja': '🟠', 'tentativa': '🔴'}
    for nivel, count in stats['por_confianza'].items():
        pct = count / stats['total'] * 100
        print(f"   {icons[nivel]} {nivel.capitalize()}: {count} artículos ({pct:.1f}%)")
    print()

    print(f"📊 Multi-línea: {stats['multi_linea']} artículos ({stats['multi_linea']/stats['total']*100:.1f}%)")
    print(f"📝 Sin abstract: {stats['sin_abstract']} artículos ({stats['sin_abstract']/stats['total']*100:.1f}%)")
    print()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("CLASIFICACIÓN ML CON EMBEDDINGS + SIMILITUD COSENO")
    print("=" * 80)
    print()

    # Detectar dispositivo
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Dispositivo: {device.upper()}")
    if device == 'cpu':
        print("   ⚠️  No GPU detectada, procesamiento en CPU (~15-20 min)")
    else:
        print("   ✓ GPU detectada, procesamiento acelerado (~5-10 min)")
    print()

    # Cargar modelo
    print("🤖 Paso 2: Cargando modelo de embeddings...")
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    print(f"   Modelo: {model_name}")
    print("   Descargando... (solo primera vez, ~420 MB)")

    model = SentenceTransformer(model_name, device=device)
    print("   ✓ Modelo cargado exitosamente")
    print()

    # Cargar artículos
    df = load_articles_with_metadata()

    # Clasificar
    resultados, stats = classify_with_embeddings(
        df,
        model,
        threshold_primary=0.35,
        threshold_secondary=0.30
    )

    print("\n📊 Paso 4: Generando estadísticas...")

    # Guardar
    print("\n💾 Paso 5: Guardando resultados...")
    save_results(resultados, stats, model_name)

    # Resumen
    print_summary(stats)

    print("=" * 80)
    print("✅ CLASIFICACIÓN CON EMBEDDINGS COMPLETADA")
    print("=" * 80)
    print()
    print("🎯 VENTAJAS DE ESTE MÉTODO:")
    print("   ✅ Más equilibrado - no favorece ninguna línea")
    print("   ✅ Similitud semántica real")
    print("   ✅ Multi-label natural")
    print("   ✅ Funciona con español e inglés")
    print()

if __name__ == '__main__':
    main()
