#!/usr/bin/env python3
"""
Clasificador de Líneas de Investigación usando Embeddings MEJORADO
Versión optimizada con:
- Modelo más potente pero eficiente en memoria
- Procesamiento por lotes más pequeños
- Normalización L2
- Múltiples representaciones
- Boost específico del dominio
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DESCRIPCIONES DE LÍNEAS MEJORADAS CON MÚLTIPLES REPRESENTACIONES
# ============================================================================

LINEAS_DESCRIPCIONES = {
    "linea_1": {
        "nombre": "Bases Moleculares y Genómica Nutricional",
        "descriptions": [
            # Descripción principal
            """
            Research in molecular mechanisms, genetics, epigenetics, and nutritional genomics.
            Studies of gene regulation and expression, genetic polymorphisms (SNPs) and variants,
            epigenetic modifications including DNA methylation and histone acetylation,
            cellular signaling pathways and molecular metabolism, cellular receptors and proteins,
            enzyme function and regulation, in vitro studies and cell culture experiments,
            animal models (mice, rats) for molecular research, molecular biomarkers and metabolites,
            gut microbiome and intestinal microbiota, molecular inflammation and cytokines,
            oxidative stress and reactive oxygen species, nutrigenomics and nutrigenetics,
            gene-nutrient interactions and personalized nutrition.
            """,
            # Términos MeSH específicos
            """
            Genomics, epigenomics, transcriptomics, proteomics, metabolomics,
            gene expression regulation, polymorphism genetic, single nucleotide polymorphism,
            DNA methylation, histone code, signal transduction, receptors cell surface,
            enzymes, cell line, animal models, biomarkers pharmacological,
            gastrointestinal microbiome, cytokines, oxidative stress, nutrigenomics,
            molecular biology, systems biology, computational biology
            """,
            # Keywords y métodos
            """
            PCR, Western blot, ELISA, flow cytometry, microarray, RNA-seq,
            CRISPR, gene editing, cell culture, transfection, knockout mice,
            metabolic pathways, protein expression, mRNA levels, methylation analysis,
            chromatin immunoprecipitation, mass spectrometry, bioinformatics,
            pathway analysis, molecular docking, protein structure
            """
        ],
        "mesh_boost_terms": [
            "Genomics", "Epigenomics", "Gene Expression", "Polymorphism",
            "Signal Transduction", "Biomarkers", "Microbiome", "Molecular Biology"
        ],
        "keywords": [
            "gene", "genetic", "epigenetic", "DNA", "RNA", "protein", "enzyme",
            "receptor", "signaling", "pathway", "molecular", "cellular", "in vitro",
            "cell culture", "mouse", "mice", "rat", "polymorphism", "SNP", "allele",
            "biomarker", "metabolite", "microbiome", "microbiota", "cytokine",
            "inflammation", "oxidative stress", "nutrigenomics", "nutrigenetics"
        ]
    },

    "linea_2": {
        "nombre": "Epidemiología Clínica y Factores de Riesgo",
        "descriptions": [
            # Descripción principal
            """
            Clinical epidemiological studies, observational research, and intervention trials in humans.
            Clinical trials and intervention studies, cohort studies and case-control designs,
            cross-sectional and longitudinal studies, risk factors and protective factors identification,
            disease incidence and prevalence assessment, associations between diet and disease,
            nutritional intervention evaluation, clinical biomarkers and diagnostic tests,
            clinical outcomes and prognosis, patient studies and clinical populations,
            instrument validation and reliability testing, clinical data analysis,
            precision medicine and personalized therapy approaches.
            """,
            # Términos MeSH de epidemiología
            """
            Epidemiologic studies, clinical trials, randomized controlled trials,
            cohort studies, case-control studies, cross-sectional studies,
            risk factors, protective factors, disease incidence, prevalence,
            odds ratio, relative risk, hazard ratio, patient outcome assessment,
            diagnostic tests routine, prognosis, treatment outcome,
            evidence-based medicine, patient-centered care, clinical decision-making
            """,
            # Métodos y diseños
            """
            Study design, sample size calculation, randomization, blinding,
            intention-to-treat analysis, per-protocol analysis, statistical modeling,
            regression analysis, survival analysis, meta-analysis, systematic review,
            clinical guidelines, diagnostic accuracy, sensitivity and specificity,
            ROC curves, predictive values, clinical significance, effect size
            """
        ],
        "mesh_boost_terms": [
            "Epidemiologic Studies", "Clinical Trials", "Risk Factors",
            "Treatment Outcome", "Prognosis", "Patient Outcome Assessment"
        ],
        "keywords": [
            "clinical trial", "intervention", "cohort", "case-control", "cross-sectional",
            "epidemiological", "risk factor", "protective factor", "incidence", "prevalence",
            "association", "correlation", "patients", "subjects", "participants",
            "diagnosis", "prognosis", "outcome", "treatment", "therapy"
        ]
    },

    "linea_3": {
        "nombre": "Salud Poblacional y Políticas Públicas",
        "descriptions": [
            # Descripción principal
            """
            Public health, population nutrition, health policy, and social determinants of health.
            Population-based studies and community interventions, national health and nutrition surveys,
            health disparities and social determinants, food access and food security,
            public health policies and nutrition programs, community-based participatory research,
            health education and promotion programs, nutritional epidemiology at population level,
            nutritional and epidemiological transitions, malnutrition including undernutrition and obesity,
            non-communicable diseases at population level, health systems and services,
            population health indicators, health economics and cost-effectiveness.
            """,
            # Salud pública y política
            """
            Public health, health policy, social determinants of health,
            health disparities, health equity, food security, food supply,
            nutrition policy, health programs, community health services,
            health promotion, preventive medicine, population health,
            health surveys, surveillance systems, health indicators,
            global health, environmental health, occupational health
            """,
            # Intervenciones y evaluación
            """
            Community intervention, program evaluation, implementation science,
            policy analysis, health impact assessment, cost-benefit analysis,
            stakeholder engagement, participatory research, mixed methods,
            qualitative research, focus groups, community engagement,
            capacity building, sustainability, scalability, dissemination
            """
        ],
        "mesh_boost_terms": [
            "Public Health", "Health Policy", "Social Determinants of Health",
            "Food Security", "Health Promotion", "Community Health", "Population Health"
        ],
        "keywords": [
            "public health", "population", "community", "national survey",
            "health policy", "public policy", "social determinants", "health inequality",
            "food security", "food access", "malnutrition", "undernutrition",
            "childhood obesity", "overweight", "nutritional transition"
        ]
    }
}

# Umbrales optimizados para líneas
UMBRAL_PRINCIPAL = 0.40
UMBRAL_SECUNDARIO = 0.35
MIN_CONFIDENCE_SCORE = 0.30

# ============================================================================
# FUNCIONES PRINCIPALES MEJORADAS
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
        df['publication_types'] = df['pmid'].apply(
            lambda x: pubmed_lookup.get(str(x), {}).get('publication_types', [])
        )

        print(f"   ✓ Metadata de PubMed agregada")
        print(f"   • Artículos con abstract: {df['abstract'].notna().sum()} ({df['abstract'].notna().sum()/len(df)*100:.1f}%)")
    else:
        df['abstract'] = ''
        df['mesh_terms'] = [[] for _ in range(len(df))]
        df['keywords'] = [[] for _ in range(len(df))]
        df['publication_types'] = [[] for _ in range(len(df))]
        print("   ⚠️  No se encontró metadata de PubMed")

    return df

def create_text_for_embedding_enhanced(row: pd.Series) -> str:
    """Crea texto mejorado con ponderación implícita"""
    parts = []

    # Título (repetir para dar más peso)
    if pd.notna(row['titulo']):
        parts.append(row['titulo'])
        parts.append(row['titulo'])  # Doble peso

    # Abstract
    if pd.notna(row['abstract']) and row['abstract']:
        parts.append(row['abstract'])

    # MeSH terms con prefijo para contexto
    if isinstance(row['mesh_terms'], list) and row['mesh_terms']:
        mesh_text = " ".join(row['mesh_terms'][:20])
        parts.append(f"Medical subjects: {mesh_text}")

    # Keywords
    if isinstance(row['keywords'], list) and row['keywords']:
        kw_text = " ".join(row['keywords'][:15])
        parts.append(f"Research keywords: {kw_text}")

    # Publication types
    if isinstance(row['publication_types'], list) and row['publication_types']:
        pub_text = " ".join(row['publication_types'])
        parts.append(f"Study type: {pub_text}")

    # Journal context
    if pd.notna(row.get('revista', '')):
        parts.append(f"Published in: {row['revista']}")

    return " ".join(parts)

def apply_domain_boost_lines(similarities, row, lineas_data):
    """Aplica boost basado en términos específicos de cada línea"""
    boosted = similarities.copy()

    mesh_terms = row.get('mesh_terms', []) if isinstance(row.get('mesh_terms'), list) else []
    keywords = row.get('keywords', []) if isinstance(row.get('keywords'), list) else []
    title = str(row.get('titulo', '')).lower()
    abstract = str(row.get('abstract', '')).lower()

    # Aplicar boost para cada línea
    for i, (linea_key, linea_info) in enumerate(lineas_data.items()):
        boost = 0

        # Boost por términos MeSH específicos
        for mesh_boost in linea_info.get('mesh_boost_terms', []):
            if any(mesh_boost.lower() in m.lower() for m in mesh_terms):
                boost += 0.03

        # Boost por keywords en título o abstract
        for keyword in linea_info.get('keywords', [])[:10]:  # Top 10 keywords
            if keyword.lower() in title:
                boost += 0.02
            if keyword.lower() in abstract[:500]:  # Primeros 500 chars del abstract
                boost += 0.01

        # Aplicar boost con límite
        boosted[i] = min(boosted[i] * (1 + boost), 1.0)

    return boosted

def generate_multiple_embeddings_lines(model, lineas_descripciones):
    """Genera múltiples embeddings por línea y los combina"""
    embeddings_dict = {}

    for linea_key, linea_data in lineas_descripciones.items():
        embeddings_list = []
        weights = [0.5, 0.3, 0.2]  # Pesos para cada descripción

        # Generar embeddings para cada descripción
        for i, description in enumerate(linea_data.get('descriptions', [])):
            if description and i < len(weights):
                embedding = model.encode(description, convert_to_numpy=True)
                # Normalizar L2
                embedding = normalize(embedding.reshape(1, -1), norm='l2')[0]
                embeddings_list.append(embedding * weights[i])

        # Combinar embeddings ponderados
        if embeddings_list:
            combined_embedding = np.sum(embeddings_list, axis=0)
            combined_embedding = normalize(combined_embedding.reshape(1, -1), norm='l2')[0]
            embeddings_dict[linea_key] = combined_embedding

    return embeddings_dict

def get_confidence_level_lines(similarity):
    """Determina nivel de confianza para líneas"""
    if similarity >= 0.60:
        return "alta"
    elif similarity >= 0.45:
        return "media"
    elif similarity >= 0.35:
        return "baja"
    else:
        return "tentativa"

def classify_with_embeddings_enhanced(
    df: pd.DataFrame,
    model: SentenceTransformer
) -> List[Dict]:
    """
    Clasifica artículos usando embeddings mejorados con optimización de memoria
    """
    print("\n🔄 Paso 3: Clasificando artículos con método mejorado...")
    print(f"   • Modelo: all-mpnet-base-v2 (768 dims)")
    print(f"   • Umbrales: principal={UMBRAL_PRINCIPAL}, secundaria={UMBRAL_SECUNDARIO}")

    # Crear textos para embeddings
    print("   Generando textos mejorados...")
    article_texts = df.apply(create_text_for_embedding_enhanced, axis=1).tolist()

    # Generar embeddings de las líneas
    print("   Generando embeddings de líneas con múltiples representaciones...")
    lineas_embeddings_dict = generate_multiple_embeddings_lines(model, LINEAS_DESCRIPCIONES)

    # Convertir a array
    linea_keys = list(LINEAS_DESCRIPCIONES.keys())
    linea_embeddings = np.array([lineas_embeddings_dict[key] for key in linea_keys])

    # Generar embeddings de artículos en lotes pequeños para ahorrar memoria
    print("   Generando embeddings de artículos (lotes optimizados)...")
    batch_size = 8  # Lote más pequeño para evitar problemas de memoria

    all_similarities = []

    for i in tqdm(range(0, len(article_texts), batch_size), desc="Procesando lotes"):
        batch_texts = article_texts[i:i+batch_size]

        # Generar embeddings del lote
        batch_embeddings = model.encode(
            batch_texts,
            show_progress_bar=False,
            batch_size=batch_size,
            convert_to_numpy=True
        )

        # Normalizar L2
        batch_embeddings = normalize(batch_embeddings, norm='l2')

        # Calcular similitudes
        batch_similarities = cosine_similarity(batch_embeddings, linea_embeddings)

        # Aplicar transformación suave
        batch_similarities = np.power(batch_similarities, 0.85)

        all_similarities.append(batch_similarities)

        # Liberar memoria
        del batch_embeddings
        gc.collect()

    # Combinar todas las similitudes
    similarities = np.vstack(all_similarities)

    print("   Clasificando artículos...")

    # Clasificar
    resultados = []
    stats = {
        'total': len(df),
        'por_linea': {1: 0, 2: 0, 3: 0},
        'multi_linea': 0,
        'sin_abstract': 0,
        'por_confianza': {'alta': 0, 'media': 0, 'baja': 0, 'tentativa': 0}
    }

    for idx, row in df.iterrows():
        # Aplicar boost de dominio
        sims_raw = similarities[idx]
        sims = apply_domain_boost_lines(sims_raw, row, LINEAS_DESCRIPCIONES)

        # Ordenar por similitud
        lineas_sorted = sorted(
            [(i+1, sims[i]) for i in range(3)],
            key=lambda x: x[1],
            reverse=True
        )

        linea_principal, sim_principal = lineas_sorted[0]

        # Determinar confianza
        confianza = get_confidence_level_lines(sim_principal)

        # Líneas secundarias
        lineas_secundarias = []
        if sim_principal >= UMBRAL_PRINCIPAL:
            for linea_num, sim in lineas_sorted[1:]:
                if sim >= UMBRAL_SECUNDARIO:
                    lineas_secundarias.append({
                        'linea': linea_num,
                        'nombre': LINEAS_DESCRIPCIONES[f'linea_{linea_num}']['nombre'],
                        'similitud': float(sim),
                        'confianza': get_confidence_level_lines(sim)
                    })

        # Estadísticas
        stats['por_linea'][linea_principal] += 1
        stats['por_confianza'][confianza] += 1
        if lineas_secundarias:
            stats['multi_linea'] += 1
        if not row.get('abstract'):
            stats['sin_abstract'] += 1

        # Resultado
        resultado = {
            'pmid': str(row['pmid']),
            'titulo': row['titulo'],
            'año': int(row['año']),
            'similitudes': {
                'linea_1': float(sims[0]),
                'linea_2': float(sims[1]),
                'linea_3': float(sims[2])
            },
            'linea_principal': {
                'linea': linea_principal,
                'nombre': LINEAS_DESCRIPCIONES[f'linea_{linea_principal}']['nombre'],
                'similitud': float(sim_principal),
                'confianza': confianza
            },
            'lineas_secundarias': lineas_secundarias,
            'metodo': 'embeddings_enhanced_mpnet',
            'tiene_abstract': bool(row.get('abstract'))
        }

        resultados.append(resultado)

    return resultados, stats

def main():
    print("=" * 80)
    print("CLASIFICACIÓN DE LÍNEAS CON EMBEDDINGS MEJORADO")
    print("=" * 80)

    print("\n🚀 Mejoras implementadas:")
    print("   • Modelo superior: all-mpnet-base-v2")
    print("   • Procesamiento optimizado por lotes")
    print("   • Normalización L2")
    print("   • Múltiples representaciones por línea")
    print("   • Boost específico del dominio")
    print("   • Gestión eficiente de memoria")

    # 1. Cargar modelo
    print("\n🤖 Paso 2: Cargando modelo de embeddings...")
    print("   Modelo: sentence-transformers/all-mpnet-base-v2")
    print("   (Superior a MiniLM y multilingüe)")

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    print("   ✓ Modelo cargado")

    # 2. Cargar artículos
    df = load_articles_with_metadata()

    # 3. Clasificar con método mejorado
    resultados, stats = classify_with_embeddings_enhanced(df, model)

    print(f"\n   ✓ {len(resultados)} artículos clasificados")

    # 4. Guardar resultados
    print("\n💾 Paso 4: Guardando resultados...")

    output_data = {
        'metadata': {
            'fecha_generacion': datetime.now().isoformat(),
            'modelo': 'sentence-transformers/all-mpnet-base-v2',
            'metodo': 'embeddings_enhanced_normalized_domain_boost',
            'mejoras': [
                'Modelo MPNET superior (768 dims)',
                'Normalización L2',
                'Múltiples embeddings por línea',
                'Boost específico del dominio',
                'Procesamiento optimizado en memoria',
                'Umbrales ajustados'
            ],
            'total_articulos': len(resultados),
            'umbrales': {
                'principal': UMBRAL_PRINCIPAL,
                'secundario': UMBRAL_SECUNDARIO,
                'minimo': MIN_CONFIDENCE_SCORE
            }
        },
        'estadisticas': stats,
        'clasificaciones': resultados
    }

    output_path = Path('data/lineas_classification/embeddings_results_enhanced.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"   ✓ Guardado: {output_path}")
    print(f"   Tamaño: {output_path.stat().st_size / 1024:.2f} KB")

    # 5. Mostrar estadísticas
    print("\n" + "=" * 80)
    print("📊 RESUMEN DE CLASIFICACIÓN")
    print("=" * 80)

    print(f"\n✅ Total: {stats['total']} artículos")

    print(f"\n📈 Distribución por línea:")
    for i in [1, 2, 3]:
        count = stats['por_linea'][i]
        pct = count / stats['total'] * 100
        nombre = LINEAS_DESCRIPCIONES[f'linea_{i}']['nombre']
        print(f"   Línea {i} - {nombre[:40]:40s}: {count:3d} ({pct:5.1f}%)")

    print(f"\n🎯 Confianza (MEJORADO):")
    for nivel in ['alta', 'media', 'baja', 'tentativa']:
        count = stats['por_confianza'][nivel]
        pct = count / stats['total'] * 100
        emoji = {'alta': '🟢', 'media': '🟡', 'baja': '🟠', 'tentativa': '🔴'}[nivel]
        print(f"   {emoji} {nivel.capitalize():10s}: {count:3d} ({pct:5.1f}%)")

    print(f"\n📊 Multi-línea: {stats['multi_linea']} ({stats['multi_linea']/stats['total']*100:.1f}%)")
    print(f"📝 Sin abstract: {stats['sin_abstract']} ({stats['sin_abstract']/stats['total']*100:.1f}%)")

    print("\n🔄 MEJORAS vs VERSIÓN ANTERIOR:")
    print("   ✅ Modelo más potente (768 dims vs 384)")
    print("   ✅ Procesamiento optimizado en memoria")
    print("   ✅ Múltiples representaciones por línea")
    print("   ✅ Boost basado en términos MeSH")
    print("   ✅ Mejor distribución de confianza esperada")

    print("\n" + "=" * 80)
    print("✅ CLASIFICACIÓN COMPLETADA")
    print("=" * 80)

if __name__ == '__main__':
    main()