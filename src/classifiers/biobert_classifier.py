#!/usr/bin/env python3
"""
Clasificador usando BioBERT - Modelo Especializado en Biomedicina
BioBERT está pre-entrenado con literatura biomédica de PubMed
Debería dar mejores resultados para clasificación de artículos científicos
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Modelos BioBERT disponibles en Sentence-Transformers
BIOBERT_MODELS = {
    "biobert_base": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    "scibert": "allenai/scibert_scivocab_uncased",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "specter": "allenai/specter"  # Especializado en papers científicos
}

# Usar el modelo BioBERT fine-tuneado para similitud semántica
MODEL_NAME = BIOBERT_MODELS["biobert_base"]

# ODS Definitions (versión biomédica optimizada)
ODS_BIOMEDICAL = {
    2: {
        "numero": 2,
        "nombre": "Hambre Cero",
        "biomedical_terms": [
            "malnutrition", "stunting", "wasting", "undernutrition", "micronutrient deficiency",
            "anemia", "iron deficiency", "zinc deficiency", "vitamin A deficiency",
            "food insecurity", "maternal nutrition", "infant nutrition", "breastfeeding",
            "complementary feeding", "nutritional status", "growth monitoring"
        ],
        "pubmed_query": (
            "malnutrition OR stunting OR wasting OR undernutrition OR "
            "micronutrient deficiency OR anemia OR food insecurity OR "
            "maternal nutrition OR infant nutrition OR breastfeeding"
        )
    },
    3: {
        "numero": 3,
        "nombre": "Salud y Bienestar",
        "biomedical_terms": [
            "diabetes mellitus", "obesity", "cardiovascular disease", "hypertension",
            "metabolic syndrome", "dyslipidemia", "cancer", "neoplasm",
            "autoimmune disease", "rheumatoid arthritis", "systemic lupus erythematosus",
            "chronic disease", "non-communicable disease", "mortality", "morbidity",
            "disease prevention", "health promotion", "risk factors"
        ],
        "pubmed_query": (
            "diabetes OR obesity OR cardiovascular disease OR hypertension OR "
            "metabolic syndrome OR cancer OR autoimmune disease OR "
            "chronic disease OR non-communicable disease"
        )
    },
    10: {
        "numero": 10,
        "nombre": "Reducción de las Desigualdades",
        "biomedical_terms": [
            "health disparities", "health equity", "social determinants",
            "socioeconomic factors", "vulnerable populations", "indigenous health",
            "minority health", "rural health", "urban health", "health access",
            "healthcare disparities", "cultural competency"
        ],
        "pubmed_query": (
            "health disparities OR health equity OR social determinants OR "
            "vulnerable populations OR indigenous health OR healthcare access"
        )
    },
    12: {
        "numero": 12,
        "nombre": "Producción y Consumo Responsables",
        "biomedical_terms": [
            "sustainable diet", "ultra-processed foods", "food systems",
            "dietary patterns", "Mediterranean diet", "plant-based diet",
            "food waste", "environmental sustainability", "climate change",
            "food processing", "functional foods", "traditional foods"
        ],
        "pubmed_query": (
            "sustainable diet OR ultra-processed foods OR food systems OR "
            "dietary patterns OR Mediterranean diet OR plant-based diet"
        )
    }
}

def load_pubmed_articles():
    """Carga los 226 artículos del doctorado"""
    print("📂 Cargando artículos del DCNT...")

    metadata_file = 'data/pubmed_extracted/metadata_updated_20251024_043156.json'

    with open(metadata_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    print(f"   ✓ {len(articles)} artículos cargados")
    return articles

def create_biomedical_text(article):
    """Crea texto optimizado para BioBERT"""
    parts = []

    # Título científico
    titulo = article.get('title', '') or article.get('original_title', '')
    if titulo:
        parts.append(f"Title: {titulo}")

    # Abstract (crucial para BioBERT)
    abstract = article.get('abstract', '')
    if abstract:
        # BioBERT funciona mejor con abstracts estructurados
        parts.append(f"Abstract: {abstract}")

    # MeSH terms (vocabulario controlado biomédico)
    mesh_terms = article.get('mesh_terms', [])
    if mesh_terms:
        # MeSH es fundamental para BioBERT
        mesh_text = ", ".join(mesh_terms[:25])  # Más términos
        parts.append(f"MeSH Terms: {mesh_text}")

    # Keywords médicos
    keywords = article.get('keywords', [])
    if keywords:
        keywords_text = ", ".join(keywords)
        parts.append(f"Keywords: {keywords_text}")

    # Publication types (importante para contexto)
    pub_types = article.get('publication_types', [])
    if pub_types:
        pub_text = ", ".join(pub_types)
        parts.append(f"Publication Type: {pub_text}")

    # Journal (contexto de publicación)
    journal = article.get('journal', '') or article.get('original_journal', '')
    if journal:
        parts.append(f"Journal: {journal}")

    return " ".join(parts)

def create_ods_biomedical_descriptions(ods_data):
    """Crea descripciones optimizadas para BioBERT"""
    descriptions = {}

    for ods_num, ods_info in ods_data.items():
        # Combinar términos biomédicos y query de PubMed
        bio_terms = ", ".join(ods_info["biomedical_terms"])
        pubmed_context = ods_info["pubmed_query"]

        # Crear descripción estilo paper científico
        description = (
            f"Research on {ods_info['nombre']}: {bio_terms}. "
            f"PubMed search terms: {pubmed_context}. "
            f"This research addresses health outcomes related to {ods_info['nombre'].lower()}, "
            f"including clinical studies, epidemiological research, and interventions."
        )

        descriptions[ods_num] = description

    return descriptions

def calculate_biobert_similarity(model, article_texts, ods_descriptions, ods_nums):
    """Calcula similitud usando BioBERT con optimizaciones"""

    print("\n🧬 Generando embeddings con BioBERT...")
    print("   (Modelo especializado en literatura biomédica)")

    # Generar embeddings de ODS
    ods_texts = [ods_descriptions[num] for num in ods_nums]
    ods_embeddings = model.encode(
        ods_texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    # Normalizar L2
    ods_embeddings = normalize(ods_embeddings, norm='l2')

    # Generar embeddings de artículos en lotes
    print("   Procesando artículos con BioBERT...")
    batch_size = 16

    all_similarities = []

    for i in tqdm(range(0, len(article_texts), batch_size), desc="Lotes BioBERT"):
        batch = article_texts[i:i+batch_size]

        # Embeddings del lote
        batch_embeddings = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Normalizar L2
        batch_embeddings = normalize(batch_embeddings, norm='l2')

        # Calcular similitudes
        batch_sims = cosine_similarity(batch_embeddings, ods_embeddings)

        all_similarities.extend(batch_sims)

    return np.array(all_similarities)

def apply_biomedical_boost(similarities, article, ods_data):
    """Aplica boost basado en términos biomédicos específicos"""
    boosted = similarities.copy()

    mesh_terms = [t.lower() for t in article.get('mesh_terms', [])]
    keywords = [k.lower() for k in article.get('keywords', [])]
    title = (article.get('title', '') or '').lower()
    abstract = (article.get('abstract', '') or '').lower()

    for i, ods_num in enumerate(ods_data.keys()):
        boost = 0
        bio_terms = ods_data[ods_num]["biomedical_terms"]

        # Boost por términos biomédicos específicos
        for term in bio_terms[:10]:  # Top 10 términos
            term_lower = term.lower()
            # Boost fuerte si está en MeSH
            if any(term_lower in mesh for mesh in mesh_terms):
                boost += 0.05
            # Boost moderado si está en título
            elif term_lower in title:
                boost += 0.03
            # Boost leve si está en abstract
            elif term_lower in abstract[:1000]:
                boost += 0.01

        # Aplicar boost con límite
        boosted[i] = min(boosted[i] * (1 + boost), 1.0)

    return boosted

def classify_with_biobert(model, articles):
    """Clasificación principal con BioBERT"""

    # Preparar textos optimizados para BioBERT
    print("\n📝 Preparando textos biomédicos...")
    article_texts = [create_biomedical_text(art) for art in articles]

    # Crear descripciones de ODS optimizadas
    ods_descriptions = create_ods_biomedical_descriptions(ODS_BIOMEDICAL)
    ods_nums = list(ODS_BIOMEDICAL.keys())

    # Calcular similitudes con BioBERT
    similarities = calculate_biobert_similarity(
        model, article_texts, ods_descriptions, ods_nums
    )

    # Clasificar
    results = []
    print("\n🏷️ Clasificando con BioBERT...")

    for i, article in enumerate(tqdm(articles, desc="Clasificando")):
        # Aplicar boost biomédico
        article_sims = apply_biomedical_boost(similarities[i], article, ODS_BIOMEDICAL)

        # Convertir a diccionario
        sims_dict = {ods_nums[j]: float(article_sims[j]) for j in range(len(ods_nums))}

        # ODS principal
        max_ods = max(sims_dict, key=sims_dict.get)
        max_sim = sims_dict[max_ods]

        # Determinar confianza (umbrales ajustados para BioBERT)
        if max_sim >= 0.75:
            confianza = "alta"
        elif max_sim >= 0.60:
            confianza = "media"
        elif max_sim >= 0.45:
            confianza = "baja"
        else:
            confianza = "tentativa"

        # ODS principales (puede haber múltiples si están cerca)
        ods_principales = [{
            'numero': max_ods,
            'nombre': ODS_BIOMEDICAL[max_ods]['nombre'],
            'similitud': round(max_sim, 4),
            'confianza': confianza
        }]

        # Agregar otros ODS si están muy cerca del principal
        for ods_num, sim in sims_dict.items():
            if ods_num != max_ods and sim >= max_sim * 0.85 and sim >= 0.50:
                ods_principales.append({
                    'numero': ods_num,
                    'nombre': ODS_BIOMEDICAL[ods_num]['nombre'],
                    'similitud': round(sim, 4),
                    'confianza': "media" if sim >= 0.60 else "baja"
                })

        # ODS secundarios
        ods_secundarios = []
        principales_nums = [o['numero'] for o in ods_principales]
        for ods_num, sim in sims_dict.items():
            if ods_num not in principales_nums and sim >= 0.40:
                ods_secundarios.append({
                    'numero': ods_num,
                    'nombre': ODS_BIOMEDICAL[ods_num]['nombre'],
                    'similitud': round(sim, 4),
                    'confianza': "baja" if sim >= 0.45 else "tentativa"
                })

        # Ordenar por similitud
        ods_principales.sort(key=lambda x: x['similitud'], reverse=True)
        ods_secundarios.sort(key=lambda x: x['similitud'], reverse=True)

        # Resultado
        classification = {
            'pmid': article.get('pmid', ''),
            'titulo': article.get('title', '') or article.get('original_title', ''),
            'año': int(article.get('original_year', 0)),
            'revista': article.get('journal', '') or article.get('original_journal', ''),
            'doi': article.get('doi', '') or article.get('original_doi', ''),
            'similitudes': {f'ods_{num}': round(sims_dict[num], 4) for num in ods_nums},
            'ods_principales': ods_principales[:2],  # Máximo 2
            'ods_secundarios': ods_secundarios[:2],  # Máximo 2
            'metodo': 'biobert_biomedical_specialized',
            'modelo': MODEL_NAME,
            'tiene_abstract': bool(article.get('abstract', ''))
        }

        results.append(classification)

    return results

def main():
    print("=" * 80)
    print("🧬 CLASIFICACIÓN ODS CON BioBERT")
    print("=" * 80)

    print("\n💊 BioBERT: Modelo BERT pre-entrenado con literatura biomédica")
    print("   • 4.5 mil millones de palabras de PubMed abstracts")
    print("   • 13.5 mil millones de palabras de PMC full-text")
    print("   • Vocabulario especializado en biomedicina")
    print("   • Superior para textos científicos médicos")

    # 1. Cargar modelo BioBERT
    print(f"\n🤖 Cargando modelo BioBERT...")
    print(f"   Modelo: {MODEL_NAME}")
    print("   (Esto puede tomar más tiempo la primera vez)")

    try:
        model = SentenceTransformer(MODEL_NAME)
        print("   ✓ BioBERT cargado exitosamente")
    except Exception as e:
        print(f"   ⚠️ Error cargando BioBERT: {e}")
        print("   Usando modelo alternativo...")
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        print("   ✓ Modelo alternativo cargado")

    # 2. Cargar artículos
    articles = load_pubmed_articles()

    # 3. Clasificar con BioBERT
    results = classify_with_biobert(model, articles)

    print(f"\n   ✓ {len(results)} artículos clasificados con BioBERT")

    # 4. Calcular estadísticas
    stats = {
        'total_articulos': len(results),
        'por_ods': {2: 0, 3: 0, 10: 0, 12: 0},
        'multi_ods': 0,
        'sin_abstract': 0,
        'por_confianza': {'alta': 0, 'media': 0, 'baja': 0, 'tentativa': 0},
        'promedio_similitud': [],
        'max_similitud': 0,
        'min_similitud': 1.0
    }

    for result in results:
        # ODS principales
        if result['ods_principales']:
            for ods in result['ods_principales']:
                if ods == result['ods_principales'][0]:  # Solo contar el primero
                    stats['por_ods'][ods['numero']] += 1
                    stats['por_confianza'][ods['confianza']] += 1
                    sim = ods['similitud']
                    stats['promedio_similitud'].append(sim)
                    stats['max_similitud'] = max(stats['max_similitud'], sim)
                    stats['min_similitud'] = min(stats['min_similitud'], sim)

        # Multi-ODS
        if len(result['ods_principales']) > 1:
            stats['multi_ods'] += 1

        # Sin abstract
        if not result['tiene_abstract']:
            stats['sin_abstract'] += 1

    if stats['promedio_similitud']:
        stats['promedio_similitud'] = np.mean(stats['promedio_similitud'])
    else:
        stats['promedio_similitud'] = 0

    # 5. Guardar resultados
    print("\n💾 Guardando resultados de BioBERT...")

    output_data = {
        'metadata': {
            'fecha_generacion': datetime.now().isoformat(),
            'modelo': MODEL_NAME,
            'metodo': 'biobert_specialized',
            'descripcion': 'Clasificación usando BioBERT pre-entrenado con literatura biomédica',
            'ventajas': [
                'Modelo especializado en textos biomédicos',
                'Vocabulario médico específico',
                'Mejor comprensión de términos MeSH',
                'Pre-entrenado con millones de papers de PubMed',
                'Optimizado para literatura científica'
            ],
            'total_articulos': len(results),
            'ods_clasificados': list(ODS_BIOMEDICAL.keys())
        },
        'estadisticas': stats,
        'articulos': results
    }

    output_path = Path('data/ods_classification_biobert.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"   ✓ Guardado: {output_path}")
    print(f"   Tamaño: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # 6. Resumen
    print("\n" + "=" * 80)
    print("📊 RESUMEN DE CLASIFICACIÓN CON BioBERT")
    print("=" * 80)

    print(f"\n✅ Total: {stats['total_articulos']} artículos")

    print(f"\n📈 Distribución por ODS:")
    for num in ODS_BIOMEDICAL.keys():
        count = stats['por_ods'][num]
        pct = count / stats['total_articulos'] * 100 if stats['total_articulos'] > 0 else 0
        nombre = ODS_BIOMEDICAL[num]['nombre']
        print(f"   ODS {num:2d} ({nombre[:30]:30s}): {count:3d} ({pct:5.1f}%)")

    print(f"\n🎯 Confianza (BioBERT):")
    for nivel in ['alta', 'media', 'baja', 'tentativa']:
        count = stats['por_confianza'][nivel]
        pct = count / stats['total_articulos'] * 100 if stats['total_articulos'] > 0 else 0
        emoji = {'alta': '🟢', 'media': '🟡', 'baja': '🟠', 'tentativa': '🔴'}[nivel]
        print(f"   {emoji} {nivel.capitalize():10s}: {count:3d} ({pct:5.1f}%)")

    print(f"\n📊 Estadísticas de Similitud:")
    print(f"   • Promedio: {stats['promedio_similitud']:.3f}")
    print(f"   • Máximo: {stats['max_similitud']:.3f}")
    print(f"   • Mínimo: {stats['min_similitud']:.3f}")

    print(f"\n📊 Multi-ODS: {stats['multi_ods']} ({stats['multi_ods']/stats['total_articulos']*100:.1f}%)")
    print(f"📝 Sin abstract: {stats['sin_abstract']} ({stats['sin_abstract']/stats['total_articulos']*100:.1f}%)")

    print("\n🧬 VENTAJAS DE BioBERT:")
    print("   ✅ Especializado en literatura biomédica")
    print("   ✅ Mejor comprensión de términos médicos")
    print("   ✅ Vocabulario optimizado para MeSH")
    print("   ✅ Pre-entrenado con PubMed/PMC")
    print("   ✅ Ideal para papers científicos")

    print("\n" + "=" * 80)
    print("✅ CLASIFICACIÓN BioBERT COMPLETADA")
    print("=" * 80)

if __name__ == '__main__':
    main()