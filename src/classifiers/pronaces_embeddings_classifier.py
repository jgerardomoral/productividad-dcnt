#!/usr/bin/env python3
"""
Clasificador PRONACES usando Embeddings (Sentence-Transformers)
Rápido y preciso: ~5-10 minutos para 226 artículos
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from datetime import datetime

# Definiciones de PRONACES con descripciones detalladas para embeddings
PRONACES_DEFINITIONS = {
    "SALUD": {
        "nombre": "PRONACE Salud",
        "codigo": "SALUD",
        "description": (
            "This research addresses non-communicable chronic diseases (NCDs) including obesity, type 2 diabetes mellitus, "
            "cardiovascular diseases, metabolic syndrome, dyslipidemia, hypertension, cancer prevention and treatment. "
            "It covers autoimmune diseases such as rheumatoid arthritis, systemic lupus erythematosus, multiple sclerosis, "
            "and systemic sclerosis. Research on chronic disease management, lifestyle interventions for health promotion, "
            "physical activity and exercise programs, sedentary behavior, body composition analysis, inflammatory biomarkers, "
            "oxidative stress, disease prevention strategies, clinical outcomes in chronic disease patients, therapeutic interventions, "
            "precision medicine, molecular determinants of disease, biomarker discovery, genomic medicine, systems biology approaches, "
            "data science applied to health, clinical trials, pharmaceutical interventions, and strategies to reduce premature "
            "mortality from non-communicable diseases."
        ),
        "areas_prioritarias": [
            "Alimentación y Salud Integral Comunitaria",
            "Enfermedades Crónicas no Transmisibles",
            "Medicina de Sistemas y Determinantes Moleculares",
            "Ciencia de Datos Aplicada a Salud",
            "Prevención de obesidad y diabetes",
            "Biomarcadores y medicina de precisión"
        ]
    },
    "SOBERANIA_ALIMENTARIA": {
        "nombre": "PRONACE Soberanía Alimentaria",
        "codigo": "SOBERANIA_ALIMENTARIA",
        "description": (
            "This research focuses on food security, food sovereignty, malnutrition, child stunting, wasting, undernutrition, "
            "anemia in women and children, micronutrient deficiencies including iron, zinc, and vitamin A deficiency, "
            "early childhood nutrition during the first 1000 days of life, maternal and infant nutrition, breastfeeding practices, "
            "complementary feeding strategies, nutritional status assessment in vulnerable populations, food insecurity, "
            "nutritional interventions to prevent malnutrition, growth monitoring, sustainable food systems, traditional and "
            "indigenous foods, functional foods from biodiversity, agroecology, regional food circuits, nutritional quality of "
            "traditional crops such as maize and beans, healthy eating education, culturally appropriate nutrition, "
            "community-based food programs, food waste reduction, local and regional food systems, organic agriculture, "
            "sustainable food production, agricultural diversity, native crops, traditional agricultural knowledge, "
            "and strategies to ensure safe, healthy, nutritious and culturally adequate food access."
        ),
        "areas_prioritarias": [
            "Seguridad y soberanía alimentaria",
            "Malnutrición infantil y anemia",
            "Sistemas alimentarios sostenibles",
            "Alimentos tradicionales y funcionales",
            "Educación para alimentación saludable",
            "Circuitos regionales de alimentos",
            "Calidad nutrimental de cultivos tradicionales",
            "Agroecología y producción sustentable"
        ]
    },
    "SISTEMAS_ALIMENTARIOS": {
        "nombre": "Sistemas Alimentarios Sostenibles",
        "codigo": "SISTEMAS_ALIMENTARIOS",
        "description": (
            "This research analyzes sustainable food systems, dietary patterns and their environmental sustainability, "
            "ultra-processed foods consumption and health impacts, sugar-sweetened beverages, food industry practices, "
            "front-of-package labeling, food marketing and its effects on health, food environments, nutrition transitions, "
            "traditional diets versus Western diets, plant-based diets, Mediterranean diet, environmental impact of dietary choices, "
            "carbon footprint of foods, sustainable nutrition interventions, circular economy in food production, "
            "food production and consumption patterns, food waste at consumer and industrial levels, responsible consumption, "
            "sustainable agriculture practices, climate change impacts on food systems, climate-resilient agriculture, "
            "food quality and safety, food processing technologies, food innovation, bioactive compounds in foods, "
            "functional foods development, nutritional epidemiology of dietary patterns, and public policies for "
            "healthy and sustainable food systems."
        ),
        "areas_prioritarias": [
            "Consumo de alimentos ultraprocesados",
            "Patrones dietéticos sostenibles",
            "Etiquetado frontal de alimentos",
            "Ambientes alimentarios saludables",
            "Impacto ambiental de la dieta",
            "Innovación en alimentos funcionales",
            "Políticas públicas alimentarias"
        ]
    }
}

# Umbrales
UMBRAL_PRINCIPAL = 0.40  # Similitud mínima para PRONACE principal
UMBRAL_SECUNDARIO = 0.30  # Similitud para PRONACES secundarios

def load_pubmed_articles():
    """Carga los 226 artículos del doctorado"""
    print("📂 Cargando artículos del DCNT...")

    metadata_file = 'data/pubmed_extracted/metadata_updated_20251024_043156.json'

    with open(metadata_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    print(f"   ✓ {len(articles)} artículos cargados")
    return articles

def prepare_text(article):
    """Prepara texto usando toda la metadata"""
    text_parts = []

    # Título
    titulo = article.get('title', '') or article.get('original_title', '')
    if titulo:
        text_parts.append(titulo)

    # Abstract (prioritario)
    abstract = article.get('abstract', '')
    if abstract:
        text_parts.append(abstract)

    # MeSH terms
    mesh_terms = article.get('mesh_terms', [])
    if mesh_terms:
        mesh_text = " ".join(mesh_terms[:15])
        text_parts.append(mesh_text)

    # Keywords
    keywords = article.get('keywords', [])
    if keywords:
        keywords_text = " ".join(keywords[:10])
        text_parts.append(keywords_text)

    return " ".join(text_parts)

def get_confidence_level(similarity):
    """Determina confianza basada en similitud de coseno"""
    if similarity >= 0.55:
        return "alta"
    elif similarity >= 0.45:
        return "media"
    elif similarity >= 0.35:
        return "baja"
    else:
        return "tentativa"

def classify_with_embeddings(model, article_texts, pronaces_embeddings, pronaces_codes, articles):
    """Clasifica artículos usando similitud de embeddings"""

    print("\n🔄 Generando embeddings de artículos...")
    print("   (Esto toma ~2-3 minutos)")

    # Generar embeddings de artículos (batch para eficiencia)
    article_embeddings = model.encode(
        article_texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )

    print("\n📊 Calculando similitudes...")

    # Calcular similitud de coseno entre cada artículo y cada PRONACE
    similarities = cosine_similarity(article_embeddings, pronaces_embeddings)

    # Clasificar cada artículo
    results = []

    for i, article in enumerate(tqdm(articles, desc="Clasificando", unit="artículo")):
        # Similitudes con cada PRONACE
        article_similarities = {
            pronaces_codes[j]: float(similarities[i][j])
            for j in range(len(pronaces_codes))
        }

        # PRONACE principal (máxima similitud)
        principal_code = max(article_similarities, key=article_similarities.get)
        principal_sim = article_similarities[principal_code]

        # Solo asignar si supera umbral mínimo
        pronaces_principales = []
        if principal_sim >= UMBRAL_PRINCIPAL:
            pronaces_principales.append({
                'codigo': principal_code,
                'nombre': PRONACES_DEFINITIONS[principal_code]['nombre'],
                'similitud': round(principal_sim, 4),
                'confianza': get_confidence_level(principal_sim)
            })
        else:
            # Si no supera umbral, asignar el mejor con confianza tentativa
            pronaces_principales.append({
                'codigo': principal_code,
                'nombre': PRONACES_DEFINITIONS[principal_code]['nombre'],
                'similitud': round(principal_sim, 4),
                'confianza': 'tentativa'
            })

        # PRONACES secundarios
        pronaces_secundarios = []
        for code, sim in article_similarities.items():
            if code != principal_code and sim >= UMBRAL_SECUNDARIO:
                pronaces_secundarios.append({
                    'codigo': code,
                    'nombre': PRONACES_DEFINITIONS[code]['nombre'],
                    'similitud': round(sim, 4),
                    'confianza': get_confidence_level(sim)
                })

        pronaces_secundarios.sort(key=lambda x: x['similitud'], reverse=True)

        # Construir resultado
        classification = {
            'pmid': article.get('pmid', ''),
            'titulo': article.get('title', '') or article.get('original_title', ''),
            'año': int(article.get('original_year', 0)),
            'revista': article.get('journal', '') or article.get('original_journal', ''),
            'doi': article.get('doi', '') or article.get('original_doi', ''),
            'similitudes': {code: round(article_similarities[code], 4) for code in pronaces_codes},
            'pronaces_principales': pronaces_principales,
            'pronaces_secundarios': pronaces_secundarios,
            'metodo': 'embeddings_cosine_similarity',
            'tiene_abstract': bool(article.get('abstract', ''))
        }

        results.append(classification)

    return results

def main():
    print("=" * 80)
    print("CLASIFICACIÓN PRONACES CON EMBEDDINGS (RÁPIDO)")
    print("=" * 80)

    print("\n🚀 Método: Embeddings con similitud de coseno")
    print("   • Rápido: ~5 minutos vs 1+ hora con zero-shot")
    print("   • Preciso: Usa toda la metadata de PubMed")
    print("   • Reproducible: Mismo resultado cada vez")

    print("\n📋 Metadata utilizada:")
    print("   • Título")
    print("   • Abstract completo")
    print("   • MeSH terms")
    print("   • Keywords")

    # 1. Cargar modelo de embeddings
    print("\n🤖 Cargando modelo de embeddings...")
    print("   Modelo: all-MiniLM-L6-v2 (ligero y eficiente)")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("   ✓ Modelo cargado")

    # 2. Cargar artículos
    articles = load_pubmed_articles()

    # 3. Preparar textos de artículos
    print("\n📝 Preparando textos de artículos...")
    article_texts = [prepare_text(article) for article in articles]
    print(f"   ✓ {len(article_texts)} textos preparados")

    # 4. Generar embeddings de PRONACES
    pronaces_codes = ["SALUD", "SOBERANIA_ALIMENTARIA", "SISTEMAS_ALIMENTARIOS"]
    print(f"\n📊 Generando embeddings de {len(pronaces_codes)} PRONACES...")
    print("   • SALUD")
    print("   • SOBERANIA_ALIMENTARIA")
    print("   • SISTEMAS_ALIMENTARIOS")

    pronaces_descriptions = [PRONACES_DEFINITIONS[code]['description'] for code in pronaces_codes]
    pronaces_embeddings = model.encode(pronaces_descriptions, convert_to_numpy=True)

    print("   ✓ Embeddings de PRONACES generados")

    # 5. Clasificar
    results = classify_with_embeddings(model, article_texts, pronaces_embeddings, pronaces_codes, articles)

    print(f"\n   ✓ {len(results)} artículos clasificados")

    # 6. Estadísticas
    print("\n📊 Generando estadísticas...")

    stats = {
        'total_articulos': len(results),
        'por_pronace': {code: 0 for code in pronaces_codes},
        'multi_pronace': 0,
        'sin_abstract': 0,
        'por_confianza': {
            'alta': 0,
            'media': 0,
            'baja': 0,
            'tentativa': 0
        }
    }

    for result in results:
        # Contar por PRONACE
        if result['pronaces_principales']:
            for pronace in result['pronaces_principales']:
                stats['por_pronace'][pronace['codigo']] += 1
                # Confianza del primero
                if pronace == result['pronaces_principales'][0]:
                    stats['por_confianza'][pronace['confianza']] += 1

        # Multi-PRONACE
        if len(result['pronaces_principales']) + len(result['pronaces_secundarios']) > 1:
            stats['multi_pronace'] += 1

        # Sin abstract
        if not result['tiene_abstract']:
            stats['sin_abstract'] += 1

    # 7. Guardar
    print("\n💾 Guardando resultados...")

    output_data = {
        'metadata': {
            'fecha_generacion': datetime.now().isoformat(),
            'modelo': 'sentence-transformers/all-MiniLM-L6-v2',
            'metodo': 'embeddings_cosine_similarity',
            'total_articulos': len(results),
            'pronaces_clasificados': pronaces_codes
        },
        'estadisticas': stats,
        'articulos': results
    }

    output_path = Path('data/pronaces_classification_embeddings.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✓ Guardado: {output_path}")
    print(f"   Tamaño: {file_size:.2f} MB")

    # 8. Resumen
    print("\n" + "=" * 80)
    print("📊 RESUMEN DE CLASIFICACIÓN PRONACES")
    print("=" * 80)

    print(f"\n✅ Total: {stats['total_articulos']} artículos")

    print(f"\n📈 Distribución por PRONACE:")
    for code in pronaces_codes:
        count = stats['por_pronace'][code]
        pct = count / stats['total_articulos'] * 100 if stats['total_articulos'] > 0 else 0
        nombre = PRONACES_DEFINITIONS[code]['nombre']
        print(f"   {nombre:40s}: {count:3d} ({pct:5.1f}%)")

    print(f"\n🎯 Confianza:")
    for nivel in ['alta', 'media', 'baja', 'tentativa']:
        count = stats['por_confianza'][nivel]
        pct = count / stats['total_articulos'] * 100 if stats['total_articulos'] > 0 else 0
        emoji = {'alta': '🟢', 'media': '🟡', 'baja': '🟠', 'tentativa': '🔴'}[nivel]
        print(f"   {emoji} {nivel.capitalize():10s}: {count:3d} ({pct:5.1f}%)")

    print(f"\n📊 Multi-PRONACE: {stats['multi_pronace']} ({stats['multi_pronace']/stats['total_articulos']*100:.1f}%)")
    print(f"📝 Sin abstract: {stats['sin_abstract']} ({stats['sin_abstract']/stats['total_articulos']*100:.1f}%)")

    print("\n" + "=" * 80)
    print("✅ CLASIFICACIÓN COMPLETADA")
    print("=" * 80)
    print(f"\n📁 Archivo: {output_path}\n")

if __name__ == '__main__':
    main()
