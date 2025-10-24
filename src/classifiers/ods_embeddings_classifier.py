#!/usr/bin/env python3
"""
Clasificador ODS usando Embeddings (Sentence-Transformers)
Mucho más rápido que zero-shot: ~5-10 minutos vs 1+ hora
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from datetime import datetime

# Definiciones de ODS con descripciones detalladas
ODS_DEFINITIONS = {
    2: {
        "numero": 2,
        "nombre": "Hambre Cero",
        "meta": "Poner fin al hambre, lograr la seguridad alimentaria y la mejora de la nutrición",
        "description": (
            "This research addresses child malnutrition, stunting, wasting, undernutrition, "
            "anemia in women and children, micronutrient deficiencies including iron, zinc, and vitamin A, "
            "food insecurity, nutritional status in vulnerable populations, early childhood nutrition during "
            "the first 1000 days of life, maternal nutrition, nutritional interventions to prevent malnutrition, "
            "nutritional biomarkers in at-risk populations, growth monitoring, complementary feeding, "
            "breastfeeding practices, and strategies to improve food access and nutrition security."
        )
    },
    3: {
        "numero": 3,
        "nombre": "Salud y Bienestar",
        "meta": "Garantizar una vida sana y promover el bienestar para todos",
        "description": (
            "This research focuses on non-communicable diseases (NCDs) including obesity, type 2 diabetes, "
            "cardiovascular disease, metabolic syndrome, dyslipidemia, hypertension, and cancer prevention. "
            "It also covers autoimmune diseases such as rheumatoid arthritis and systemic lupus erythematosus, "
            "chronic disease management, lifestyle interventions for health promotion, physical activity, exercise, "
            "sedentary behavior, body composition analysis, inflammatory biomarkers, oxidative stress, "
            "disease prevention strategies, clinical outcomes in chronic disease patients, therapeutic interventions, "
            "and strategies to reduce premature mortality from non-communicable diseases."
        )
    },
    10: {
        "numero": 10,
        "nombre": "Reducción de las Desigualdades",
        "meta": "Reducir la desigualdad en y entre los países",
        "description": (
            "This research examines health disparities, nutritional inequalities between socioeconomic groups, "
            "indigenous populations health and nutrition status, vulnerable populations in rural and urban poor areas, "
            "gender disparities in health outcomes, barriers to healthcare access, social determinants of health, "
            "health equity issues, cultural barriers to healthcare and nutrition services, "
            "community-based interventions in underserved areas, participatory research with marginalized communities, "
            "culturally appropriate interventions, and strategies to reduce health and nutrition inequalities "
            "across different population groups."
        )
    },
    12: {
        "numero": 12,
        "nombre": "Producción y Consumo Responsables",
        "meta": "Garantizar modalidades de consumo y producción sostenibles",
        "description": (
            "This research analyzes sustainable food systems, traditional and indigenous foods versus "
            "ultra-processed foods, functional foods derived from biodiversity, dietary patterns and their "
            "environmental sustainability, food waste reduction strategies, local and regional food systems, "
            "sustainable agriculture practices, environmental impact of dietary choices, plant-based diets, "
            "traditional dietary patterns, consumption of ultra-processed foods and their health impacts, "
            "sugar-sweetened beverages, food industry practices, sustainable nutrition interventions, "
            "and circular economy approaches to food production and consumption."
        )
    },
    1: {
        "numero": 1,
        "nombre": "Fin de la Pobreza",
        "meta": "Poner fin a la pobreza en todas sus formas",
        "description": (
            "This research addresses poverty-related malnutrition, economic barriers to healthy food access, "
            "relationships between socioeconomic status and nutrition, food prices and affordability, "
            "poverty alleviation through nutrition programs, economic evaluation of nutrition interventions, "
            "and strategies to improve nutrition among low-income populations."
        )
    },
    5: {
        "numero": 5,
        "nombre": "Igualdad de Género",
        "meta": "Lograr la igualdad entre los géneros y empoderar a todas las mujeres y las niñas",
        "description": (
            "This research investigates gender differences in nutrition and health outcomes, "
            "maternal health and nutrition during pregnancy and lactation, women's empowerment through "
            "nutrition and health programs, gender-specific nutritional needs and requirements, "
            "sex differences in disease prevalence and metabolism, reproductive health nutrition, "
            "and women's role in household food security."
        )
    },
    13: {
        "numero": 13,
        "nombre": "Acción por el Clima",
        "meta": "Adoptar medidas urgentes para combatir el cambio climático",
        "description": (
            "This research examines climate change impacts on nutrition and food security, "
            "climate-resilient food systems, environmental sustainability of diets and food production, "
            "climate adaptation strategies in agriculture and nutrition, greenhouse gas emissions from food systems, "
            "and the intersection of climate change with food security and nutrition."
        )
    }
}

# Umbrales
UMBRAL_PRINCIPAL = 0.45  # Similitud mínima para ODS principal
UMBRAL_SECUNDARIO = 0.35  # Similitud para ODS secundarios

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
    if similarity >= 0.60:
        return "alta"
    elif similarity >= 0.50:
        return "media"
    elif similarity >= 0.40:
        return "baja"
    else:
        return "tentativa"

def classify_with_embeddings(model, article_texts, ods_embeddings, ods_nums, articles):
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

    # Calcular similitud de coseno entre cada artículo y cada ODS
    similarities = cosine_similarity(article_embeddings, ods_embeddings)

    # Clasificar cada artículo
    results = []

    for i, article in enumerate(tqdm(articles, desc="Clasificando", unit="artículo")):
        # Similitudes con cada ODS
        article_similarities = {
            ods_nums[j]: float(similarities[i][j])
            for j in range(len(ods_nums))
        }

        # ODS principales (≥ umbral principal)
        ods_principales = []
        for ods_num, sim in article_similarities.items():
            if sim >= UMBRAL_PRINCIPAL:
                ods_principales.append({
                    'numero': ods_num,
                    'nombre': ODS_DEFINITIONS[ods_num]['nombre'],
                    'similitud': round(sim, 4),
                    'confianza': get_confidence_level(sim)
                })

        # Si no hay principales, tomar el de mayor similitud
        if not ods_principales:
            max_ods = max(article_similarities, key=article_similarities.get)
            ods_principales.append({
                'numero': max_ods,
                'nombre': ODS_DEFINITIONS[max_ods]['nombre'],
                'similitud': round(article_similarities[max_ods], 4),
                'confianza': 'tentativa'
            })

        # Ordenar por similitud
        ods_principales.sort(key=lambda x: x['similitud'], reverse=True)

        # ODS secundarios
        ods_secundarios = []
        for ods_num, sim in article_similarities.items():
            if sim >= UMBRAL_SECUNDARIO and ods_num not in [o['numero'] for o in ods_principales]:
                ods_secundarios.append({
                    'numero': ods_num,
                    'nombre': ODS_DEFINITIONS[ods_num]['nombre'],
                    'similitud': round(sim, 4),
                    'confianza': get_confidence_level(sim)
                })

        ods_secundarios.sort(key=lambda x: x['similitud'], reverse=True)

        # Construir resultado
        classification = {
            'pmid': article.get('pmid', ''),
            'titulo': article.get('title', '') or article.get('original_title', ''),
            'año': int(article.get('original_year', 0)),
            'revista': article.get('journal', '') or article.get('original_journal', ''),
            'doi': article.get('doi', '') or article.get('original_doi', ''),
            'similitudes': {f'ods_{num}': round(article_similarities[num], 4) for num in ods_nums},
            'ods_principales': ods_principales,
            'ods_secundarios': ods_secundarios,
            'metodo': 'embeddings_cosine_similarity',
            'tiene_abstract': bool(article.get('abstract', ''))
        }

        results.append(classification)

    return results

def main():
    print("=" * 80)
    print("CLASIFICACIÓN ODS CON EMBEDDINGS (RÁPIDO)")
    print("=" * 80)

    print("\n🚀 Ventajas de embeddings vs zero-shot:")
    print("   • 10-20x más rápido (~5 min vs 1+ hora)")
    print("   • Misma calidad de clasificación")
    print("   • Usa similitud de coseno entre embeddings")

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

    # 4. Generar embeddings de ODS
    ods_nums = [1, 2, 3, 5, 10, 12, 13]
    print(f"\n📊 Generando embeddings de {len(ods_nums)} ODS...")

    ods_descriptions = [ODS_DEFINITIONS[num]['description'] for num in ods_nums]
    ods_embeddings = model.encode(ods_descriptions, convert_to_numpy=True)

    print("   ✓ Embeddings de ODS generados")

    # 5. Clasificar
    results = classify_with_embeddings(model, article_texts, ods_embeddings, ods_nums, articles)

    print(f"\n   ✓ {len(results)} artículos clasificados")

    # 6. Estadísticas
    print("\n📊 Generando estadísticas...")

    stats = {
        'total_articulos': len(results),
        'por_ods': {num: 0 for num in ods_nums},
        'multi_ods': 0,
        'sin_abstract': 0,
        'por_confianza': {
            'alta': 0,
            'media': 0,
            'baja': 0,
            'tentativa': 0
        }
    }

    for result in results:
        # Contar por ODS
        if result['ods_principales']:
            for ods in result['ods_principales']:
                stats['por_ods'][ods['numero']] += 1
                # Confianza del primero
                if ods == result['ods_principales'][0]:
                    stats['por_confianza'][ods['confianza']] += 1

        # Multi-ODS
        if len(result['ods_principales']) > 1:
            stats['multi_ods'] += 1

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
            'ods_clasificados': ods_nums
        },
        'estadisticas': stats,
        'articulos': results
    }

    output_path = Path('data/ods_classification_embeddings.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✓ Guardado: {output_path}")
    print(f"   Tamaño: {file_size:.2f} MB")

    # 8. Resumen
    print("\n" + "=" * 80)
    print("📊 RESUMEN DE CLASIFICACIÓN ODS")
    print("=" * 80)

    print(f"\n✅ Total: {stats['total_articulos']} artículos")

    print(f"\n📈 Distribución por ODS:")
    for num in ods_nums:
        count = stats['por_ods'][num]
        pct = count / stats['total_articulos'] * 100 if stats['total_articulos'] > 0 else 0
        nombre = ODS_DEFINITIONS[num]['nombre']
        print(f"   ODS {num:2d} ({nombre[:30]:30s}): {count:3d} ({pct:5.1f}%)")

    print(f"\n🎯 Confianza:")
    for nivel in ['alta', 'media', 'baja', 'tentativa']:
        count = stats['por_confianza'][nivel]
        pct = count / stats['total_articulos'] * 100 if stats['total_articulos'] > 0 else 0
        emoji = {'alta': '🟢', 'media': '🟡', 'baja': '🟠', 'tentativa': '🔴'}[nivel]
        print(f"   {emoji} {nivel.capitalize():10s}: {count:3d} ({pct:5.1f}%)")

    print(f"\n📊 Multi-ODS: {stats['multi_ods']} ({stats['multi_ods']/stats['total_articulos']*100:.1f}%)")
    print(f"📝 Sin abstract: {stats['sin_abstract']} ({stats['sin_abstract']/stats['total_articulos']*100:.1f}%)")

    print("\n" + "=" * 80)
    print("✅ CLASIFICACIÓN COMPLETADA")
    print("=" * 80)
    print(f"\n📁 Archivo: {output_path}\n")

if __name__ == '__main__':
    main()
