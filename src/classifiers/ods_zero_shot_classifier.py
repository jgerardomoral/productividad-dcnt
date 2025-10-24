#!/usr/bin/env python3
"""
Clasificador Zero-Shot para ODS usando transformers con metadata completa de PubMed
"""

import json
import torch
import glob
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm
from datetime import datetime

# Definiciones de ODS relevantes para DCNT (hipótesis específicas para Zero-Shot)
ODS_DEFINITIONS = {
    2: {
        "numero": 2,
        "nombre": "Hambre Cero",
        "meta": "Poner fin al hambre, lograr la seguridad alimentaria y la mejora de la nutrición",
        "hypothesis": (
            "This article studies child malnutrition, stunting, wasting, undernutrition, "
            "anemia in women and children, micronutrient deficiencies (iron, zinc, vitamin A), "
            "food insecurity, nutritional status in vulnerable populations, early childhood nutrition, "
            "first 1000 days, maternal nutrition, nutritional interventions to prevent malnutrition, "
            "or nutritional biomarkers in at-risk populations"
        )
    },
    3: {
        "numero": 3,
        "nombre": "Salud y Bienestar",
        "meta": "Garantizar una vida sana y promover el bienestar para todos",
        "hypothesis": (
            "This article studies non-communicable diseases (NCDs) such as obesity, diabetes, "
            "cardiovascular disease, metabolic syndrome, dyslipidemia, hypertension, cancer prevention, "
            "autoimmune diseases (rheumatoid arthritis, systemic lupus erythematosus), "
            "chronic disease management, lifestyle interventions for health, physical activity and exercise, "
            "body composition, inflammatory biomarkers, disease prevention strategies, "
            "or clinical outcomes in chronic disease patients"
        )
    },
    10: {
        "numero": 10,
        "nombre": "Reducción de las Desigualdades",
        "meta": "Reducir la desigualdad en y entre los países",
        "hypothesis": (
            "This article studies health disparities, nutritional inequalities between socioeconomic groups, "
            "indigenous populations health and nutrition, vulnerable populations (rural, low-income), "
            "gender disparities in health, access to healthcare services, social determinants of health, "
            "health equity, cultural barriers to healthcare, community-based interventions in underserved areas, "
            "or interventions to reduce health inequalities"
        )
    },
    12: {
        "numero": 12,
        "nombre": "Producción y Consumo Responsables",
        "meta": "Garantizar modalidades de consumo y producción sostenibles",
        "hypothesis": (
            "This article studies sustainable food systems, traditional foods vs ultra-processed foods, "
            "functional foods from biodiversity, dietary patterns and sustainability, "
            "food waste reduction, local food systems, sustainable agriculture, "
            "environmental impact of diet, plant-based diets, traditional diets, "
            "ultra-processed food consumption, sugar-sweetened beverage impact, "
            "or sustainable nutrition interventions"
        )
    },
    # Incluir otros ODS potencialmente relevantes
    1: {
        "numero": 1,
        "nombre": "Fin de la Pobreza",
        "meta": "Poner fin a la pobreza en todas sus formas",
        "hypothesis": (
            "This article studies poverty-related malnutrition, economic barriers to healthy food access, "
            "socioeconomic status and nutrition, food prices and affordability, "
            "poverty alleviation through nutrition programs"
        )
    },
    5: {
        "numero": 5,
        "nombre": "Igualdad de Género",
        "meta": "Lograr la igualdad entre los géneros y empoderar a todas las mujeres y las niñas",
        "hypothesis": (
            "This article studies gender differences in nutrition and health outcomes, "
            "maternal health and nutrition, women's empowerment through nutrition programs, "
            "gender-specific nutritional needs, pregnancy and lactation nutrition"
        )
    },
    13: {
        "numero": 13,
        "nombre": "Acción por el Clima",
        "meta": "Adoptar medidas urgentes para combatir el cambio climático",
        "hypothesis": (
            "This article studies climate change impact on nutrition and food security, "
            "climate-resilient food systems, environmental sustainability of diets, "
            "climate adaptation in agriculture and nutrition"
        )
    }
}

# Umbrales de clasificación
UMBRAL_PRINCIPAL = 0.30  # Umbral mínimo para ODS principal
UMBRAL_SECUNDARIO = 0.25  # Umbral para ODS secundarios

def load_pubmed_articles():
    """Carga los 226 artículos del doctorado con metadata completa"""
    print("📂 Cargando artículos del DCNT...")

    metadata_file = 'data/pubmed_extracted/metadata_updated_20251024_043156.json'

    with open(metadata_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    print(f"   ✓ {len(articles)} artículos del doctorado cargados")
    return articles

def prepare_text(article):
    """
    Prepara texto para clasificación usando TODA la metadata disponible

    Prioridad:
    1. Título
    2. Abstract (máxima prioridad si existe)
    3. MeSH terms (vocabulario controlado - muy informativo)
    4. Keywords
    5. Publication types
    """
    text_parts = []

    # 1. Título
    titulo = article.get('title', '')
    if titulo:
        text_parts.append(titulo)

    # 2. Abstract (si existe)
    abstract = article.get('abstract', '')
    if abstract:
        text_parts.append(abstract)

    # 3. MeSH terms (siempre - muy informativos)
    mesh_terms = article.get('mesh_terms', [])
    if mesh_terms:
        # Limitar a primeros 15 para no saturar
        mesh_text = "MeSH terms: " + ", ".join(mesh_terms[:15])
        text_parts.append(mesh_text)

    # 4. Keywords
    keywords = article.get('keywords', [])
    if keywords:
        keywords_text = "Keywords: " + ", ".join(keywords[:10])
        text_parts.append(keywords_text)

    # 5. Publication types
    pub_types = article.get('pub_types', [])
    if pub_types:
        pub_types_text = "Publication type: " + ", ".join(pub_types)
        text_parts.append(pub_types_text)

    return " ".join(text_parts)

def get_confidence_level(prob):
    """Determina nivel de confianza según probabilidad"""
    if prob >= 0.50:
        return "alta"
    elif prob >= 0.35:
        return "media"
    elif prob >= 0.25:
        return "baja"
    else:
        return "tentativa"

def classify_article(classifier, article, ods_nums):
    """
    Clasifica un artículo en ODS usando Zero-Shot Classification

    Args:
        classifier: pipeline de zero-shot
        article: diccionario con metadata del artículo
        ods_nums: lista de números de ODS a clasificar

    Returns:
        dict con clasificación completa
    """
    # Preparar texto
    text = prepare_text(article)

    # Hipótesis para clasificación
    hypotheses = [ODS_DEFINITIONS[num]["hypothesis"] for num in ods_nums]

    # Clasificar (multi_label=True permite múltiples ODS)
    result = classifier(text, hypotheses, multi_label=True)

    # Mapear probabilidades a números de ODS
    probs = {ods_nums[i]: result['scores'][i] for i in range(len(ods_nums))}

    # Determinar ODS principales (todos los que superen umbral)
    ods_principales = []
    for ods_num, prob in probs.items():
        if prob >= UMBRAL_PRINCIPAL:
            ods_principales.append({
                'numero': ods_num,
                'nombre': ODS_DEFINITIONS[ods_num]['nombre'],
                'probabilidad': round(prob, 4),
                'confianza': get_confidence_level(prob)
            })

    # Ordenar por probabilidad
    ods_principales.sort(key=lambda x: x['probabilidad'], reverse=True)

    # Si no hay principales, asignar el de mayor probabilidad con confianza baja
    if not ods_principales:
        max_ods = max(probs, key=probs.get)
        ods_principales.append({
            'numero': max_ods,
            'nombre': ODS_DEFINITIONS[max_ods]['nombre'],
            'probabilidad': round(probs[max_ods], 4),
            'confianza': 'tentativa'
        })

    # Determinar ODS secundarios (≥ umbral secundario pero no principales)
    ods_secundarios = []
    for ods_num, prob in probs.items():
        if prob >= UMBRAL_SECUNDARIO and ods_num not in [o['numero'] for o in ods_principales]:
            ods_secundarios.append({
                'numero': ods_num,
                'nombre': ODS_DEFINITIONS[ods_num]['nombre'],
                'probabilidad': round(prob, 4),
                'confianza': get_confidence_level(prob)
            })

    # Ordenar secundarios por probabilidad
    ods_secundarios.sort(key=lambda x: x['probabilidad'], reverse=True)

    # Construir resultado
    classification = {
        'pmid': article.get('pmid', ''),
        'titulo': article.get('title', '') or article.get('original_title', ''),
        'año': int(article.get('original_year', 0)),
        'revista': article.get('journal', '') or article.get('original_journal', ''),
        'doi': article.get('doi', '') or article.get('original_doi', ''),
        'probabilidades': {f'ods_{num}': round(probs[num], 4) for num in ods_nums},
        'ods_principales': ods_principales,
        'ods_secundarios': ods_secundarios,
        'metodo': 'zero_shot_ml',
        'tiene_abstract': bool(article.get('abstract', ''))
    }

    return classification

def main():
    print("=" * 80)
    print("CLASIFICACIÓN ODS CON ZERO-SHOT + METADATA COMPLETA")
    print("=" * 80)

    # 1. Verificar dispositivo
    device = 0 if torch.cuda.is_available() else -1
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n🖥️  Dispositivo: {device_name}")

    if device == -1:
        print("   ⚠️  No GPU detectada, procesamiento será más lento")
    else:
        print("   ✅ GPU detectada, procesamiento será rápido")

    print("\n📋 Metadata utilizada:")
    print("   • Título")
    print("   • Abstract")
    print("   • MeSH terms (vocabulario controlado)")
    print("   • Keywords")
    print("   • Tipo de publicación")

    # 2. Cargar artículos
    articles = load_pubmed_articles()

    # 3. Cargar modelo Zero-Shot
    print("\n🤖 Cargando modelo Zero-Shot...")
    print("   Modelo: facebook/bart-large-mnli")

    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device
    )

    print("   ✓ Modelo cargado")

    # 4. ODS a clasificar
    ods_nums = [1, 2, 3, 5, 10, 12, 13]
    print(f"\n📊 Clasificando en {len(ods_nums)} ODS:")
    for num in ods_nums:
        print(f"   • ODS {num}: {ODS_DEFINITIONS[num]['nombre']}")

    # 5. Clasificar artículos
    print(f"\n🔄 Clasificando {len(articles)} artículos...")

    results = []
    for article in tqdm(articles, desc="Clasificando", unit="artículo"):
        classification = classify_article(classifier, article, ods_nums)
        results.append(classification)

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
        # Contar artículos por ODS (principal)
        if result['ods_principales']:
            for ods in result['ods_principales']:
                stats['por_ods'][ods['numero']] += 1
                # Contar confianza del primer ODS principal
                if ods == result['ods_principales'][0]:
                    stats['por_confianza'][ods['confianza']] += 1

        # Multi-ODS (más de 1 principal)
        if len(result['ods_principales']) > 1:
            stats['multi_ods'] += 1

        # Sin abstract
        if not result['tiene_abstract']:
            stats['sin_abstract'] += 1

    # 7. Guardar resultados
    print("\n💾 Guardando resultados...")

    output_data = {
        'metadata': {
            'fecha_generacion': datetime.now().isoformat(),
            'modelo': 'facebook/bart-large-mnli',
            'metodo': 'zero_shot_classification',
            'dispositivo': device_name,
            'total_articulos': len(results),
            'ods_clasificados': ods_nums
        },
        'estadisticas': stats,
        'articulos': results
    }

    output_path = Path('data/ods_classification_zeroshot.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✓ Resultados guardados: {output_path}")
    print(f"   Tamaño: {file_size:.2f} MB")

    # 8. Resumen
    print("\n" + "=" * 80)
    print("📊 RESUMEN DE CLASIFICACIÓN ODS")
    print("=" * 80)

    print(f"\n✅ Total clasificados: {stats['total_articulos']} artículos")

    print(f"\n📈 Distribución por ODS (principales):")
    for num in ods_nums:
        count = stats['por_ods'][num]
        pct = count / stats['total_articulos'] * 100 if stats['total_articulos'] > 0 else 0
        print(f"   ODS {num:2d} ({ODS_DEFINITIONS[num]['nombre'][:25]}...): {count:3d} artículos ({pct:5.1f}%)")

    print(f"\n🎯 Distribución por confianza:")
    for nivel in ['alta', 'media', 'baja', 'tentativa']:
        count = stats['por_confianza'][nivel]
        pct = count / stats['total_articulos'] * 100 if stats['total_articulos'] > 0 else 0
        emoji = {'alta': '🟢', 'media': '🟡', 'baja': '🟠', 'tentativa': '🔴'}[nivel]
        print(f"   {emoji} {nivel.capitalize():10s}: {count:3d} artículos ({pct:5.1f}%)")

    print(f"\n📊 Multi-ODS: {stats['multi_ods']} artículos ({stats['multi_ods']/stats['total_articulos']*100:.1f}%)")
    print(f"📝 Sin abstract: {stats['sin_abstract']} artículos ({stats['sin_abstract']/stats['total_articulos']*100:.1f}%)")

    print("\n" + "=" * 80)
    print("✅ CLASIFICACIÓN ODS COMPLETADA")
    print("=" * 80)
    print(f"\n📁 Archivo generado: {output_path}")
    print("   Puedes usar este archivo en el dashboard\n")

if __name__ == '__main__':
    main()
