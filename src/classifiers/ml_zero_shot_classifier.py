#!/usr/bin/env python3
"""
Clasificador Zero-Shot usando transformers para clasificar 226 artículos al 100%
"""

import json
import torch
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm
from datetime import datetime

# Definiciones de líneas de investigación (hipótesis MEJORADAS para Zero-Shot)
# Versión 2: Hipótesis más específicas y mutuamente excluyentes
LINEA_DEFINITIONS = {
    1: {
        "nombre": "Bases Moleculares y Genómica Nutricional",
        "hypothesis": (
            "This article studies genetic polymorphisms, single nucleotide polymorphisms (SNPs), "
            "gene-diet interactions, nutrigenetics, nutrigenomics, DNA methylation, "
            "epigenetic modifications, gene expression, genome-wide association studies (GWAS), "
            "or genetic variants specifically related to nutritional metabolism and personalized nutrition"
        )
    },
    2: {
        "nombre": "Alimentación y Nutrición Humana en Salud Pública",
        "hypothesis": (
            "This article studies population-level nutritional interventions, food policy, "
            "food security or food insecurity, community-based nutrition programs, "
            "nutritional surveillance systems, health disparities, epidemiological studies "
            "of dietary patterns in populations, public health campaigns, or nutrition program evaluation"
        )
    },
    3: {
        "nombre": "Alimentación y Nutrición Humana",
        "hypothesis": (
            "This article studies clinical nutrition therapy for individual patients, "
            "dietary counseling, therapeutic diets, functional foods, nutraceuticals, "
            "breastfeeding practices, eating behavior and disorders, food intake assessment methods, "
            "nutritional supplementation, or medical nutrition therapy in clinical settings"
        )
    }
}

# Umbrales de clasificación
UMBRAL_SECUNDARIO = 0.40  # Para líneas secundarias
DIFERENCIA_MAXIMA = 0.15  # Diferencia máxima con principal para ser secundaria

# Niveles de confianza
def get_confidence_level(prob):
    """Determina nivel de confianza según probabilidad"""
    if prob >= 0.65:
        return "alta"
    elif prob >= 0.45:
        return "media"
    elif prob >= 0.30:
        return "baja"
    else:
        return "tentativa"

def load_articles(input_path='data/lineas_classification/classification_input.json'):
    """Carga artículos con metadata completa"""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_text(article):
    """
    Prepara texto para clasificación usando TODA la metadata disponible

    Usa (en orden de prioridad):
    1. Título (siempre)
    2. Abstract (si existe)
    3. MeSH terms (siempre - términos controlados muy informativos)
    4. Keywords (siempre)
    5. Publication types (Clinical Trial, Review, etc.)

    MeSH terms son especialmente útiles porque son vocabulario controlado:
    - "Genome-Wide Association Study" → Línea 1 (Genómica)
    - "Public Health" → Línea 2 (Salud Pública)
    - "Diet Therapy" → Línea 3 (Nutrición Clínica)
    """
    text_parts = []

    # 1. Título (siempre)
    titulo = article.get('titulo', '')
    if titulo:
        text_parts.append(titulo)

    # 2. Abstract (si existe - máxima prioridad)
    if article.get('has_abstract') and article.get('abstract'):
        text_parts.append(article['abstract'])

    # 3. MeSH terms (SIEMPRE - muy informativos)
    # MEJORA: Ya no limitamos a solo cuando no hay abstract
    mesh_terms = article.get('mesh_terms', [])
    if mesh_terms:
        # Limitar a primeros 15 MeSH para no saturar el modelo
        mesh_text = "MeSH terms: " + ", ".join(mesh_terms[:15])
        text_parts.append(mesh_text)

    # 4. Keywords (SIEMPRE)
    keywords = article.get('keywords', [])
    if keywords:
        keywords_text = "Keywords: " + ", ".join(keywords[:10])
        text_parts.append(keywords_text)

    # 5. Publication types (nuevo - contexto importante)
    pub_types = article.get('pub_types', [])
    if pub_types:
        # Tipos útiles: Clinical Trial, Review, Meta-Analysis, etc.
        pub_types_text = "Publication type: " + ", ".join(pub_types)
        text_parts.append(pub_types_text)

    return " ".join(text_parts)

def classify_article(classifier, article):
    """
    Clasifica un artículo usando Zero-Shot Classification

    Returns:
        dict con clasificación completa
    """
    # Preparar texto
    text = prepare_text(article)

    # Hipótesis (labels para Zero-Shot)
    hypotheses = [LINEA_DEFINITIONS[i]["hypothesis"] for i in [1, 2, 3]]

    # Clasificar (multi_label=True permite múltiples líneas)
    result = classifier(text, hypotheses, multi_label=True)

    # Extraer probabilidades
    probs = {
        1: result['scores'][0],  # Línea 1
        2: result['scores'][1],  # Línea 2
        3: result['scores'][2]   # Línea 3
    }

    # Determinar línea principal (máxima probabilidad)
    linea_principal = max(probs, key=probs.get)
    prob_principal = probs[linea_principal]

    # Determinar líneas secundarias
    # Criterio: prob ≥ umbral Y diferencia con principal < 0.15
    lineas_secundarias = []
    for linea, prob in probs.items():
        if linea != linea_principal:
            if prob >= UMBRAL_SECUNDARIO and (prob_principal - prob) <= DIFERENCIA_MAXIMA:
                lineas_secundarias.append({
                    'linea': linea,
                    'probabilidad': round(prob, 4),
                    'confianza': get_confidence_level(prob)
                })

    # Ordenar secundarias por probabilidad
    lineas_secundarias.sort(key=lambda x: x['probabilidad'], reverse=True)

    # Construir resultado
    classification = {
        'pmid': article['pmid'],
        'titulo': article.get('titulo', ''),
        'año': article.get('año', 0),
        'probabilidades': {
            'linea_1': round(probs[1], 4),
            'linea_2': round(probs[2], 4),
            'linea_3': round(probs[3], 4)
        },
        'linea_principal': {
            'linea': linea_principal,
            'nombre': LINEA_DEFINITIONS[linea_principal]['nombre'],
            'probabilidad': round(prob_principal, 4),
            'confianza': get_confidence_level(prob_principal)
        },
        'lineas_secundarias': lineas_secundarias,
        'metodo': 'zero_shot_ml',
        'tiene_abstract': article.get('has_abstract', False)
    }

    return classification

def main():
    print("=" * 80)
    print("CLASIFICACIÓN ML CON ZERO-SHOT + METADATA COMPLETA (100% COBERTURA)")
    print("=" * 80)

    # 1. Verificar dispositivo (GPU/CPU)
    device = 0 if torch.cuda.is_available() else -1
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n🖥️  Dispositivo: {device_name}")

    if device == -1:
        print("   ⚠️  No GPU detectada, procesamiento será más lento (~15-20 min)")
    else:
        print("   ✅ GPU detectada, procesamiento será rápido (~3-5 min)")

    print("\n📋 Metadata utilizada para clasificación:")
    print("   • Título (siempre)")
    print("   • Abstract (si disponible)")
    print("   • MeSH terms (vocabulario controlado)")
    print("   • Keywords del autor")
    print("   • Tipo de publicación (Clinical Trial, Review, etc.)")

    # 2. Cargar artículos
    print("\n📂 Paso 1: Cargando artículos...")
    articles = load_articles()
    print(f"   ✓ {len(articles)} artículos cargados")

    # 3. Cargar modelo Zero-Shot
    print("\n🤖 Paso 2: Cargando modelo Zero-Shot...")
    print("   Modelo: facebook/bart-large-mnli (~380 MB)")
    print("   Descargando... (solo primera vez)")

    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device
    )

    print("   ✓ Modelo cargado exitosamente")

    # 4. Clasificar todos los artículos
    print(f"\n🔄 Paso 3: Clasificando {len(articles)} artículos...")
    print("   Esto tomará varios minutos...")

    results = []

    for article in tqdm(articles, desc="Clasificando", unit="artículo"):
        classification = classify_article(classifier, article)
        results.append(classification)

    print(f"\n   ✓ {len(results)} artículos clasificados (100%)")

    # 5. Estadísticas
    print("\n📊 Paso 4: Generando estadísticas...")

    stats = {
        'total_articulos': len(results),
        'por_linea': {1: 0, 2: 0, 3: 0},
        'por_confianza': {
            'alta': 0,
            'media': 0,
            'baja': 0,
            'tentativa': 0
        },
        'multi_linea': 0,
        'sin_abstract': 0
    }

    for result in results:
        # Línea principal
        linea = result['linea_principal']['linea']
        stats['por_linea'][linea] += 1

        # Confianza
        confianza = result['linea_principal']['confianza']
        stats['por_confianza'][confianza] += 1

        # Multi-línea
        if result['lineas_secundarias']:
            stats['multi_linea'] += 1

        # Sin abstract
        if not result['tiene_abstract']:
            stats['sin_abstract'] += 1

    # 6. Guardar resultados
    print("\n💾 Paso 5: Guardando resultados...")

    output_data = {
        'metadata': {
            'fecha_generacion': datetime.now().isoformat(),
            'modelo': 'facebook/bart-large-mnli',
            'metodo': 'zero_shot_classification',
            'dispositivo': device_name,
            'total_articulos': len(results)
        },
        'estadisticas': stats,
        'clasificaciones': results
    }

    output_path = Path('data/lineas_classification/ml_zero_shot_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    file_size = output_path.stat().st_size / 1024
    print(f"   ✓ Resultados guardados: {output_path}")
    print(f"   Tamaño: {file_size:.1f} KB")

    # 7. Resumen final
    print("\n" + "=" * 80)
    print("📊 RESUMEN DE CLASIFICACIÓN ML")
    print("=" * 80)

    print(f"\n✅ Total clasificados: {stats['total_articulos']} artículos (100%)")

    print(f"\n📈 Distribución por línea (principales):")
    for linea in [1, 2, 3]:
        count = stats['por_linea'][linea]
        pct = count / stats['total_articulos'] * 100
        print(f"   Línea {linea}: {count} artículos ({pct:.1f}%)")

    print(f"\n🎯 Distribución por confianza:")
    for nivel in ['alta', 'media', 'baja', 'tentativa']:
        count = stats['por_confianza'][nivel]
        pct = count / stats['total_articulos'] * 100
        emoji = {'alta': '🟢', 'media': '🟡', 'baja': '🟠', 'tentativa': '🔴'}[nivel]
        print(f"   {emoji} {nivel.capitalize()}: {count} artículos ({pct:.1f}%)")

    print(f"\n📊 Multi-línea: {stats['multi_linea']} artículos ({stats['multi_linea']/stats['total_articulos']*100:.1f}%)")
    print(f"📝 Sin abstract: {stats['sin_abstract']} artículos ({stats['sin_abstract']/stats['total_articulos']*100:.1f}%)")

    print("\n" + "=" * 80)
    print("✅ CLASIFICACIÓN ML COMPLETADA")
    print("=" * 80)

    print(f"\n🎯 PRÓXIMO PASO:")
    print(f"   python3 src/aggregate_ml_classifications.py")
    print(f"   (Fusionar con clasificaciones de agentes)\n")

if __name__ == '__main__':
    main()
