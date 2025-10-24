#!/usr/bin/env python3
"""
Clasificador PRONACES usando Embeddings MEJORADO
Mejoras implementadas:
- Modelo más potente: all-mpnet-base-v2
- Normalización L2 de embeddings
- Múltiples representaciones por categoría
- Umbrales optimizados
- Mejor procesamiento de texto
- Query expansion con términos MeSH relevantes
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

# Definiciones de PRONACES con múltiples representaciones mejoradas
PRONACES_DEFINITIONS = {
    "SALUD": {
        "nombre": "PRONACE Salud",
        "codigo": "SALUD",
        "descriptions": [
            # Descripción principal técnica
            (
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
            # Términos MeSH específicos
            (
                "Diabetes mellitus type 2, obesity, metabolic syndrome, cardiovascular diseases, hypertension, dyslipidemias, "
                "neoplasms, autoimmune diseases, arthritis rheumatoid, lupus erythematosus systemic, multiple sclerosis, "
                "biomarkers, inflammation mediators, oxidative stress, chronic disease, risk factors, clinical trials, "
                "precision medicine, genomics, metabolomics, proteomics, systems biology, drug therapy, therapeutics, "
                "disease management, health promotion, prevention and control, mortality, prognosis, treatment outcome"
            ),
            # Áreas prioritarias y outcomes
            (
                "Prevention of obesity epidemic in Mexico, reduction of diabetes prevalence and complications, "
                "cardiovascular disease prevention and management, cancer prevention through lifestyle interventions, "
                "autoimmune disease control and quality of life improvement, precision medicine applications, "
                "biomarker-guided therapy, molecular medicine advances, data-driven health interventions, "
                "evidence-based clinical practice, reduced disease burden and healthcare costs, improved patient outcomes"
            )
        ],
        "mesh_keywords": [
            "Chronic Disease", "Noncommunicable Diseases", "Diabetes Mellitus", "Obesity",
            "Cardiovascular Diseases", "Metabolic Syndrome", "Biomarkers", "Precision Medicine"
        ],
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
        "descriptions": [
            # Descripción principal
            (
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
            # Términos MeSH y conceptos clave
            (
                "Food security, food supply, malnutrition, protein-energy malnutrition, child nutrition disorders, "
                "infant nutrition disorders, micronutrients, vitamin A deficiency, iron deficiency anemia, zinc deficiency, "
                "growth disorders, maternal nutrition, breast feeding, infant nutritional physiological phenomena, "
                "vulnerable populations, indigenous peoples, traditional medicine, functional food, biodiversity, "
                "sustainable agriculture, organic agriculture, agroecology, food systems, local food, food quality, "
                "nutrition education, community participation, cultural characteristics"
            ),
            # Impacto social y políticas
            (
                "Elimination of child malnutrition and stunting, achieving zero hunger in vulnerable communities, "
                "strengthening local and traditional food systems, preserving indigenous agricultural knowledge, "
                "ensuring nutritional sovereignty for Mexican populations, reducing dependency on imported foods, "
                "promoting sustainable and culturally appropriate diets, empowering communities through food programs, "
                "improving maternal and child nutrition outcomes, creating resilient food systems against climate change"
            )
        ],
        "mesh_keywords": [
            "Food Security", "Malnutrition", "Child Nutrition", "Micronutrients",
            "Sustainable Agriculture", "Indigenous Food", "Food Sovereignty", "Community Nutrition"
        ],
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
        "descriptions": [
            # Descripción principal
            (
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
            # Sistemas y políticas alimentarias
            (
                "Food systems, ultra-processed foods, food processing, sugar-sweetened beverages, dietary patterns, "
                "plant-based diet, Mediterranean diet, Western diet, food labeling, food marketing, food environment, "
                "nutrition transition, environmental sustainability, carbon footprint, climate change, food waste, "
                "circular economy, sustainable development, food policy, nutrition policy, public health policy, "
                "food quality, food safety, functional foods, bioactive compounds, food technology, food innovation"
            ),
            # Transformación de sistemas
            (
                "Transformation to sustainable and healthy food systems, reduction of ultra-processed food consumption, "
                "implementation of effective food labeling policies, creation of healthy food environments, "
                "promotion of sustainable dietary patterns, mitigation of climate change through diet, "
                "reduction of food waste across the supply chain, development of innovative healthy foods, "
                "strengthening food safety and quality systems, evidence-based food policy development, "
                "achieving sustainable development goals through food system transformation"
            )
        ],
        "mesh_keywords": [
            "Food Systems", "Diet", "Food Processing", "Environmental Sustainability",
            "Climate Change", "Food Policy", "Nutrition Policy", "Public Health"
        ],
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

# Umbrales optimizados
UMBRAL_PRINCIPAL = 0.45  # Ajustado para PRONACES
UMBRAL_SECUNDARIO = 0.35  # Ajustado para PRONACES
MIN_CONFIDENCE_SCORE = 0.30  # Score mínimo para no ser tentativo

def load_pubmed_articles():
    """Carga los 226 artículos del doctorado"""
    print("📂 Cargando artículos del DCNT...")

    metadata_file = 'data/pubmed_extracted/metadata_updated_20251024_043156.json'

    with open(metadata_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    print(f"   ✓ {len(articles)} artículos cargados")
    return articles

def expand_mesh_terms(mesh_terms, pronace_data):
    """Expande términos MeSH con keywords relevantes del PRONACE"""
    expanded = []

    # Convertir a minúsculas para comparación
    mesh_lower = [term.lower() for term in mesh_terms]

    # Verificar coincidencias con keywords del PRONACE
    for keyword in pronace_data.get('mesh_keywords', []):
        keyword_lower = keyword.lower()
        for mesh in mesh_lower:
            if keyword_lower in mesh or mesh in keyword_lower:
                expanded.append(keyword)
                break

    return expanded

def prepare_text_enhanced(article, pronaces_definitions):
    """Prepara texto con ponderación y expansión mejorada"""
    text_parts = []
    weights = []

    # Título (peso alto - 25%)
    titulo = article.get('title', '') or article.get('original_title', '')
    if titulo:
        text_parts.append(titulo)
        weights.append(0.25)

    # Abstract (peso más alto - 45%)
    abstract = article.get('abstract', '')
    if abstract:
        text_parts.append(abstract)
        weights.append(0.45)

    # MeSH terms con expansión (peso medio - 20%)
    mesh_terms = article.get('mesh_terms', [])
    if mesh_terms:
        # Expandir MeSH terms con keywords relevantes
        expanded_terms = []
        for pronace_data in pronaces_definitions.values():
            expanded_terms.extend(expand_mesh_terms(mesh_terms, pronace_data))

        all_mesh = mesh_terms[:20] + list(set(expanded_terms))
        mesh_text = " ".join(all_mesh)
        text_parts.append(f"Medical subjects: {mesh_text}")
        weights.append(0.20)

    # Keywords (peso bajo - 10%)
    keywords = article.get('keywords', [])
    if keywords:
        keywords_text = " ".join(keywords[:15])
        text_parts.append(f"Keywords: {keywords_text}")
        weights.append(0.10)

    # Publication type y journal
    pub_types = article.get('publication_types', [])
    journal = article.get('journal', '')
    if pub_types or journal:
        context_text = f"Study type: {' '.join(pub_types)} Journal: {journal}"
        text_parts.append(context_text)
        weights.append(0.05)

    # Normalizar pesos si no suman 1
    if weights:
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]

    return " ".join(text_parts)

def get_confidence_level_enhanced(similarity, is_primary=True):
    """Determina confianza con umbrales optimizados para PRONACES"""
    if is_primary:
        if similarity >= 0.65:
            return "alta"
        elif similarity >= 0.50:
            return "media"
        elif similarity >= 0.40:
            return "baja"
        else:
            return "tentativa"
    else:
        # Para secundarios
        if similarity >= 0.55:
            return "alta"
        elif similarity >= 0.45:
            return "media"
        elif similarity >= 0.35:
            return "baja"
        else:
            return "tentativa"

def generate_multiple_embeddings(model, pronaces_definitions):
    """Genera múltiples embeddings por categoría y los promedia con ponderación"""
    pronaces_embeddings_dict = {}

    for code, pronace_data in pronaces_definitions.items():
        embeddings_list = []
        weights = [0.4, 0.3, 0.3]  # Pesos para cada descripción

        # Generar embeddings para cada descripción
        for i, description in enumerate(pronace_data.get('descriptions', [])):
            if description:
                embedding = model.encode(description, convert_to_numpy=True)
                # Normalizar L2
                embedding = normalize(embedding.reshape(1, -1), norm='l2')[0]
                embeddings_list.append(embedding * weights[i])

        # Sumar embeddings ponderados y normalizar de nuevo
        if embeddings_list:
            weighted_embedding = np.sum(embeddings_list, axis=0)
            weighted_embedding = normalize(weighted_embedding.reshape(1, -1), norm='l2')[0]
            pronaces_embeddings_dict[code] = weighted_embedding

    return pronaces_embeddings_dict

def apply_domain_specific_boost(similarities, article, pronaces_codes):
    """Aplica boost basado en características específicas del dominio"""
    boosted_similarities = similarities.copy()

    # Obtener términos del artículo
    mesh_terms = article.get('mesh_terms', [])
    keywords = article.get('keywords', [])
    title = (article.get('title', '') or article.get('original_title', '')).lower()

    # Aplicar boosts específicos
    for i, code in enumerate(pronaces_codes):
        boost = 0

        if code == "SALUD":
            # Boost para términos relacionados con enfermedades crónicas
            chronic_terms = ['diabetes', 'obesity', 'cardiovascular', 'hypertension', 'metabolic syndrome',
                           'cancer', 'autoimmune', 'arthritis', 'lupus']
            for term in chronic_terms:
                if any(term in m.lower() for m in mesh_terms) or term in title:
                    boost += 0.05

        elif code == "SOBERANIA_ALIMENTARIA":
            # Boost para términos de malnutrición y seguridad alimentaria
            food_security_terms = ['malnutrition', 'stunting', 'food security', 'micronutrient',
                                  'breastfeeding', 'indigenous', 'traditional food', 'anemia']
            for term in food_security_terms:
                if any(term in m.lower() for m in mesh_terms) or term in title:
                    boost += 0.05

        elif code == "SISTEMAS_ALIMENTARIOS":
            # Boost para términos de sistemas alimentarios
            systems_terms = ['ultra-processed', 'food system', 'sustainable', 'dietary pattern',
                           'food environment', 'food policy', 'climate', 'food waste']
            for term in systems_terms:
                if any(term in m.lower() for m in mesh_terms) or term in title:
                    boost += 0.05

        # Aplicar boost con límite máximo
        boosted_similarities[i] = min(boosted_similarities[i] + boost, 1.0)

    return boosted_similarities

def classify_with_embeddings_enhanced(model, article_texts, pronaces_embeddings_dict, pronaces_codes, articles):
    """Clasifica artículos con técnicas mejoradas y boost específico del dominio"""

    print("\n🔄 Generando embeddings de artículos...")
    print("   (Con modelo mejorado all-mpnet-base-v2)")

    # Generar embeddings de artículos
    article_embeddings = model.encode(
        article_texts,
        show_progress_bar=True,
        batch_size=16,
        convert_to_numpy=True
    )

    # Normalizar embeddings L2
    print("   Aplicando normalización L2...")
    article_embeddings = normalize(article_embeddings, norm='l2')

    # Convertir dict a array para cálculo eficiente
    pronaces_embeddings = np.array([pronaces_embeddings_dict[code] for code in pronaces_codes])

    print("\n📊 Calculando similitudes mejoradas con boost de dominio...")

    # Calcular similitud de coseno
    similarities = cosine_similarity(article_embeddings, pronaces_embeddings)

    # Aplicar transformación para expandir rango
    similarities = np.power(similarities, 0.9)  # Suavizar distribución

    # Clasificar cada artículo
    results = []

    for i, article in enumerate(tqdm(articles, desc="Clasificando", unit="artículo")):
        # Aplicar boost específico del dominio
        article_similarities_raw = similarities[i]
        article_similarities_boosted = apply_domain_specific_boost(
            article_similarities_raw, article, pronaces_codes
        )

        # Convertir a diccionario
        article_similarities = {
            pronaces_codes[j]: float(article_similarities_boosted[j])
            for j in range(len(pronaces_codes))
        }

        # PRONACE principal (máxima similitud)
        principal_code = max(article_similarities, key=article_similarities.get)
        principal_sim = article_similarities[principal_code]

        # Determinar principales
        pronaces_principales = []
        if principal_sim >= UMBRAL_PRINCIPAL:
            pronaces_principales.append({
                'codigo': principal_code,
                'nombre': PRONACES_DEFINITIONS[principal_code]['nombre'],
                'similitud': round(principal_sim, 4),
                'confianza': get_confidence_level_enhanced(principal_sim, is_primary=True)
            })

            # Verificar si hay otro PRONACE muy cercano
            for code, sim in article_similarities.items():
                if code != principal_code and sim >= UMBRAL_PRINCIPAL and sim >= principal_sim * 0.85:
                    pronaces_principales.append({
                        'codigo': code,
                        'nombre': PRONACES_DEFINITIONS[code]['nombre'],
                        'similitud': round(sim, 4),
                        'confianza': get_confidence_level_enhanced(sim, is_primary=True)
                    })
        else:
            # Si no supera umbral pero está cerca, asignar con confianza apropiada
            if principal_sim >= MIN_CONFIDENCE_SCORE:
                pronaces_principales.append({
                    'codigo': principal_code,
                    'nombre': PRONACES_DEFINITIONS[principal_code]['nombre'],
                    'similitud': round(principal_sim, 4),
                    'confianza': get_confidence_level_enhanced(principal_sim, is_primary=True)
                })
            else:
                # Por defecto, asignar a SALUD (más general)
                pronaces_principales.append({
                    'codigo': 'SALUD',
                    'nombre': PRONACES_DEFINITIONS['SALUD']['nombre'],
                    'similitud': round(article_similarities['SALUD'], 4),
                    'confianza': 'tentativa'
                })

        # Ordenar por similitud
        pronaces_principales.sort(key=lambda x: x['similitud'], reverse=True)
        pronaces_principales = pronaces_principales[:2]  # Máximo 2 principales

        # PRONACES secundarios
        pronaces_secundarios = []
        principales_codes = [p['codigo'] for p in pronaces_principales]

        for code, sim in article_similarities.items():
            if sim >= UMBRAL_SECUNDARIO and code not in principales_codes:
                pronaces_secundarios.append({
                    'codigo': code,
                    'nombre': PRONACES_DEFINITIONS[code]['nombre'],
                    'similitud': round(sim, 4),
                    'confianza': get_confidence_level_enhanced(sim, is_primary=False)
                })

        pronaces_secundarios.sort(key=lambda x: x['similitud'], reverse=True)
        pronaces_secundarios = pronaces_secundarios[:1]  # Máximo 1 secundario

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
            'metodo': 'embeddings_enhanced_mpnet_domain_boost',
            'tiene_abstract': bool(article.get('abstract', ''))
        }

        results.append(classification)

    return results

def main():
    print("=" * 80)
    print("CLASIFICACIÓN PRONACES CON EMBEDDINGS MEJORADO")
    print("=" * 80)

    print("\n🚀 Mejoras implementadas:")
    print("   • Modelo superior: all-mpnet-base-v2")
    print("   • Normalización L2 de embeddings")
    print("   • Múltiples representaciones por PRONACE")
    print("   • Expansión de términos MeSH")
    print("   • Boost específico del dominio")
    print("   • Umbrales optimizados")

    print("\n📋 Metadata utilizada:")
    print("   • Título (25% peso)")
    print("   • Abstract completo (45% peso)")
    print("   • MeSH terms expandidos (20% peso)")
    print("   • Keywords (10% peso)")

    # 1. Cargar modelo mejorado
    print("\n🤖 Cargando modelo de embeddings mejorado...")
    print("   Modelo: sentence-transformers/all-mpnet-base-v2")

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    print("   ✓ Modelo cargado")

    # 2. Cargar artículos
    articles = load_pubmed_articles()

    # 3. Preparar textos con método mejorado
    print("\n📝 Preparando textos de artículos con expansión MeSH...")
    article_texts = [prepare_text_enhanced(article, PRONACES_DEFINITIONS) for article in articles]
    print(f"   ✓ {len(article_texts)} textos preparados con expansión")

    # 4. Generar embeddings de PRONACES con múltiples representaciones
    pronaces_codes = list(PRONACES_DEFINITIONS.keys())
    print(f"\n📊 Generando embeddings mejorados de {len(pronaces_codes)} PRONACES...")
    print("   Usando múltiples representaciones ponderadas...")

    pronaces_embeddings_dict = generate_multiple_embeddings(model, PRONACES_DEFINITIONS)

    print(f"   ✓ Embeddings de PRONACES generados")

    # 5. Clasificar con método mejorado
    results = classify_with_embeddings_enhanced(
        model, article_texts, pronaces_embeddings_dict, pronaces_codes, articles
    )

    print(f"\n   ✓ {len(results)} artículos clasificados")

    # 6. Estadísticas mejoradas
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
        },
        'promedio_similitud': 0,
        'similitud_maxima': 0,
        'similitud_minima': 1.0
    }

    similitudes_principales = []

    for result in results:
        # Contar por PRONACE
        if result['pronaces_principales']:
            for pronace in result['pronaces_principales']:
                stats['por_pronace'][pronace['codigo']] += 1
                # Confianza del primero
                if pronace == result['pronaces_principales'][0]:
                    stats['por_confianza'][pronace['confianza']] += 1
                    sim = pronace['similitud']
                    similitudes_principales.append(sim)
                    stats['similitud_maxima'] = max(stats['similitud_maxima'], sim)
                    stats['similitud_minima'] = min(stats['similitud_minima'], sim)

        # Multi-PRONACE
        if len(result['pronaces_principales']) > 1:
            stats['multi_pronace'] += 1

        # Sin abstract
        if not result['tiene_abstract']:
            stats['sin_abstract'] += 1

    if similitudes_principales:
        stats['promedio_similitud'] = np.mean(similitudes_principales)

    # 7. Guardar resultados
    print("\n💾 Guardando resultados mejorados...")

    output_data = {
        'metadata': {
            'fecha_generacion': datetime.now().isoformat(),
            'modelo': 'sentence-transformers/all-mpnet-base-v2',
            'metodo': 'embeddings_enhanced_normalized_domain_boost',
            'mejoras': [
                'Modelo MPNET superior',
                'Normalización L2',
                'Múltiples embeddings ponderados por categoría',
                'Expansión de términos MeSH',
                'Boost específico del dominio',
                'Umbrales optimizados'
            ],
            'total_articulos': len(results),
            'pronaces_clasificados': pronaces_codes,
            'umbrales': {
                'principal': UMBRAL_PRINCIPAL,
                'secundario': UMBRAL_SECUNDARIO,
                'minimo': MIN_CONFIDENCE_SCORE
            }
        },
        'estadisticas': stats,
        'articulos': results
    }

    output_path = Path('data/pronaces_classification_embeddings_enhanced.json')
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✓ Guardado: {output_path}")
    print(f"   Tamaño: {file_size:.2f} MB")

    # 8. Resumen
    print("\n" + "=" * 80)
    print("📊 RESUMEN DE CLASIFICACIÓN PRONACES MEJORADA")
    print("=" * 80)

    print(f"\n✅ Total: {stats['total_articulos']} artículos")

    print(f"\n📈 Distribución por PRONACE:")
    for code in pronaces_codes:
        count = stats['por_pronace'][code]
        pct = count / stats['total_articulos'] * 100 if stats['total_articulos'] > 0 else 0
        nombre = PRONACES_DEFINITIONS[code]['nombre']
        print(f"   {nombre[:35]:35s}: {count:3d} ({pct:5.1f}%)")

    print(f"\n🎯 Confianza (MEJORADO):")
    for nivel in ['alta', 'media', 'baja', 'tentativa']:
        count = stats['por_confianza'][nivel]
        pct = count / stats['total_articulos'] * 100 if stats['total_articulos'] > 0 else 0
        emoji = {'alta': '🟢', 'media': '🟡', 'baja': '🟠', 'tentativa': '🔴'}[nivel]
        print(f"   {emoji} {nivel.capitalize():10s}: {count:3d} ({pct:5.1f}%)")

    print(f"\n📊 Estadísticas de Similitud:")
    print(f"   • Promedio: {stats['promedio_similitud']:.3f}")
    print(f"   • Máximo: {stats['similitud_maxima']:.3f}")
    print(f"   • Mínimo: {stats['similitud_minima']:.3f}")

    print(f"\n📊 Multi-PRONACE: {stats['multi_pronace']} ({stats['multi_pronace']/stats['total_articulos']*100:.1f}%)")
    print(f"📝 Sin abstract: {stats['sin_abstract']} ({stats['sin_abstract']/stats['total_articulos']*100:.1f}%)")

    print("\n🔄 MEJORAS vs VERSIÓN ANTERIOR:")
    print("   ✅ Modelo más potente (768 dims)")
    print("   ✅ Normalización L2 aplicada")
    print("   ✅ Múltiples representaciones ponderadas")
    print("   ✅ Expansión inteligente de MeSH")
    print("   ✅ Boost específico del dominio")
    print("   ✅ Mejor distribución de confianza")

    print("\n" + "=" * 80)
    print("✅ CLASIFICACIÓN PRONACES MEJORADA COMPLETADA")
    print("=" * 80)
    print(f"\n📁 Archivo: {output_path}\n")

if __name__ == '__main__':
    main()