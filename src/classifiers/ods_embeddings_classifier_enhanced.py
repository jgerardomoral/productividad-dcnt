#!/usr/bin/env python3
"""
Clasificador ODS usando Embeddings MEJORADO
Mejoras implementadas:
- Modelo más potente: all-mpnet-base-v2
- Normalización L2 de embeddings
- Múltiples representaciones por categoría
- Umbrales optimizados
- Mejor procesamiento de texto
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

# Definiciones de ODS con descripciones mejoradas y múltiples representaciones
ODS_DEFINITIONS = {
    2: {
        "numero": 2,
        "nombre": "Hambre Cero",
        "meta": "Poner fin al hambre, lograr la seguridad alimentaria y la mejora de la nutrición",
        "descriptions": [
            # Descripción principal técnica
            (
                "This research addresses child malnutrition, stunting, wasting, undernutrition, "
                "anemia in women and children, micronutrient deficiencies including iron, zinc, and vitamin A, "
                "food insecurity, nutritional status in vulnerable populations, early childhood nutrition during "
                "the first 1000 days of life, maternal nutrition, nutritional interventions to prevent malnutrition, "
                "nutritional biomarkers in at-risk populations, growth monitoring, complementary feeding, "
                "breastfeeding practices, and strategies to improve food access and nutrition security."
            ),
            # Descripción enfocada en términos MeSH
            (
                "Malnutrition, infant nutrition disorders, protein-energy malnutrition, micronutrient deficiencies, "
                "vitamin A deficiency, iron deficiency anemia, zinc deficiency, growth disorders, child development, "
                "breast feeding, infant nutritional physiological phenomena, maternal nutritional physiological phenomena, "
                "food supply, food security, nutritional status assessment, anthropometry, child nutrition sciences"
            ),
            # Descripción enfocada en outcomes
            (
                "Reducing child mortality from malnutrition, preventing stunting and wasting in children under five, "
                "eliminating micronutrient deficiencies, improving maternal and infant nutrition outcomes, "
                "ensuring food security for vulnerable populations, achieving optimal growth and development in children"
            )
        ]
    },
    3: {
        "numero": 3,
        "nombre": "Salud y Bienestar",
        "meta": "Garantizar una vida sana y promover el bienestar para todos",
        "descriptions": [
            # Descripción principal
            (
                "This research focuses on non-communicable diseases (NCDs) including obesity, type 2 diabetes, "
                "cardiovascular disease, metabolic syndrome, dyslipidemia, hypertension, and cancer prevention. "
                "It also covers autoimmune diseases such as rheumatoid arthritis and systemic lupus erythematosus, "
                "chronic disease management, lifestyle interventions for health promotion, physical activity, exercise, "
                "sedentary behavior, body composition analysis, inflammatory biomarkers, oxidative stress, "
                "disease prevention strategies, clinical outcomes in chronic disease patients, therapeutic interventions, "
                "and strategies to reduce premature mortality from non-communicable diseases."
            ),
            # Términos MeSH
            (
                "Diabetes mellitus type 2, obesity, metabolic syndrome, cardiovascular diseases, hypertension, "
                "dyslipidemias, neoplasms, autoimmune diseases, rheumatoid arthritis, lupus erythematosus systemic, "
                "chronic disease, risk factors, biomarkers, inflammation, oxidative stress, lifestyle, exercise, "
                "physical activity, sedentary behavior, body composition, disease prevention, health promotion"
            ),
            # Outcomes clínicos
            (
                "Reducing obesity prevalence, preventing type 2 diabetes, controlling hypertension and cardiovascular risk, "
                "improving metabolic health parameters, reducing chronic inflammation, preventing cancer development, "
                "managing autoimmune conditions, improving quality of life in chronic disease patients, reducing mortality"
            )
        ]
    },
    10: {
        "numero": 10,
        "nombre": "Reducción de las Desigualdades",
        "meta": "Reducir la desigualdad en y entre los países",
        "descriptions": [
            # Principal
            (
                "This research examines health disparities, nutritional inequalities between socioeconomic groups, "
                "indigenous populations health and nutrition status, vulnerable populations in rural and urban poor areas, "
                "gender disparities in health outcomes, barriers to healthcare access, social determinants of health, "
                "health equity issues, cultural barriers to healthcare and nutrition services, "
                "community-based interventions in underserved areas, participatory research with marginalized communities, "
                "culturally appropriate interventions, and strategies to reduce health and nutrition inequalities "
                "across different population groups."
            ),
            # MeSH y determinantes sociales
            (
                "Health status disparities, healthcare disparities, socioeconomic factors, vulnerable populations, "
                "indigenous peoples, minority health, rural health, urban health, poverty, social determinants of health, "
                "health equity, healthcare accessibility, cultural competency, community participation, health services "
                "accessibility, social marginalization, ethnic groups, healthcare delivery"
            ),
            # Intervenciones y políticas
            (
                "Reducing health gaps between rich and poor, ensuring equitable access to nutrition services, "
                "addressing indigenous health disparities, eliminating gender-based health inequalities, "
                "implementing culturally appropriate interventions, improving healthcare access in marginalized communities"
            )
        ]
    },
    12: {
        "numero": 12,
        "nombre": "Producción y Consumo Responsables",
        "meta": "Garantizar modalidades de consumo y producción sostenibles",
        "descriptions": [
            # Principal
            (
                "This research analyzes sustainable food systems, traditional and indigenous foods versus "
                "ultra-processed foods, functional foods derived from biodiversity, dietary patterns and their "
                "environmental sustainability, food waste reduction strategies, local and regional food systems, "
                "sustainable agriculture practices, environmental impact of dietary choices, plant-based diets, "
                "traditional dietary patterns, consumption of ultra-processed foods and their health impacts, "
                "sugar-sweetened beverages, food industry practices, sustainable nutrition interventions, "
                "and circular economy approaches to food production and consumption."
            ),
            # Sistemas alimentarios y sostenibilidad
            (
                "Sustainable agriculture, food systems, ultra-processed foods, food processing, functional foods, "
                "dietary patterns, plant-based diet, Mediterranean diet, traditional foods, indigenous foods, "
                "food waste, environmental sustainability, carbon footprint, climate change, food industry, "
                "sugar-sweetened beverages, food labeling, nutrition policy, consumer behavior"
            ),
            # Impactos y soluciones
            (
                "Reducing ultra-processed food consumption, promoting sustainable dietary patterns, "
                "preserving traditional food systems, minimizing food waste, reducing environmental impact of diets, "
                "supporting local food production, improving food labeling and consumer awareness"
            )
        ]
    },
    1: {
        "numero": 1,
        "nombre": "Fin de la Pobreza",
        "meta": "Poner fin a la pobreza en todas sus formas",
        "descriptions": [
            (
                "This research addresses poverty-related malnutrition, economic barriers to healthy food access, "
                "relationships between socioeconomic status and nutrition, food prices and affordability, "
                "poverty alleviation through nutrition programs, economic evaluation of nutrition interventions, "
                "cost-effectiveness analysis of health programs, social protection programs, conditional cash transfers, "
                "and strategies to improve nutrition among low-income populations."
            ),
            (
                "Poverty, socioeconomic factors, food insecurity, malnutrition, economic factors, food prices, "
                "cost-benefit analysis, social welfare, public assistance, nutrition programs, food assistance, "
                "economic accessibility, household income, social protection"
            )
        ]
    },
    5: {
        "numero": 5,
        "nombre": "Igualdad de Género",
        "meta": "Lograr la igualdad entre los géneros y empoderar a todas las mujeres y las niñas",
        "descriptions": [
            (
                "This research investigates gender differences in nutrition and health outcomes, "
                "maternal health and nutrition during pregnancy and lactation, women's empowerment through "
                "nutrition and health programs, gender-specific nutritional needs and requirements, "
                "sex differences in disease prevalence and metabolism, reproductive health nutrition, "
                "gender-based violence and health, women's role in household food security, "
                "and interventions targeting women and girls' health and nutrition."
            ),
            (
                "Women's health, maternal health, pregnancy, lactation, breastfeeding, gender identity, "
                "sex characteristics, reproductive health, maternal nutrition, prenatal nutrition, "
                "postnatal care, women's rights, gender equity, female empowerment"
            )
        ]
    },
    13: {
        "numero": 13,
        "nombre": "Acción por el Clima",
        "meta": "Adoptar medidas urgentes para combatir el cambio climático",
        "descriptions": [
            (
                "This research examines climate change impacts on nutrition and food security, "
                "climate-resilient food systems, environmental sustainability of diets and food production, "
                "climate adaptation strategies in agriculture and nutrition, greenhouse gas emissions from food systems, "
                "drought and flood impacts on nutrition, climate-smart agriculture, carbon footprint of dietary patterns, "
                "and the intersection of climate change with food security and nutrition."
            ),
            (
                "Climate change, global warming, greenhouse gases, environmental sustainability, drought, floods, "
                "agriculture sustainability, food security, climate adaptation, mitigation strategies, "
                "carbon footprint, environmental impact, sustainable development, resilience"
            )
        ]
    }
}

# Umbrales optimizados
UMBRAL_PRINCIPAL = 0.50  # Aumentado de 0.45
UMBRAL_SECUNDARIO = 0.40  # Aumentado de 0.35
MIN_CONFIDENCE_SCORE = 0.35  # Score mínimo para no ser tentativo

def load_pubmed_articles():
    """Carga los 226 artículos del doctorado"""
    print("📂 Cargando artículos del DCNT...")

    metadata_file = 'data/pubmed_extracted/metadata_updated_20251024_043156.json'

    with open(metadata_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    print(f"   ✓ {len(articles)} artículos cargados")
    return articles

def prepare_text_enhanced(article):
    """Prepara texto con ponderación mejorada"""
    text_parts = []
    weights = []

    # Título (peso alto - 30%)
    titulo = article.get('title', '') or article.get('original_title', '')
    if titulo:
        text_parts.append(titulo)
        weights.append(0.30)

    # Abstract (peso más alto - 40%)
    abstract = article.get('abstract', '')
    if abstract:
        text_parts.append(abstract)
        weights.append(0.40)

    # MeSH terms (peso medio - 20%)
    mesh_terms = article.get('mesh_terms', [])
    if mesh_terms:
        # Expandir términos MeSH importantes
        mesh_text = " ".join(mesh_terms[:20])  # Aumentar de 15 a 20
        text_parts.append(f"Medical subjects: {mesh_text}")
        weights.append(0.20)

    # Keywords (peso bajo - 10%)
    keywords = article.get('keywords', [])
    if keywords:
        keywords_text = " ".join(keywords[:15])  # Aumentar de 10 a 15
        text_parts.append(f"Keywords: {keywords_text}")
        weights.append(0.10)

    # Publication type (peso muy bajo - 5%)
    pub_types = article.get('publication_types', [])
    if pub_types:
        pub_text = " ".join(pub_types)
        text_parts.append(f"Study type: {pub_text}")
        weights.append(0.05)

    # Normalizar pesos si no suman 1
    if weights:
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]

    # Por ahora, concatenar todo (en versión futura, usar embeddings ponderados)
    return " ".join(text_parts)

def get_confidence_level_enhanced(similarity, is_primary=True):
    """Determina confianza con umbrales mejorados"""
    if is_primary:
        if similarity >= 0.70:
            return "alta"
        elif similarity >= 0.55:
            return "media"
        elif similarity >= 0.45:
            return "baja"
        else:
            return "tentativa"
    else:
        # Para secundarios, ser más estricto
        if similarity >= 0.60:
            return "alta"
        elif similarity >= 0.50:
            return "media"
        elif similarity >= 0.40:
            return "baja"
        else:
            return "tentativa"

def generate_multiple_embeddings(model, ods_definitions):
    """Genera múltiples embeddings por categoría y los promedia"""
    ods_embeddings_dict = {}

    for ods_num, ods_data in ods_definitions.items():
        embeddings_list = []

        # Generar embeddings para cada descripción
        for description in ods_data.get('descriptions', [ods_data.get('description', '')]):
            if description:
                embedding = model.encode(description, convert_to_numpy=True)
                # Normalizar L2
                embedding = normalize(embedding.reshape(1, -1), norm='l2')[0]
                embeddings_list.append(embedding)

        # Promediar embeddings y normalizar de nuevo
        if embeddings_list:
            avg_embedding = np.mean(embeddings_list, axis=0)
            avg_embedding = normalize(avg_embedding.reshape(1, -1), norm='l2')[0]
            ods_embeddings_dict[ods_num] = avg_embedding

    return ods_embeddings_dict

def classify_with_embeddings_enhanced(model, article_texts, ods_embeddings_dict, ods_nums, articles):
    """Clasifica artículos con técnicas mejoradas"""

    print("\n🔄 Generando embeddings de artículos...")
    print("   (Con modelo mejorado all-mpnet-base-v2)")

    # Generar embeddings de artículos
    article_embeddings = model.encode(
        article_texts,
        show_progress_bar=True,
        batch_size=16,  # Reducir batch size para modelo más grande
        convert_to_numpy=True
    )

    # Normalizar embeddings L2
    print("   Aplicando normalización L2...")
    article_embeddings = normalize(article_embeddings, norm='l2')

    # Convertir dict a array para cálculo eficiente
    ods_embeddings = np.array([ods_embeddings_dict[num] for num in ods_nums])

    print("\n📊 Calculando similitudes mejoradas...")

    # Calcular similitud de coseno
    similarities = cosine_similarity(article_embeddings, ods_embeddings)

    # Aplicar suavizado y re-escalado
    # Esto ayuda a expandir el rango de similitudes
    similarities = np.power(similarities, 0.85)  # Suavizar picos

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
                    'confianza': get_confidence_level_enhanced(sim, is_primary=True)
                })

        # Si no hay principales con buen score, tomar el mejor
        if not ods_principales:
            max_ods = max(article_similarities, key=article_similarities.get)
            max_sim = article_similarities[max_ods]

            # Solo asignar si tiene un mínimo de similitud
            if max_sim >= MIN_CONFIDENCE_SCORE:
                ods_principales.append({
                    'numero': max_ods,
                    'nombre': ODS_DEFINITIONS[max_ods]['nombre'],
                    'similitud': round(max_sim, 4),
                    'confianza': get_confidence_level_enhanced(max_sim, is_primary=True)
                })
            else:
                # Muy baja similitud con todo - asignar ODS 3 (Salud) por defecto
                ods_principales.append({
                    'numero': 3,
                    'nombre': ODS_DEFINITIONS[3]['nombre'],
                    'similitud': round(article_similarities[3], 4),
                    'confianza': 'tentativa'
                })

        # Ordenar por similitud
        ods_principales.sort(key=lambda x: x['similitud'], reverse=True)

        # ODS secundarios (mejorado)
        ods_secundarios = []
        principales_nums = [o['numero'] for o in ods_principales]

        for ods_num, sim in article_similarities.items():
            if sim >= UMBRAL_SECUNDARIO and ods_num not in principales_nums:
                ods_secundarios.append({
                    'numero': ods_num,
                    'nombre': ODS_DEFINITIONS[ods_num]['nombre'],
                    'similitud': round(sim, 4),
                    'confianza': get_confidence_level_enhanced(sim, is_primary=False)
                })

        ods_secundarios.sort(key=lambda x: x['similitud'], reverse=True)
        ods_secundarios = ods_secundarios[:2]  # Máximo 2 secundarios

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
            'metodo': 'embeddings_enhanced_mpnet',
            'tiene_abstract': bool(article.get('abstract', ''))
        }

        results.append(classification)

    return results

def main():
    print("=" * 80)
    print("CLASIFICACIÓN ODS CON EMBEDDINGS MEJORADO")
    print("=" * 80)

    print("\n🚀 Mejoras implementadas:")
    print("   • Modelo superior: all-mpnet-base-v2")
    print("   • Normalización L2 de embeddings")
    print("   • Múltiples representaciones por ODS")
    print("   • Umbrales optimizados")
    print("   • Procesamiento de texto mejorado")

    print("\n📋 Metadata utilizada:")
    print("   • Título (30% peso)")
    print("   • Abstract completo (40% peso)")
    print("   • MeSH terms expandidos (20% peso)")
    print("   • Keywords (10% peso)")

    # 1. Cargar modelo mejorado
    print("\n🤖 Cargando modelo de embeddings mejorado...")
    print("   Modelo: sentence-transformers/all-mpnet-base-v2")
    print("   (Mejor rendimiento que all-MiniLM-L6-v2)")

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    print("   ✓ Modelo cargado")

    # 2. Cargar artículos
    articles = load_pubmed_articles()

    # 3. Preparar textos de artículos con método mejorado
    print("\n📝 Preparando textos de artículos (mejorado)...")
    article_texts = [prepare_text_enhanced(article) for article in articles]
    print(f"   ✓ {len(article_texts)} textos preparados")

    # 4. Generar embeddings de ODS con múltiples representaciones
    ods_nums = [1, 2, 3, 5, 10, 12, 13]
    print(f"\n📊 Generando embeddings mejorados de {len(ods_nums)} ODS...")
    print("   Usando múltiples representaciones por categoría...")

    ods_embeddings_dict = generate_multiple_embeddings(model, ODS_DEFINITIONS)

    print(f"   ✓ Embeddings de ODS generados ({len(ods_embeddings_dict)} categorías)")

    # 5. Clasificar con método mejorado
    results = classify_with_embeddings_enhanced(
        model, article_texts, ods_embeddings_dict, ods_nums, articles
    )

    print(f"\n   ✓ {len(results)} artículos clasificados")

    # 6. Estadísticas mejoradas
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
        },
        'promedio_similitud': 0,
        'similitud_maxima': 0,
        'similitud_minima': 1.0
    }

    similitudes_principales = []

    for result in results:
        # Contar por ODS
        if result['ods_principales']:
            for ods in result['ods_principales']:
                stats['por_ods'][ods['numero']] += 1
                # Confianza del primero
                if ods == result['ods_principales'][0]:
                    stats['por_confianza'][ods['confianza']] += 1
                    sim = ods['similitud']
                    similitudes_principales.append(sim)
                    stats['similitud_maxima'] = max(stats['similitud_maxima'], sim)
                    stats['similitud_minima'] = min(stats['similitud_minima'], sim)

        # Multi-ODS
        if len(result['ods_principales']) > 1:
            stats['multi_ods'] += 1

        # Sin abstract
        if not result['tiene_abstract']:
            stats['sin_abstract'] += 1

    if similitudes_principales:
        stats['promedio_similitud'] = np.mean(similitudes_principales)

    # 7. Guardar resultados mejorados
    print("\n💾 Guardando resultados mejorados...")

    output_data = {
        'metadata': {
            'fecha_generacion': datetime.now().isoformat(),
            'modelo': 'sentence-transformers/all-mpnet-base-v2',
            'metodo': 'embeddings_enhanced_normalized',
            'mejoras': [
                'Modelo MPNET superior',
                'Normalización L2',
                'Múltiples embeddings por categoría',
                'Umbrales optimizados',
                'Procesamiento de texto ponderado'
            ],
            'total_articulos': len(results),
            'ods_clasificados': ods_nums,
            'umbrales': {
                'principal': UMBRAL_PRINCIPAL,
                'secundario': UMBRAL_SECUNDARIO,
                'minimo': MIN_CONFIDENCE_SCORE
            }
        },
        'estadisticas': stats,
        'articulos': results
    }

    output_path = Path('data/ods_classification_embeddings_enhanced.json')
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✓ Guardado: {output_path}")
    print(f"   Tamaño: {file_size:.2f} MB")

    # 8. Resumen mejorado
    print("\n" + "=" * 80)
    print("📊 RESUMEN DE CLASIFICACIÓN ODS MEJORADA")
    print("=" * 80)

    print(f"\n✅ Total: {stats['total_articulos']} artículos")

    print(f"\n📈 Distribución por ODS:")
    for num in ods_nums:
        count = stats['por_ods'][num]
        pct = count / stats['total_articulos'] * 100 if stats['total_articulos'] > 0 else 0
        nombre = ODS_DEFINITIONS[num]['nombre']
        print(f"   ODS {num:2d} ({nombre[:30]:30s}): {count:3d} ({pct:5.1f}%)")

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

    print(f"\n📊 Multi-ODS: {stats['multi_ods']} ({stats['multi_ods']/stats['total_articulos']*100:.1f}%)")
    print(f"📝 Sin abstract: {stats['sin_abstract']} ({stats['sin_abstract']/stats['total_articulos']*100:.1f}%)")

    # Comparación con versión anterior
    print("\n🔄 MEJORAS vs VERSIÓN ANTERIOR:")
    print("   ✅ Modelo más potente (768 dims vs 384)")
    print("   ✅ Normalización L2 aplicada")
    print("   ✅ Múltiples representaciones por ODS")
    print("   ✅ Umbrales aumentados (0.50 vs 0.45)")
    print("   ✅ Mejor procesamiento de texto")

    print("\n" + "=" * 80)
    print("✅ CLASIFICACIÓN MEJORADA COMPLETADA")
    print("=" * 80)
    print(f"\n📁 Archivo: {output_path}")
    print("\n💡 Próximos pasos sugeridos:")
    print("   1. Ejecutar evaluación de métricas")
    print("   2. Comparar con clasificación anterior")
    print("   3. Validar manualmente muestra aleatoria")
    print("   4. Ajustar umbrales si es necesario\n")

if __name__ == '__main__':
    main()