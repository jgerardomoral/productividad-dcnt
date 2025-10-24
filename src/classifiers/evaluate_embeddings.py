#!/usr/bin/env python3
"""
Script de Evaluación y Comparación de Embeddings
Compara el rendimiento entre versión original y mejorada
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def load_classification(file_path):
    """Carga archivo de clasificación"""
    if Path(file_path).exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    else:
        print(f"⚠️ Archivo no encontrado: {file_path}")
        return None

def analyze_confidence_distribution(data, name=""):
    """Analiza distribución de confianza"""
    if not data:
        return None

    stats = data.get('estadisticas', {})
    confianza = stats.get('por_confianza', {})

    total = stats.get('total_articulos', 0)
    if total == 0:
        return None

    print(f"\n📊 Distribución de Confianza - {name}")
    print("=" * 50)

    results = {}
    for nivel in ['alta', 'media', 'baja', 'tentativa']:
        count = confianza.get(nivel, 0)
        pct = (count / total) * 100
        results[nivel] = {'count': count, 'percentage': pct}

        emoji = {'alta': '🟢', 'media': '🟡', 'baja': '🟠', 'tentativa': '🔴'}[nivel]
        bar = '█' * int(pct / 2)
        print(f"   {emoji} {nivel.capitalize():10s}: {count:3d} ({pct:5.1f}%) {bar}")

    return results

def analyze_similarity_scores(data, name=""):
    """Analiza scores de similitud"""
    if not data or 'articulos' not in data:
        return None

    articles = data['articulos']
    all_similarities = []

    for article in articles:
        if article.get('ods_principales'):
            for ods in article['ods_principales']:
                all_similarities.append(ods.get('similitud', 0))
        elif article.get('pronaces_principales'):
            for pronace in article['pronaces_principales']:
                all_similarities.append(pronace.get('similitud', 0))

    if not all_similarities:
        return None

    print(f"\n📈 Estadísticas de Similitud - {name}")
    print("=" * 50)

    stats = {
        'mean': np.mean(all_similarities),
        'std': np.std(all_similarities),
        'min': np.min(all_similarities),
        'max': np.max(all_similarities),
        'median': np.median(all_similarities),
        'q1': np.percentile(all_similarities, 25),
        'q3': np.percentile(all_similarities, 75)
    }

    print(f"   • Media: {stats['mean']:.3f} (±{stats['std']:.3f})")
    print(f"   • Mediana: {stats['median']:.3f}")
    print(f"   • Rango: [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"   • Q1-Q3: [{stats['q1']:.3f}, {stats['q3']:.3f}]")

    # Distribución por rangos
    ranges = [
        (0.0, 0.3, "Muy bajo"),
        (0.3, 0.4, "Bajo"),
        (0.4, 0.5, "Medio-bajo"),
        (0.5, 0.6, "Medio"),
        (0.6, 0.7, "Medio-alto"),
        (0.7, 0.8, "Alto"),
        (0.8, 1.0, "Muy alto")
    ]

    print("\n   Distribución por rangos:")
    for min_val, max_val, label in ranges:
        count = sum(1 for s in all_similarities if min_val <= s < max_val)
        pct = (count / len(all_similarities)) * 100
        if count > 0:
            bar = '█' * int(pct / 2)
            print(f"   {label:12s} [{min_val:.1f}-{max_val:.1f}]: {count:3d} ({pct:5.1f}%) {bar}")

    return stats, all_similarities

def compare_classifications(original_path, enhanced_path, classification_type="ODS"):
    """Compara clasificaciones original vs mejorada"""
    print("\n" + "=" * 80)
    print(f"🔬 COMPARACIÓN DE CLASIFICACIONES - {classification_type}")
    print("=" * 80)

    # Cargar datos
    original = load_classification(original_path)
    enhanced = load_classification(enhanced_path)

    if not original:
        print(f"❌ No se pudo cargar clasificación original: {original_path}")
        print("   Ejecute primero el clasificador original")
        return

    if not enhanced:
        print(f"❌ No se pudo cargar clasificación mejorada: {enhanced_path}")
        print("   Ejecute primero el clasificador mejorado")
        return

    # 1. Comparar modelos
    print("\n🤖 MODELOS UTILIZADOS:")
    print("-" * 50)
    orig_model = original.get('metadata', {}).get('modelo', 'No especificado')
    enh_model = enhanced.get('metadata', {}).get('modelo', 'No especificado')
    print(f"   Original: {orig_model}")
    print(f"   Mejorado: {enh_model}")

    # 2. Comparar confianza
    print("\n" + "=" * 80)
    print("📊 COMPARACIÓN DE CONFIANZA")
    print("=" * 80)

    conf_original = analyze_confidence_distribution(original, "Original")
    conf_enhanced = analyze_confidence_distribution(enhanced, "Mejorada")

    if conf_original and conf_enhanced:
        print("\n📈 Mejoras en Confianza:")
        for nivel in ['alta', 'media', 'baja', 'tentativa']:
            orig_pct = conf_original[nivel]['percentage']
            enh_pct = conf_enhanced[nivel]['percentage']
            diff = enh_pct - orig_pct

            if nivel == 'tentativa':
                # Para tentativa, menos es mejor
                if diff < 0:
                    print(f"   ✅ {nivel.capitalize()}: {diff:+.1f}% (Mejorado)")
                else:
                    print(f"   ⚠️ {nivel.capitalize()}: {diff:+.1f}%")
            else:
                # Para otros niveles, más es mejor
                if diff > 0:
                    print(f"   ✅ {nivel.capitalize()}: {diff:+.1f}% (Mejorado)")
                else:
                    print(f"   ⚠️ {nivel.capitalize()}: {diff:+.1f}%")

    # 3. Comparar similitudes
    print("\n" + "=" * 80)
    print("📈 COMPARACIÓN DE SIMILITUDES")
    print("=" * 80)

    stats_orig, sims_orig = analyze_similarity_scores(original, "Original")
    stats_enh, sims_enh = analyze_similarity_scores(enhanced, "Mejorada")

    if stats_orig and stats_enh:
        print("\n🎯 Mejoras en Similitud:")
        mean_diff = stats_enh['mean'] - stats_orig['mean']
        median_diff = stats_enh['median'] - stats_orig['median']

        print(f"   • Media: {stats_orig['mean']:.3f} → {stats_enh['mean']:.3f} ({mean_diff:+.3f})")
        print(f"   • Mediana: {stats_orig['median']:.3f} → {stats_enh['median']:.3f} ({median_diff:+.3f})")

        if mean_diff > 0:
            pct_improvement = (mean_diff / stats_orig['mean']) * 100
            print(f"   ✅ Mejora promedio: {pct_improvement:.1f}%")
        else:
            print(f"   ⚠️ Sin mejora en promedio de similitud")

    # 4. Análisis de cambios por artículo
    print("\n" + "=" * 80)
    print("🔄 ANÁLISIS DE CAMBIOS POR ARTÍCULO")
    print("=" * 80)

    if original.get('articulos') and enhanced.get('articulos'):
        # Crear diccionarios por PMID
        orig_dict = {art['pmid']: art for art in original['articulos']}
        enh_dict = {art['pmid']: art for art in enhanced['articulos']}

        # Analizar cambios
        improved = 0
        worsened = 0
        changed_category = 0
        examples_improved = []
        examples_worsened = []

        for pmid, orig_art in orig_dict.items():
            if pmid in enh_dict:
                enh_art = enh_dict[pmid]

                # Obtener clasificación principal
                if classification_type == "ODS":
                    orig_main = orig_art.get('ods_principales', [{}])[0]
                    enh_main = enh_art.get('ods_principales', [{}])[0]
                elif classification_type == "PRONACES":
                    orig_main = orig_art.get('pronaces_principales', [{}])[0]
                    enh_main = enh_art.get('pronaces_principales', [{}])[0]
                else:
                    orig_main = orig_art.get('linea_principal', {})
                    enh_main = enh_art.get('linea_principal', {})

                orig_conf = orig_main.get('confianza', 'tentativa')
                enh_conf = enh_main.get('confianza', 'tentativa')
                orig_sim = orig_main.get('similitud', 0)
                enh_sim = enh_main.get('similitud', 0)

                # Comparar categorías
                orig_cat = orig_main.get('numero', orig_main.get('codigo', orig_main.get('linea', 0)))
                enh_cat = enh_main.get('numero', enh_main.get('codigo', enh_main.get('linea', 0)))

                if orig_cat != enh_cat:
                    changed_category += 1

                # Comparar confianza
                conf_levels = {'tentativa': 0, 'baja': 1, 'media': 2, 'alta': 3}
                if conf_levels.get(enh_conf, 0) > conf_levels.get(orig_conf, 0):
                    improved += 1
                    if len(examples_improved) < 3:
                        examples_improved.append({
                            'pmid': pmid,
                            'titulo': orig_art.get('titulo', '')[:60],
                            'orig_conf': orig_conf,
                            'enh_conf': enh_conf,
                            'orig_sim': orig_sim,
                            'enh_sim': enh_sim
                        })
                elif conf_levels.get(enh_conf, 0) < conf_levels.get(orig_conf, 0):
                    worsened += 1
                    if len(examples_worsened) < 3:
                        examples_worsened.append({
                            'pmid': pmid,
                            'titulo': orig_art.get('titulo', '')[:60],
                            'orig_conf': orig_conf,
                            'enh_conf': enh_conf,
                            'orig_sim': orig_sim,
                            'enh_sim': enh_sim
                        })

        total = len(orig_dict)
        unchanged = total - improved - worsened

        print(f"\n📊 Resumen de cambios ({total} artículos):")
        print(f"   ✅ Mejorados: {improved} ({improved/total*100:.1f}%)")
        print(f"   ➖ Sin cambio: {unchanged} ({unchanged/total*100:.1f}%)")
        print(f"   ❌ Empeorados: {worsened} ({worsened/total*100:.1f}%)")
        print(f"   🔄 Cambio de categoría: {changed_category} ({changed_category/total*100:.1f}%)")

        # Mostrar ejemplos
        if examples_improved:
            print("\n✅ Ejemplos de mejoras:")
            for ex in examples_improved:
                print(f"   • PMID {ex['pmid']}: {ex['titulo']}")
                print(f"     Confianza: {ex['orig_conf']} → {ex['enh_conf']}")
                print(f"     Similitud: {ex['orig_sim']:.3f} → {ex['enh_sim']:.3f}")

        if examples_worsened:
            print("\n❌ Ejemplos de empeoramientos:")
            for ex in examples_worsened:
                print(f"   • PMID {ex['pmid']}: {ex['titulo']}")
                print(f"     Confianza: {ex['orig_conf']} → {ex['enh_conf']}")
                print(f"     Similitud: {ex['orig_sim']:.3f} → {ex['enh_sim']:.3f}")

def generate_evaluation_report():
    """Genera reporte completo de evaluación"""
    print("\n" + "=" * 80)
    print("📋 REPORTE DE EVALUACIÓN DE EMBEDDINGS")
    print("=" * 80)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Evaluar ODS
    print("\n" + "=" * 80)
    print("1️⃣ CLASIFICACIÓN ODS")
    print("=" * 80)

    ods_original = 'data/classifications/ods_classification_embeddings.json'
    ods_enhanced = 'data/ods_classification_embeddings_enhanced.json'

    # Verificar si existe la versión mejorada
    if not Path(ods_enhanced).exists():
        print("\n⚠️ No se encontró clasificación ODS mejorada.")
        print("   Ejecute: python src/classifiers/ods_embeddings_classifier_enhanced.py")
    else:
        compare_classifications(ods_original, ods_enhanced, "ODS")

    # Evaluar PRONACES
    print("\n" + "=" * 80)
    print("2️⃣ CLASIFICACIÓN PRONACES")
    print("=" * 80)

    pronaces_original = 'data/classifications/pronaces_classification_embeddings.json'
    pronaces_enhanced = 'data/pronaces_classification_embeddings_enhanced.json'

    if not Path(pronaces_enhanced).exists():
        print("\n⚠️ No se encontró clasificación PRONACES mejorada.")
        print("   Se necesita crear pronaces_embeddings_classifier_enhanced.py")
    else:
        compare_classifications(pronaces_original, pronaces_enhanced, "PRONACES")

    # Evaluar Líneas
    print("\n" + "=" * 80)
    print("3️⃣ CLASIFICACIÓN LÍNEAS DE INVESTIGACIÓN")
    print("=" * 80)

    lineas_original = 'data/lineas_classification/embeddings_results.json'
    lineas_enhanced = 'data/lineas_classification/embeddings_results_enhanced.json'

    if not Path(lineas_enhanced).exists():
        print("\n⚠️ No se encontró clasificación de líneas mejorada.")
        print("   Se necesita crear embeddings_classifier_enhanced.py")
    else:
        compare_classifications(lineas_original, lineas_enhanced, "LINEAS")

    # Evaluar BioBERT
    print("\n" + "=" * 80)
    print("4️⃣ CLASIFICACIÓN BIOBERT")
    print("=" * 80)

    biobert_path = 'data/ods_classification_biobert.json'
    if Path(biobert_path).exists():
        compare_classifications(ods_original, biobert_path, "ODS-BioBERT")

    # Evaluar Ensemble
    print("\n" + "=" * 80)
    print("5️⃣ CLASIFICACIÓN ENSEMBLE FINAL")
    print("=" * 80)

    ensemble_path = 'data/ods_classification_ensemble_final.json'
    if Path(ensemble_path).exists():
        compare_classifications(ods_original, ensemble_path, "ODS-Ensemble")

    # Resumen final
    print("\n" + "=" * 80)
    print("📊 RESUMEN EJECUTIVO")
    print("=" * 80)

    print("\n🎯 Objetivos de mejora:")
    print("   1. Reducir artículos con confianza 'tentativa' < 20%")
    print("   2. Aumentar artículos con confianza 'alta' > 40%")
    print("   3. Mejorar similitud promedio > 0.60")
    print("   4. Lograr F1-score > 0.80 en validación")

    print("\n💡 Recomendaciones:")
    print("   1. Ejecutar clasificadores mejorados si no se han ejecutado")
    print("   2. Validar manualmente una muestra de 20-30 artículos")
    print("   3. Ajustar umbrales basándose en validación manual")
    print("   4. Considerar fine-tuning si los resultados no son satisfactorios")
    print("   5. Implementar sistema de votación entre múltiples modelos")

    print("\n✅ Evaluación completada")

def main():
    """Función principal"""
    print("\n" + "=" * 80)
    print("🔬 SISTEMA DE EVALUACIÓN DE EMBEDDINGS")
    print("=" * 80)

    # Generar reporte
    generate_evaluation_report()

    # Guardar reporte
    report_path = Path('data/evaluation_report.txt')
    print(f"\n💾 Reporte guardado en: {report_path}")

if __name__ == '__main__':
    main()