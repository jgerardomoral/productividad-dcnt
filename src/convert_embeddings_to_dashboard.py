#!/usr/bin/env python3
"""
Convierte embeddings_results.json al formato esperado por el dashboard
"""

import json
from pathlib import Path
from datetime import datetime

def convert_embeddings_to_dashboard_format():
    """Convierte clasificación de embeddings al formato del dashboard"""

    # Cargar resultados de embeddings
    input_file = Path('data/lineas_classification/embeddings_results.json')
    with open(input_file, 'r', encoding='utf-8') as f:
        embeddings_data = json.load(f)

    print("📂 Convirtiendo clasificación de embeddings a formato dashboard...")
    print(f"   Archivo origen: {input_file}")

    # Preparar estructura para dashboard
    articulos_dashboard = []
    stats_por_linea = {str(i): 0 for i in range(1, 4)}

    for art in embeddings_data['clasificaciones']:
        # Línea principal
        linea_principal = art['linea_principal']
        linea_num = linea_principal['linea']

        # Incrementar contador
        stats_por_linea[str(linea_num)] += 1

        # Construir líneas principales (incluye secundarias si aplica)
        lineas_principales = [{
            'linea': linea_num,
            'nombre': linea_principal['nombre'],
            'similitud': linea_principal['similitud'],
            'confianza': linea_principal['confianza']
        }]

        # Agregar líneas secundarias como principales adicionales (para multi-línea)
        for sec in art['lineas_secundarias']:
            lineas_principales.append({
                'linea': sec['linea'],
                'nombre': sec['nombre'],
                'similitud': sec['similitud'],
                'confianza': 'secundaria'
            })
            # También incrementar contador de líneas secundarias
            stats_por_linea[str(sec['linea'])] += 1

        # Determinar flags
        multi_linea = len(art['lineas_secundarias']) > 0
        alta_confianza = linea_principal['confianza'] in ['alta', 'media']

        # Construir artículo para dashboard
        articulo_dashboard = {
            'pmid': art['pmid'],
            'titulo': art['titulo'],
            'año': art['año'],
            'clasificacion': {
                'lineas_principales': lineas_principales,
                'metodo': 'embeddings_cosine_similarity',
                'similitudes': art['similitudes'],
                'flags': {
                    'multi_linea': multi_linea,
                    'alta_confianza': alta_confianza,
                    'tiene_abstract': art['tiene_abstract']
                }
            }
        }

        articulos_dashboard.append(articulo_dashboard)

    # Preparar estadísticas
    total_articulos = len(articulos_dashboard)
    multi_linea_count = sum(1 for a in articulos_dashboard if a['clasificacion']['flags']['multi_linea'])
    alta_confianza_count = sum(1 for a in articulos_dashboard if a['clasificacion']['flags']['alta_confianza'])

    # Estructura final
    dashboard_data = {
        'metadata': {
            'fecha_generacion': datetime.now().isoformat(),
            'metodo': 'embeddings_cosine_similarity',
            'modelo': embeddings_data['metadata']['modelo'],
            'total_articulos': total_articulos,
            'descripcion': 'Clasificación de artículos usando embeddings y similitud coseno'
        },
        'estadisticas': {
            'total_articulos': total_articulos,
            'por_linea': stats_por_linea,
            'multi_linea': multi_linea_count,
            'alta_confianza': alta_confianza_count,
            'por_confianza': embeddings_data['estadisticas']['por_confianza']
        },
        'articulos': articulos_dashboard
    }

    # Guardar
    output_file = Path('data/lineas_classification/final_classification.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Conversión completada!")
    print(f"   Archivo destino: {output_file}")
    print(f"   Tamaño: {output_file.stat().st_size / 1024:.1f} KB")
    print()
    print("📊 Estadísticas:")
    print(f"   Total artículos: {total_articulos}")
    print(f"   Línea 1: {stats_por_linea['1']} asignaciones")
    print(f"   Línea 2: {stats_por_linea['2']} asignaciones")
    print(f"   Línea 3: {stats_por_linea['3']} asignaciones")
    print(f"   Multi-línea: {multi_linea_count} ({multi_linea_count/total_articulos*100:.1f}%)")
    print(f"   Alta confianza: {alta_confianza_count} ({alta_confianza_count/total_articulos*100:.1f}%)")
    print()
    print("🎯 Archivo listo para el dashboard!")
    print("   Ejecuta: streamlit run src/app.py")

if __name__ == '__main__':
    convert_embeddings_to_dashboard_format()
