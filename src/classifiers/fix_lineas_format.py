#!/usr/bin/env python3
"""
Script para convertir el formato de líneas al formato esperado por el dashboard
"""

import json
from pathlib import Path

def convert_lineas_format():
    """Convierte el formato de líneas al esperado por el dashboard"""

    # Cargar archivo de líneas
    with open('data/lineas_classification/final_classification.json', 'r', encoding='utf-8') as f:
        lineas_data = json.load(f)

    # Crear nueva lista de artículos con el formato correcto
    converted_articles = []

    for article in lineas_data['articulos']:
        # Extraer la línea principal del campo clasificacion
        clasificacion = article.get('clasificacion', {})
        lineas_principales = clasificacion.get('lineas_principales', [])

        # Crear estructura convertida
        converted = {
            'pmid': article['pmid'],
            'titulo': article['titulo'],
            'año': article['año']
        }

        # Asignar línea principal
        if lineas_principales:
            linea_principal = lineas_principales[0]  # Tomar la primera como principal
            converted['linea_principal'] = linea_principal['linea']
            converted['similitud_linea'] = linea_principal['similitud']
            converted['confianza'] = linea_principal['confianza']

            # Si hay líneas secundarias
            if len(lineas_principales) > 1:
                converted['lineas_secundarias'] = []
                for linea_sec in lineas_principales[1:]:
                    converted['lineas_secundarias'].append({
                        'linea': linea_sec['linea'],
                        'nombre': linea_sec['nombre'],
                        'similitud': linea_sec['similitud']
                    })

        # Agregar flags si existen
        flags = clasificacion.get('flags', {})
        if flags:
            converted['multi_linea'] = flags.get('multi_linea', False)
            converted['alta_confianza'] = flags.get('alta_confianza', False)

        converted_articles.append(converted)

    # Actualizar estructura completa
    output_data = {
        'metadata': lineas_data['metadata'],
        'estadisticas': lineas_data['estadisticas'],
        'articulos': converted_articles
    }

    # Guardar archivo convertido
    output_path = Path('data/lineas_classification/final_classification_fixed.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Archivo convertido guardado en: {output_path}")
    print(f"   Total artículos: {len(converted_articles)}")

    # Estadísticas
    total_con_linea = sum(1 for a in converted_articles if 'linea_principal' in a)
    total_multilinea = sum(1 for a in converted_articles if a.get('multi_linea', False))

    # Contar por línea
    lineas_count = {}
    for art in converted_articles:
        if 'linea_principal' in art:
            linea = art['linea_principal']
            lineas_count[linea] = lineas_count.get(linea, 0) + 1

    print(f"   Con línea asignada: {total_con_linea}")
    print(f"   Multi-línea: {total_multilinea}")
    print(f"   Distribución por línea: {lineas_count}")

    return output_path

if __name__ == '__main__':
    convert_lineas_format()