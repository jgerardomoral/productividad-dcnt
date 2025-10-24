#!/usr/bin/env python3
"""
Script para convertir el formato del ensemble al formato esperado por el dashboard
"""

import json
from pathlib import Path

# Definiciones de ODS para nombres
ODS_NAMES = {
    1: "Fin de la Pobreza",
    2: "Hambre Cero",
    3: "Salud y Bienestar",
    5: "Igualdad de Género",
    10: "Reducción de las Desigualdades",
    12: "Producción y Consumo Responsables",
    13: "Acción por el Clima"
}

def convert_ensemble_format():
    """Convierte el formato del ensemble al esperado por el dashboard"""

    # Cargar archivo ensemble
    with open('data/ods_classification_ensemble_final.json', 'r', encoding='utf-8') as f:
        ensemble_data = json.load(f)

    # Crear nueva estructura
    converted_articles = []

    for article in ensemble_data['articulos']:
        # Convertir estructura
        converted = {
            'pmid': article['pmid'],
            'titulo': article['titulo'],
            'año': article['año'],
            'revista': article.get('revista', ''),
            'doi': article.get('doi', ''),
            'ods_principales': [{
                'numero': article['ods_principal'],
                'nombre': ODS_NAMES.get(article['ods_principal'], f"ODS {article['ods_principal']}"),
                'similitud': article['avg_similarity'],
                'confianza': article['confidence']
            }],
            'ods_secundarios': []
        }

        # Agregar ODS secundarios si existen
        for ods_sec in article.get('ods_secundarios', []):
            if isinstance(ods_sec, dict):
                converted['ods_secundarios'].append({
                    'numero': ods_sec.get('numero', 0),
                    'nombre': ODS_NAMES.get(ods_sec.get('numero', 0), ''),
                    'similitud': ods_sec.get('avg_similarity', ods_sec.get('score', 0)),
                    'confianza': 'media'
                })
            elif isinstance(ods_sec, int):
                converted['ods_secundarios'].append({
                    'numero': ods_sec,
                    'nombre': ODS_NAMES.get(ods_sec, f"ODS {ods_sec}"),
                    'similitud': 0.4,
                    'confianza': 'media'
                })

        converted_articles.append(converted)

    # Crear estructura completa
    output_data = {
        'metadata': ensemble_data['metadata'],
        'estadisticas': ensemble_data['estadisticas'],
        'articulos': converted_articles
    }

    # Guardar archivo convertido
    output_path = Path('data/ods_classification_ensemble_fixed.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Archivo convertido guardado en: {output_path}")
    print(f"   Total artículos: {len(converted_articles)}")

    # Estadísticas
    total_con_principales = sum(1 for a in converted_articles if a['ods_principales'])
    total_con_secundarios = sum(1 for a in converted_articles if a['ods_secundarios'])

    print(f"   Con ODS principales: {total_con_principales}")
    print(f"   Con ODS secundarios: {total_con_secundarios}")

    return output_path

if __name__ == '__main__':
    convert_ensemble_format()