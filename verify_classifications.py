#!/usr/bin/env python3
"""
Script to verify that classification files are loading correctly
"""

import json
from pathlib import Path
import pandas as pd

def verify_classifications():
    """Verify all classification files and their data"""

    base_dir = Path(__file__).parent / "data"

    # Load publications base
    publications = pd.read_csv(base_dir / "publications_base.csv")
    print(f"Total publications in base CSV: {len(publications)}")

    # 1. Check ODS classification (ensemble fixed)
    print("\n=== ODS Classification (Ensemble Fixed) ===")
    try:
        with open(base_dir / "ods_classification_ensemble_fixed.json", 'r', encoding='utf-8') as f:
            ods_data = json.load(f)
            articles = ods_data.get('articulos', [])
            print(f"Total articles: {len(articles)}")

            # Count classifications by confidence
            confidence_counts = {}
            ods_counts = {}
            for article in articles:
                if article.get('ods_principales'):
                    for ods in article['ods_principales']:
                        conf = ods.get('confianza', 'unknown')
                        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
                        ods_num = ods.get('numero')
                        if ods_num:
                            ods_counts[ods_num] = ods_counts.get(ods_num, 0) + 1

            print(f"Confidence distribution: {confidence_counts}")
            print(f"ODS distribution: {ods_counts}")

            # Check if all articles have classifications
            articles_with_ods = sum(1 for a in articles if a.get('ods_principales'))
            print(f"Articles with ODS principales: {articles_with_ods}/{len(articles)}")

            articles_with_secondary = sum(1 for a in articles if a.get('ods_secundarios'))
            print(f"Articles with ODS secundarios: {articles_with_secondary}/{len(articles)}")

    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}")
    except Exception as e:
        print(f"ERROR loading ODS: {e}")

    # 2. Check PRONACES classification
    print("\n=== PRONACES Classification ===")
    try:
        with open(base_dir / "classifications" / "pronaces_classification_embeddings.json", 'r', encoding='utf-8') as f:
            pronaces_data = json.load(f)
            articles = pronaces_data.get('articulos', [])
            print(f"Total articles: {len(articles)}")

            # Count PRONACES
            pronaces_counts = {}
            for article in articles:
                if article.get('pronaces_principales'):
                    for pronace in article['pronaces_principales']:
                        name = pronace.get('nombre', 'unknown')
                        pronaces_counts[name] = pronaces_counts.get(name, 0) + 1

            print(f"PRONACES distribution: {pronaces_counts}")
            articles_with_pronaces = sum(1 for a in articles if a.get('pronaces_principales'))
            print(f"Articles with PRONACES: {articles_with_pronaces}/{len(articles)}")

    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}")
    except Exception as e:
        print(f"ERROR loading PRONACES: {e}")

    # 3. Check Themes classification
    print("\n=== Themes Classification ===")
    try:
        with open(base_dir / "classifications" / "themes_classification.json", 'r', encoding='utf-8') as f:
            themes_data = json.load(f)
            print(f"Total articles: {len(themes_data)}")

            # Count themes
            theme_counts = {}
            for article in themes_data:
                if article.get('temas'):
                    for tema in article['temas']:
                        name = tema.get('nombre', 'unknown')
                        theme_counts[name] = theme_counts.get(name, 0) + 1

            # Sort themes by count
            sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
            print(f"Top 10 themes: {sorted_themes[:10]}")

            articles_with_themes = sum(1 for a in themes_data if a.get('temas'))
            print(f"Articles with themes: {articles_with_themes}/{len(themes_data)}")

    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}")
    except Exception as e:
        print(f"ERROR loading Themes: {e}")

    # 4. Check research lines classification
    print("\n=== Research Lines Classification ===")
    try:
        # Use final_classification_fixed.json which has the correct format
        final_path = base_dir / "lineas_classification" / "final_classification_fixed.json"
        with open(final_path, 'r', encoding='utf-8') as f:
            lines_data = json.load(f)
            print(f"Using fixed classification")

        articles = lines_data.get('articulos', [])
        print(f"Total articles: {len(articles)}")

        # Count lines
        line_counts = {}
        for article in articles:
            if article.get('linea_principal'):
                line = article['linea_principal']
                line_counts[line] = line_counts.get(line, 0) + 1

        print(f"Research lines distribution: {line_counts}")

    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}")
    except Exception as e:
        print(f"ERROR loading Lines: {e}")

if __name__ == '__main__':
    verify_classifications()