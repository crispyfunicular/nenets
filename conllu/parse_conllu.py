#!/usr/bin/env python3
"""
Parse des fichiers CoNLL-U (nénetse toundra, MapTask) et construit
un dictionnaire dont les clés sont POS+Lemme et les valeurs contiennent
features, glose, translittération et fréquence.

Les tokens sans lemme propre (lemma == '_') sont ignorés :
ce sont les affixes/clitiques agglutinés (ex: -дʼ, -мʼ, -вна…).

Les colonnes Head et DepRel (colonnes 7 et 8) sont ignorées
dans le dictionnaire de sortie.

Usage :
    python3 scripts/parse_conllu.py conllu/yrk_MapTask_*.conllu
    python3 scripts/parse_conllu.py conllu/yrk_MapTask_*.conllu --json output.json
    python3 scripts/parse_conllu.py conllu/yrk_MapTask_*.conllu --reverse-index reverse.json
"""

import os
import sys
import json
import glob
import argparse
from collections import defaultdict


def parse_misc_field(misc_str: str) -> dict[str, str]:
    """Parse le champ Misc (colonne 10) en dictionnaire.

    Exemple d'entrée :
        AlignBegin=979.85|AlignEnd=1364.09|Gloss=fence|LTranslit=marʡ|Translit=maŕa

    Retourne :
        {'AlignBegin': '979.85', 'AlignEnd': '1364.09',
         'Gloss': 'fence', 'LTranslit': 'marʡ', 'Translit': 'maŕa'}
    """
    if not misc_str or misc_str == '_':
        return {}
    result = {}
    for pair in misc_str.split('|'):
        if '=' in pair:
            key, value = pair.split('=', 1)
            result[key] = value
    return result


def parse_conllu_file(file: str) -> list[dict]:
    """Parse un fichier CoNLL-U et retourne une liste de tokens (dictionnaires)."""
    tokens = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')

            # Ignorer les lignes vides et les commentaires (métadonnées)
            if not line or line.startswith('#'):
                continue

            cols = line.split('\t')
            if len(cols) != 10:
                continue

            # Ignorer les tokens multi-mots (ID contenant un tiret, ex: "1-2")
            token_id = cols[0]
            if '-' in token_id or '.' in token_id:
                continue

            token = {
                'id': token_id,
                'form': cols[1],
                'lemma': cols[2],
                'upos': cols[3],
                'xpos': cols[4],
                'feats': cols[5],
                'head': cols[6],
                'deprel': cols[7],
                'deps': cols[8],
                'misc': cols[9],
            }
            tokens.append(token)
    return tokens


def build_dictionary(file_paths: list[str]) -> dict[str, dict]:
    """Construit le dictionnaire POS+Lemme : informations linguistiques.

    Retourne un dict :
        {
            "NOUN+яха": {
                "frequency": 15,
                "entries": [
                    {
                        "features": None,
                        "gloss": "river",
                        "translit": "jaxa",
                        "ltranslit": "jaxa",
                        "form": "яха"
                    },
                    ...
                ]
            },
            ...
        }
    """
    dictionary = defaultdict(lambda: {"frequency": 0, "entries": []})

    for filepath in file_paths:
        tokens = parse_conllu_file(filepath)

        for token in tokens:
            lemma = token['lemma']
            upos = token['upos']

            # Ignorer les tokens sans lemme propre (affixes/clitiques)
            if lemma == '_':
                continue

            key = f"{upos}+{lemma}"

            # Extraire les infos du champ Misc
            misc = parse_misc_field(token['misc'])

            # Features (colonne 6)
            feats = token['feats'] if token['feats'] != '_' else None

            # Glose et translittération depuis Misc
            gloss = misc.get('Gloss', None)
            translit = misc.get('Translit', None)
            ltranslit = misc.get('LTranslit', None)

            entry = {
                "form": token['form'],
                "features": feats,
                "gloss": gloss,
                "translit": translit,
                "ltranslit": ltranslit,
            }

            dictionary[key]["frequency"] += 1

            # Ajouter l'entrée seulement si combinaison unique
            if entry not in dictionary[key]["entries"]:
                dictionary[key]["entries"].append(entry)

    return dict(dictionary)


def build_reverse_index(dictionary: dict[str, dict]) -> dict[str, list[dict]]:
    """Construit un index inversé : forme : liste de candidats POS+Lemme.

    Pour chaque forme de surface rencontrée, liste les clés POS+Lemme
    possibles, triées par fréquence décroissante.
    -> Evite d'avoir à parcourir tout le dictionnaire (accès direct et non linéaire)

    Retourne un dict :
        {
            "яда": [
                {"key": "VERB+яда-", "frequency": 12, "gloss": "walk", ...},
                {"key": "ADV+яда",  "frequency": 8,  "gloss": "on.foot", ...}
            ],
            ...
        }
    """
    reverse = defaultdict(list)

    for key, data in dictionary.items():
        for entry in data["entries"]:
            form = entry["form"]
            # Vérifier si ce candidat existe déjà pour cette forme
            existing = None
            for candidate in reverse[form]:
                if candidate["key"] == key:
                    existing = candidate
                    break

            if existing:
                # Mettre à jour la fréquence (le max)
                existing["frequency"] = data["frequency"]
            else:
                reverse[form].append({
                    "key": key,
                    "frequency": data["frequency"],
                    "gloss": entry.get("gloss"),
                    "ltranslit": entry.get("ltranslit"),
                })

    # Trier chaque liste de candidats par fréquence décroissante
    for form in reverse:
        reverse[form].sort(key=lambda c: c["frequency"], reverse=True)

    return dict(reverse)


def print_dictionary(dictionary: dict[str, dict]) -> None:
    """Affiche le dictionnaire de manière lisible."""
    print(f"{'='*60}")
    print(f"DICTIONNAIRE POS+LEMME - {len(dictionary)} entrées")
    print(f"{'='*60}\n")

    for key in sorted(dictionary.keys()):
        data = dictionary[key]
        print(f"{key} (fréquence : {data['frequency']})")
        for entry in data["entries"]:
            parts = []
            if entry["form"]:
                parts.append(f"Forme: {entry['form']}")
            if entry["features"]:
                parts.append(f"Features: {entry['features']}")
            if entry["gloss"]:
                parts.append(f"Glose: {entry['gloss']}")
            if entry["translit"]:
                parts.append(f"Translit: {entry['translit']}")
            if entry["ltranslit"]:
                parts.append(f"LTranslit: {entry['ltranslit']}")
            print(f" -> {', '.join(parts)}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="parse_conllu.py",
        description="Parse des fichiers CoNLL-U (nénetse de la toundra) et construit un dictionnaire POS+Lemme.",
        epilog='"Ex : python3 parse_conllu.py yrk_MapTask_*.conllu --json dict.json --reverse-index reverse.json"
    )
    parser.add_argument(
        "fichiers",
        nargs="+",
        metavar='"FICHIER"',
        help="Fichiers CoNLL-U à traiter (globs acceptés, ex: yrk_MapTask_*.conllu)"
    )
    parser.add_argument(
        "--json",
        metavar="FICHIER",
        help="Sauvegarder le dictionnaire POS+Lemme dans ce fichier JSON"
    )
    parser.add_argument(
        "--reverse-index",
        metavar="FICHIER",
        help="Sauvegarder l'index inversé (forme -> candidats POS+Lemme) dans ce fichier JSON"
    )
    args = parser.parse_args()

    # Expansion des globs
    file_paths = []
    for pattern in args.fichiers:
        expanded = sorted(glob.glob(pattern))
        if expanded:
            file_paths.extend(expanded)
        else:
            file_paths.append(pattern)

    # Vérifier que les fichiers existent
    for fp in file_paths:
        if not os.path.isfile(fp):
            parser.error(f"fichier introuvable : {fp}")

    print(f"Traitement de {len(file_paths)} fichier(s) :")
    for fp in file_paths:
        print(f" - {fp}")
    print()

    # Construire et afficher le dictionnaire
    dictionary = build_dictionary(file_paths)
    print_dictionary(dictionary)

    # Sauvegarder le dictionnaire en JSON
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, ensure_ascii=False, indent=2)
        print(f"Dictionnaire sauvegardé dans : {args.json}")

    # Construire et sauvegarder l'index inversé
    if args.reverse_index:
        reverse_index = build_reverse_index(dictionary)
        with open(args.reverse_index, 'w', encoding='utf-8') as f:
            json.dump(reverse_index, f, ensure_ascii=False, indent=2)
        print(f"Index inversé sauvegardé dans : {args.reverse_index} ({len(reverse_index)} formes)")


if __name__ == '__main__':
    main()
