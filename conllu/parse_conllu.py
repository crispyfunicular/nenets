#!/usr/bin/env python3
"""
Parse des fichiers CoNLL-U (nénétse toundra, MapTask) et construit
un dictionnaire dont les clés sont des paires lemme_POS et les valeurs
contiennent les formes de surface, features, glose, translittération
et fréquence (globale et par entrée).

Les tokens sans lemme propre (lemma == '_') sont inclus : ce sont les
affixes et clitiques agglutinés (ex : -дʼ, -мʼ, -вна…). Pour ces tokens,
la forme de surface (form) est utilisée comme substitut de lemme dans la clé.

Les colonnes Head et DepRel (colonnes 7 et 8) sont ignorées
dans le dictionnaire de sortie.

Usage :
    python3 parse_conllu.py yrk_MapTask_*.conllu
    python3 parse_conllu.py yrk_MapTask_*.conllu --json output.json
    python3 parse_conllu.py yrk_MapTask_*.conllu --reverse-index reverse.json
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

            # Ignorer les tokens multi-mots (ID contenant un tiret, ex : "1-2")
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
    """Construit le dictionnaire lemme_POS → informations linguistiques.

    Inclut les tokens ordinaires (avec lemme) et les affixes/clitiques
    (lemma == '_'), pour lesquels la forme de surface sert de substitut de lemme.

    Retourne un dict :
        {
            "яха_NOUN": {
                "frequency": 15,
                "entries": [
                    {
                        "form": "яха",
                        "features": null,
                        "gloss": "river",
                        "translit": "jaxa",
                        "ltranslit": "jaxa",
                        "frequency": 15
                    },
                    ...
                ]
            },
            "-мʼ_ADP": {
                "frequency": 42,
                "entries": [
                    {
                        "form": "-мʼ",
                        "features": null,
                        "gloss": "-acc",
                        "translit": "-mʔ",
                        "ltranslit": null,
                        "frequency": 42
                    }
                ]
            },
            ...
        }
    """
    dictionary: dict[str, dict] = defaultdict(lambda: {"frequency": 0, "entries": []})

    for filepath in file_paths:
        tokens = parse_conllu_file(filepath)

        for token in tokens:
            lemma = token['lemma']
            upos = token['upos']

            # Construction de la clé lemme_POS.
            # Pour les affixes/clitiques (lemma == '_'), on utilise la forme
            # de surface comme substitut de lemme.
            # ex. affixes : "-мʼ_ADP", "-вна_ADP", "-да_DET"
            # ex. tokens ordinaires : "марˮ_NOUN", "яда-_VERB"
            if lemma == '_':
                key = f"{token['form']}_{upos}"
            else:
                key = f"{lemma}_{upos}"  # ex. : "марˮ_NOUN"

            # Extraire les infos du champ Misc
            misc = parse_misc_field(token['misc'])

            # Features (colonne 6)
            feats = token['feats'] if token['feats'] != '_' else None

            # Glose et translittération depuis Misc
            gloss = misc.get('Gloss', None)
            translit = misc.get('Translit', None)
            ltranslit = misc.get('LTranslit', None)

            # Fréquence globale de la clé (toutes formes/entrées confondues)
            dictionary[key]["frequency"] += 1

            # Recherche d'une entrée existante avec les mêmes informations
            # linguistiques (form + features + gloss + translit + ltranslit).
            # Si trouvée, on incrémente sa fréquence individuelle ;
            # sinon, on crée une nouvelle entrée avec frequency=1.
            existing = next(
                (e for e in dictionary[key]["entries"]
                 if e["form"] == token['form']
                 and e["features"] == feats
                 and e["gloss"] == gloss
                 and e["translit"] == translit
                 and e["ltranslit"] == ltranslit),
                None
            )
            if existing:
                existing["frequency"] += 1
            else:
                dictionary[key]["entries"].append({
                    "form": token['form'],
                    "features": feats,
                    "gloss": gloss,
                    "translit": translit,
                    "ltranslit": ltranslit,
                    "frequency": 1,
                })

    return dict(dictionary)


def build_reverse_index(dictionary: dict[str, dict]) -> dict[str, list[dict]]:
    """Construit un index inversé : forme → liste de candidats lemme_POS.

    Pour chaque forme de surface rencontrée, liste les clés lemme_POS
    possibles. Chaque candidat porte deux fréquences :
    - entry_frequency : fréquence de cette entrée spécifique (cette forme
      précise pour cette clé). Utilisée pour choisir l'annotation la plus
      probable en cas d'ambiguïté sur la forme.
    - key_frequency  : fréquence globale de la clé lemme_POS (toutes formes
      confondues). Utilisée comme critère de départage secondaire.

    Les candidats sont triés par entry_frequency décroissante, puis par
    key_frequency décroissante.

    → Évite de parcourir tout le dictionnaire (accès direct et non linéaire).

    Retourne un dict :
        {
            "яда": [
                {
                    "key": "яда-_VERB",
                    "entry_frequency": 12,
                    "key_frequency": 15,
                    "gloss": "walk",
                    "ltranslit": "jada-"
                },
                ...
            ],
            "-мʼ": [
                {
                    "key": "-мʼ_ADP",
                    "entry_frequency": 40,
                    "key_frequency": 42,
                    "gloss": "-acc",
                    "ltranslit": null
                },
                ...
            ],
            ...
        }
    """
    reverse: dict[str, list] = defaultdict(list)

    for key, data in dictionary.items():
        for entry in data["entries"]:
            form = entry["form"]
            # Vérifier si ce candidat (clé lemme_POS) existe déjà pour cette forme
            existing = next(
                (c for c in reverse[form] if c["key"] == key),
                None
            )
            if existing:
                # Mise à jour des fréquences (cas rare : même forme vue dans
                # plusieurs entrées du dictionnaire pour la même clé)
                existing["entry_frequency"] = entry["frequency"]
                existing["key_frequency"] = data["frequency"]
            else:
                reverse[form].append({
                    "key": key,
                    "entry_frequency": entry["frequency"],
                    "key_frequency": data["frequency"],
                    "gloss": entry.get("gloss"),
                    "ltranslit": entry.get("ltranslit"),
                })

    # Tri : d'abord sur la fréquence de l'entrée, ensuite sur la fréquence
    # globale de la clé comme critère de départage
    for form in reverse:
        reverse[form].sort(
            key=lambda c: (c["entry_frequency"], c["key_frequency"]),
            reverse=True
        )

    return dict(reverse)


def print_dictionary(dictionary: dict[str, dict]) -> None:
    """Affiche le dictionnaire de manière lisible."""
    print(f"{'='*60}")
    print(f"DICTIONNAIRE lemme_POS — {len(dictionary)} entrées")
    print(f"{'='*60}\n")

    for key in sorted(dictionary.keys()):
        data = dictionary[key]
        print(f"{key} (fréquence globale : {data['frequency']})")
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
            parts.append(f"Fréquence: {entry['frequency']}")
            print(f" -> {', '.join(parts)}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="parse_conllu.py",
        description="Parse des fichiers CoNLL-U (nénétse de la toundra) et construit un dictionnaire lemme_POS.",
        epilog='Ex : python3 parse_conllu.py yrk_MapTask_*.conllu --json dict.json --reverse-index reverse.json'
    )
    parser.add_argument(
        "fichiers",
        nargs="+",
        metavar='FICHIER',
        help="Fichiers CoNLL-U à traiter (globs acceptés, ex : yrk_MapTask_*.conllu)"
    )
    parser.add_argument(
        "--json",
        metavar="FICHIER",
        help="Sauvegarder le dictionnaire lemme_POS dans ce fichier JSON"
    )
    parser.add_argument(
        "--reverse-index",
        metavar="FICHIER",
        help="Sauvegarder l'index inversé (forme → candidats lemme_POS) dans ce fichier JSON"
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
