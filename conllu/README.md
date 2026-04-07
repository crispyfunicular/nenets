# Dossier `conllu`

Fichiers CoNLL-U du nénetse de la toundra (corpus MapTask) et script de parsing.

## Fichiers

| Fichier | Description |
|---|---|
| `yrk_MapTask_00N.conllu` | Corpus annoté (4 fichiers) |
| `parse_conllu.py` | Script de construction du lexique |
| `dict_parsing.json` | Dictionnaire POS+Lemme (généré) |
| `reverse_index.json` | Index inversé forme → candidats (généré) |

## Commandes

```bash
# Afficher le dictionnaire (stdout)
python3 parse_conllu.py yrk_MapTask_*.conllu

# Sauvegarder le dictionnaire en JSON
python3 parse_conllu.py yrk_MapTask_*.conllu --json dict_parsing.json

# Générer aussi l'index inversé (pour l'étiquetage)
python3 parse_conllu.py yrk_MapTask_*.conllu --json dict_parsing.json --reverse-index reverse_index.json

# Aide
python3 parse_conllu.py --help
```

## Format de sortie

### Dictionnaire (`--json`)

Clé : `POS+Lemme`. Valeur : fréquence + liste des formes attestées avec leurs informations.

```json
{
  "NOUN+яха": {
    "frequency": 15,
    "entries": [
      {
        "form": "яха",
        "features": null,
        "gloss": "river",
        "translit": "jaxa",
        "ltranslit": "jaxa"
      }
    ]
  },
  "VERB+яда-": {
    "frequency": 12,
    "entries": [
      {
        "form": "яда",
        "features": null,
        "gloss": "walk",
        "translit": "jada",
        "ltranslit": "jada-"
      }
    ]
  }
}
```

### Index inversé (`--reverse-index`)

Clé : forme de surface. Valeur : liste de candidats POS+Lemme triés par fréquence décroissante.
Utilisé pour étiqueter de nouvelles phrases segmentées : on choisit le candidat le plus fréquent.

```json
{
  "яда": [
    { "key": "VERB+яда-", "frequency": 12, "gloss": "walk",    "ltranslit": "jada-" },
    { "key": "ADV+яда",   "frequency": 7,  "gloss": "on.foot", "ltranslit": "jada"  }
  ],
  "яха": [
    { "key": "NOUN+яха",  "frequency": 15, "gloss": "river",   "ltranslit": "jaxa"  }
  ]
}
```

> Les tokens sans lemme propre (`_`) sont ignorés : ce sont les affixes et clitiques agglutinés (ex. `-дʼ`, `-мʼ`, `-вна`).
