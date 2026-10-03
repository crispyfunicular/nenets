# Nenets ASR Pipeline

Pipeline de reconnaissance automatique de la parole (ASR) pour le **nénètse**, langue à très faibles ressources (Samoyède, Sibérie). Le projet compare plusieurs approches : fine-tuning de modèles pré-entraînés multilingues et évaluation zero-shot via un LLM audio.

---

## Structure du projet

### Contenu versionné (branche `dev`)

```
nenets/
├── OmnilingualZS/               # Pipeline zero-shot Omnilingual ASR
│   ├── corpus_entrainement/     # Segments train (90 %, 157 fichiers .wav)
│   ├── corpus_evaluation/       # Segments test (10 %, 17 fichiers .wav)
│   ├── test_omni_ZS.py          # Test / exploration Omnilingual
│   └── 9_evaluate_omni_zs.py    # Évaluation formelle WER/CER Omnilingual
├── conllu/                      # Corpus CoNLL-U MapTask + lexique (parse_conllu.py)
├── scripts/                     # Scripts du pipeline ASR (préparation → évaluation)
├── requirements.txt
├── vocab.json                   # Vocabulaire phonétique nénètse (Wav2Vec2)
└── LICENSE
```

### Dossiers locaux attendus (non versionnés, voir `.gitignore`)

Ces répertoires sont requis par les scripts d'entraînement et d'inférence, mais **ne sont pas poussés sur GitHub** (volume, droits, fichiers audio). À recréer localement ou à récupérer séparément.

```
nenets/
├── 0_raw_data/
│   └── untranscribed_audio/     # Audio brut (.wav) pour VAD et découpe
├── 1_data_prepared/
│   ├── processed_audio_16k/     # Segments annotés 16 kHz + metadata.csv (fine-tuning)
│   ├── inference_segments/      # Segments à transcrire + metadata_inference.csv
│   └── inference_textgrids/     # TextGrids VAD / inférence
├── 2_models/                    # Checkpoints et poids finaux des modèles fine-tunés
├── 3_results/                   # CSV de transcriptions, évaluations, recap
└── monolingual_texts/           # Textes monolingues (scripts/normalize_monolingual_texts.py)
```

---

## Corpus

- **~15 minutes** de parole nénètse transcrite manuellement (174 segments) + **81 segments** `KhO_ArcticReindeer` (~2,5 min) ajoutés au train le 2026-10-02
- Split de test figé : **90 % / 10 %** sur les 174 segments historiques (`seed=42`, 18 fichiers de test) ; les nouveaux segments sont **train only**
- Corpus d’entraînement courant : `1_data_prepared/processed_audio_16k_plus_2oct26/`

---

## Modèles et résultats

Évaluation sur le **test historique fixe** (`processed_audio_16k`, seed=42). Détail : [`3_results/eval_plus_2oct26_summary.md`](3_results/eval_plus_2oct26_summary.md).

| Modèle | Type | Fine-tuning | WER | CER |
|--------|------|-------------|-----|-----|
| **Wav2Vec2 XLSR-53** | CTC | gold + KhO (2 oct 2026) | **68.42%** | **15.51%** |
| Whisper Small (russe) | Seq2seq | gold + KhO (2 oct 2026) | 106.32% | 57.25% |
| Whisper Large v3 (russe) | Seq2seq | Corpus 174 seg. | 88.42% | 26.38% |
| Whisper Small (sans langue) | Seq2seq | Corpus 174 seg. | 174.76% | 102.39% |
| **Omnilingual ZS 7B** | Zero-shot | Aucun | 142.86% | 64.34% |

> **Meilleur modèle** : Wav2Vec2 XLSR-53 re-entraîné avec les données du 2 octobre (WER 68.42%, CER 15.51% ; avant : 70.87% / 16.47%). Whisper Small RU se dégrade sur le test fixe après cet ajout.

---

## Pipeline

### A. Préparation des données

```bash
# 1. Détection d'activité vocale (VAD)
python scripts/1_generate_vad.py

# 2. Découpe de l'audio brut en segments
python scripts/2_cut_raw_audio.py
```

### B. Fine-tuning

```bash
# Wav2Vec2 XLSR-53 (meilleur modèle)
python scripts/3_train_wav2vec_long.py

# Whisper (variantes)
python scripts/3_train_whisper_ru.py          # Whisper Small, tokenizer russe
python scripts/3_train_whisper_large_ru.py    # Whisper Large v3, tokenizer russe
```

### C. Inférence et transcription

```bash
# Évaluation sur le split test (modèles fine-tunés)
python scripts/4_inference.py

# Transcription de nouveaux fichiers audio
python scripts/5_transcribe_new.py
python scripts/5_transcribe_whisper_ru.py
python scripts/5_transcribe_whisper_large_ru.py
```

### D. Évaluation (WER / CER)

```bash
# Évaluation Wav2Vec2
python scripts/8_evaluate_cer.py

# Évaluation Whisper variantes
python scripts/8_evaluate_cer_ru.py
python scripts/8_evaluate_cer_large_ru.py

# Évaluation zero-shot Omnilingual
python OmnilingualZS/9_evaluate_omni_zs.py [--num-context N] [--output FILE]
```

### E. Post-traitement

```bash
# Formatage final des transcriptions
python scripts/6_final_formatting.py

# Fusion dans des TextGrids globaux
python scripts/7_merge_textgrids.py

# Export CSV pour relecture humaine
python scripts/9_export_for_review_ru.py
python scripts/9_export_for_review_large_ru.py
```

---

## Omnilingual ASR (Zero-Shot)

Le dossier `OmnilingualZS/` contient le pipeline d'évaluation du modèle [`omniASR_LLM_7B_ZS`](https://github.com/juice500ml/omnilingual-asr) (Meta AI), pré-entraîné sur 1600+ langues.

**Protocole** : 10 exemples du corpus servent de contexte audio (*few-shot prompting*), sans aucun fine-tuning. La sélection des exemples favorise des segments propres (sans marqueurs d'hésitation) et de durée idéale (~4 s).

```bash
# Test rapide / exploration
python OmnilingualZS/test_omni_ZS.py

# Évaluation formelle avec rapport WER/CER
python OmnilingualZS/9_evaluate_omni_zs.py --num-context 10
```

---

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Dépendances principales** : `torch`, `torchaudio`, `transformers`, `accelerate`, `datasets`, `librosa`, `jiwer`

---

## Références

- [Wav2Vec2 / XLSR-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) — Facebook AI
- [Whisper](https://huggingface.co/openai/whisper-large-v3) — OpenAI
- [Omnilingual ASR](https://github.com/juice500ml/omnilingual-asr) — Meta AI / Eungbeom Ha et al.
- [MoDyCo](https://www.modyco.fr/) — Modèles, Dynamiques, Corpus