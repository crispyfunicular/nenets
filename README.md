# Nenets ASR Pipeline

Pipeline de reconnaissance automatique de la parole (ASR) pour le **nénètse**, langue à très faibles ressources (Samoyède, Sibérie). Le projet compare plusieurs approches : fine-tuning de modèles pré-entraînés multilingues et évaluation zero-shot via un LLM audio.

---

## Structure du projet

```
nenets/
├── 0_raw_data/                  # Fichiers audio bruts et TextGrids sources
├── 1_data_prepared/
│   ├── processed_audio_16k/     # Segments audio normalisés (16 kHz mono) + metadata.csv
│   ├── inference_segments/      # Segments à transcrire (audio non annoté)
│   └── inference_textgrids/     # TextGrids d'inférence
├── 2_models/                    # Checkpoints et poids finaux des modèles fine-tunés
├── 3_results/                   # CSV de transcriptions, fichiers d'évaluation, recap
├── OmnilingualZS/               # Pipeline zero-shot Omnilingual ASR
│   ├── corpus_entrainement/     # Exemples de contexte (90 % du corpus)
│   ├── corpus_evaluation/       # Segments de test (10 % du corpus)
│   ├── test_omni_ZS.py          # Script de test/exploration Omnilingual
│   └── 9_evaluate_omni_zs.py   # Évaluation formelle WER/CER Omnilingual
├── scripts/                     # Tous les scripts du pipeline
├── monolingual_texts/           # Textes monolingues nénètse (normalisation)
├── requirements.txt
└── vocab.json                   # Vocabulaire phonétique nénètse (Wav2Vec2)
```

---

## Corpus

- **~15 minutes** de parole nénètse transcrite manuellement
- **174 segments** annotés (TextGrid → CSV)
- Split reproductible : **90 % entraînement / 10 % test** (`seed=42`)

---

## Modèles et résultats

| Modèle | Type | Fine-tuning | WER | CER |
|--------|------|-------------|-----|-----|
| **Wav2Vec2 XLSR-53** | CTC | Corpus complet | **70.87%** | **16.47%** |
| Whisper Small (russe) | Seq2seq | Corpus complet | 72.82% | 22.84% |
| Whisper Large v3 (russe) | Seq2seq | Corpus complet | 88.42% | 26.38% |
| Whisper Small (sans langue) | Seq2seq | Corpus complet | 174.76% | 102.39% |
| **Omnilingual ZS 7B** | Zero-shot | Aucun | 142.86% | 64.34% |

> **Meilleur modèle** : Wav2Vec2 XLSR-53 fine-tuné avec alphabet phonétique adapté (WER 70.87%, CER 16.47%), malgré seulement ~15 min d'audio d'entraînement.

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
python scripts/9_evaluate_omni_zs.py [--num-context N] [--output FILE]
# ou depuis le dossier dédié :
python OmnilingualZS/9_evaluate_omni_zs.py
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