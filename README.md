# Nenets ASR Pipeline

Pipeline de reconnaissance automatique de la parole (ASR) pour le **nénètse**, langue à très faibles ressources (Samoyède, Sibérie). Le projet compare plusieurs approches : fine-tuning de modèles pré-entraînés multilingues et évaluation zero-shot via un LLM audio.

---

## Structure du projet

### Contenu versionné (branche `dev`)

```
nenets/
├── OmnilingualZS/               # Pipeline zero-shot Omnilingual ASR
│   ├── corpus_entrainement/     # Segments train (alignés sur l’inventaire)
│   ├── corpus_evaluation/       # Segments eval (holdout inventaire, 17 .wav)
│   ├── test_omni_ZS.py          # Test / exploration Omnilingual
│   └── 9_evaluate_omni_zs.py    # Évaluation formelle WER/CER Omnilingual
├── conllu/                      # Corpus CoNLL-U MapTask + lexique (parse_conllu.py)
├── scripts/                     # Scripts du pipeline ASR (préparation → évaluation)
├── splits/                      # Holdout eval + exclusions (no_content)
├── reports/v2/                  # Résultats Wav2Vec2 v2 (training data 2, 14 août 2026)
├── requirements.txt
├── vocab.json                   # Vocabulaire phonétique nénètse (Wav2Vec2 v1)
└── LICENSE
```

### Dossiers locaux attendus (non versionnés, voir `.gitignore`)

Ces répertoires sont requis par les scripts d'entraînement et d'inférence, mais **ne sont pas poussés sur GitHub** (volume, droits, fichiers audio). À recréer localement ou à récupérer séparément.

```
nenets/
├── 0_raw_data/
│   └── untranscribed_audio/     # Audio brut (.wav) pour VAD et découpe
├── training data 2/             # Nouveaux récits alignés (WAV 44.1 kHz stéréo + TextGrid)
├── 1_data_prepared/
│   ├── processed_audio_16k/     # Segments annotés 16 kHz + metadata.csv (fine-tuning v1)
│   ├── training_data_2_segments/          # Phrases extraites de training data 2
│   ├── processed_audio_16k_combined/      # Corpus v2 : train/ (gold+nouveau) + test/ gelé
│   ├── experiment_runs/run{1..4}/         # Corpora cumulatifs inventaire (train/ + test/)
│   ├── inference_segments/      # Segments à transcrire + metadata_inference.csv
│   └── inference_textgrids/     # TextGrids VAD / inférence
├── 2_models/                    # Checkpoints et poids finaux des modèles fine-tunés
│   ├── wav2vec2-large-xlsr-nenets/        # v1 (corpus original)
│   ├── wav2vec2-large-xlsr-nenets-v2/     # v2 (corpus original + training data 2)
│   ├── whisper-small-nenets-ru-runN/      # Runs inventaire (Whisper Small RU)
│   └── wav2vec2-large-xlsr-nenets-runN/   # Runs inventaire (XLSR)
├── 3_results/                   # CSV de transcriptions, évaluations, recap
├── logs/                        # Logs d’entraînement (MoDyCo / local)
└── monolingual_texts/           # Textes monolingues (scripts/normalize_monolingual_texts.py)
```

---

## Corpus

- Inventaire de référence : `Tundra Nenets data and metadata - training data.csv` (colonne Run1–Run4)
- Holdout d’évaluation **figé** : `splits/eval_holdout.txt` (**17** segments) — *pas* un resplit `seed=42`
- Exclusions : `splits/exclude_no_content.txt` (7 segments PearStory sans contenu)
- Corpora cumulatifs pour les expériences Run1–4 :
  - Run1 : 150 train / 17 test
  - Run2 : 294 train / 17 test
  - Run3 : 346 train / 17 test
  - Run4 : 701 train / 17 test  
  → `1_data_prepared/experiment_runs/run{1..4}/` (reconstruire : `python scripts/11_build_experiment_corpus.py --sync-omni`)
- **training data 2** (14 août 2026) et **KhO** (2 oct 2026) restent documentés dans les runs historiques / `reports/v2/`

---

## Modèles et résultats

### Runs inventaire (protocole Aleksandra, oct. 2026)

Même holdout de 17 segments pour tous les runs. Sélection du checkpoint par **eval WER** (`load_best_model_at_end`, `greater_is_better=False`). En fin d’entraînement : `eval_predictions.txt` (une hyp / ligne) pour IC bootstrap.

Tous les runs (Whisper + XLSR, 1–4) sont **terminés** (MoDyCo, 2026-10-08). Détail : [`3_results/experiment_runs_summary.md`](3_results/experiment_runs_summary.md).

| Run | Train | XLSR WER / CER (export) | Whisper WER / CER (export) |
|-----|------:|-------------------------:|---------------------------:|
| 1 | 150 | 77.6% / 18.2% | 73.5% / 23.7% |
| 2 | 294 | 63.3% / 15.6% | 72.4% / 21.0% |
| 3 | 346 | 69.4% / 15.9% | 79.6% / 21.6% |
| 4 | 701 | **58.2% / 14.1%** | 79.6% / 22.6% |

#### Speed perturbation (train ×3 : 0.9× + 1.1×)

| Run | Train SP | XLSR SP WER / CER | Whisper SP WER / CER |
|-----|---------:|------------------:|---------------------:|
| 1 | 450 | 64.3% / 16.3% | 75.5% / 24.7% |
| 2 | 882 | **62.2% / 14.9%** | **70.4% / 18.4%** |
| 3 | 1038 | 65.3% / 14.1% | 72.4% / 18.6% |
| 4 | 2103 | **61.2% / 14.7%** (relance 2026-10-09) | **70.4% / 17.8%** |

Hyps : `3_results/eval_predictions/{xlsr,whisper}[_sp]_run{N}/`. Logs : `logs/train_{whisper,xlsr}[_sp]_run{N}.log`. Poids : MoDyCo (`2_models/`, non versionnés).

### Résultats historiques (ancien split seed=42)

Évaluation sur le **test historique** `processed_audio_16k` (seed=42). Détail : [`3_results/eval_plus_2oct26_summary.md`](3_results/eval_plus_2oct26_summary.md).

| Modèle | Type | Fine-tuning | WER | CER |
|--------|------|-------------|-----|-----|
| **Whisper Small RU (`checkpoint-1200`)** | Seq2seq | gold + KhO ; sélection par **eval_wer** | **45.26%** | **14.64%** |
| **Wav2Vec2 XLSR-53** | CTC | gold + KhO (2 oct 2026) | **68.42%** | **15.51%** |
| Whisper Small RU (best **eval_loss**) | Seq2seq | gold + KhO ; défaut du script | 106.32% | 57.25% |
| Whisper Large v3 (russe) | Seq2seq | Corpus 174 seg. | 88.42% | 26.38% |
| Whisper Small (sans langue) | Seq2seq | Corpus 174 seg. | 174.76% | 102.39% |
| **Omnilingual ZS 7B** | Zero-shot | Aucun | 142.86% | 64.34% |

> Ces scores **ne sont pas directement comparables** aux WER des runs inventaire (holdout et inventaire différents).

---

## Pipeline

### A. Préparation des données

```bash
# 1. Détection d'activité vocale (VAD)
python scripts/1_generate_vad.py

# 2. Découpe de l'audio brut en segments
python scripts/2_cut_raw_audio.py
```

### B. Fine-tuning (runs inventaire Run1–4)

Prérequis : corpora sous `1_data_prepared/experiment_runs/run{N}/` + `splits/`.

```bash
# Reconstruire les 4 corpora (+ sync OmnilingualZS)
python scripts/11_build_experiment_corpus.py --sync-omni

# Un run (ex. Whisper RU, Run=1)
RUN=1 python scripts/3_train_whisper_ru.py
RUN=1 python scripts/3_train_wav2vec_long.py

# Enchaînement sur MoDyCo (skip si eval_predictions.txt existe déjà)
bash scripts/launch_experiment_runs.sh whisper 1 2 3 4
bash scripts/launch_experiment_runs.sh xlsr 1 2 3 4

# File d’attente après un job déjà en cours (Whisper restants puis XLSR)
# nohup bash scripts/queue_remaining_runs.sh > logs/nohup_queue.log 2>&1 &

# Export manuel des hyps eval (une colonne) pour IC
python scripts/12_export_eval_predictions.py \
  --model-dir 2_models/whisper-small-nenets-ru-run1 \
  --model-type whisper --run 1

# Speed perturbation (0.9× + 1.1× via ffmpeg atempo ; test non perturbé)
# Note : atempo=0.9 (pas 0.1) pour le ralentissement.
python scripts/13_build_speed_perturbed_corpus.py
# Si besoin : rééchantillonner uniquement experiment_runs_sp/ vers 16 kHz (sources intactes)
# python scripts/14_resample_sp_corpus_16k.py --runs 4
# XLSR en priorité, puis Whisper — outputs *-sp-runN
bash scripts/launch_speed_perturbed_runs.sh xlsr
bash scripts/launch_speed_perturbed_runs.sh whisper
# ou enchaîné : bash scripts/launch_speed_perturbed_runs.sh all
```

Variantes hors inventaire :

```bash
python scripts/3_train_whisper_large_ru.py    # Whisper Large v3, tokenizer russe
```

### B2. Fine-tuning v2 (training data 2)

Les nouveaux TextGrids sont alignés au palier `sentences`. L’orthographe `ʹ`/`ʺ` est normalisée vers `'`/`"` (convention MapTask). Le test v1 de 17 segments n’est **pas** resplit.

```bash
# 1. Extraire les phrases (mono 16 kHz) depuis training data 2
python scripts/10_prepare_training_data_2.py

# 2. Fusionner gold OmnilingualZS + nouveaux segments (test gelé)
python scripts/11_merge_datasets.py

# 3. Fine-tuning Wav2Vec2 XLSR-53 → 2_models/wav2vec2-large-xlsr-nenets-v2
python scripts/3_train_wav2vec_v2.py

# 4. Comparer WER/CER v1 vs v2 sur les 17 segments gelés
python scripts/8_evaluate_cer_v2.py
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
# Évaluation Wav2Vec2 v1 (split seed=42 sur processed_audio_16k)
python scripts/8_evaluate_cer.py

# Évaluation Wav2Vec2 v1 vs v2 (test gelé de 17 segments)
python scripts/8_evaluate_cer_v2.py

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

**Dépendances principales** : `torch`, `torchaudio`, `transformers`, `accelerate`, `datasets`, `librosa`, `jiwer`, `TextGrid`

---

## Références

- [Wav2Vec2 / XLSR-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) — Facebook AI
- [Whisper](https://huggingface.co/openai/whisper-large-v3) — OpenAI
- [Omnilingual ASR](https://github.com/juice500ml/omnilingual-asr) — Meta AI / Eungbeom Ha et al.
- [MoDyCo](https://www.modyco.fr/) — Modèles, Dynamiques, Corpus