# Évaluation — modèles re-entraînés avec données du 2 octobre 2026

**Date** : 2026-10-02  
**Entraînement** : corpus historique (split figé seed=42) + 81 segments `KhO_ArcticReindeer` en train uniquement  
**Test** : `1_data_prepared/processed_audio_16k`, split 90/10, `seed=42` (18 segments) — comparable au protocole précédent  
**Script** : `scripts/8_evaluate_cer_ru.py`  
**Log brut** : `3_results/eval_plus_2oct26_cer_ru.txt`

## Scores sur le test fixe

| Modèle | Fine-tuning | WER | CER | Avant (corpus 174 seg.) |
|--------|-------------|-----|-----|-------------------------|
| **Wav2Vec2 XLSR-53** | gold + KhO (237 train / 18 val) | **68.42%** | **15.51%** | 70.87% / 16.47% |
| Whisper Small (russe) | gold + KhO (237 train / 18 val) | 106.32% | 57.25% | 72.82% / 22.84% |

## Notes d’entraînement (validation interne)

| Modèle | Durée | Early stop / epochs | Meilleur eval_loss (val interne) |
|--------|-------|---------------------|----------------------------------|
| Whisper Small RU | ~14 min | early stop ~epoch 40 | 0.717 (epoch ~6.7) |
| Wav2Vec2 XLSR | ~12 min | 100 epochs | 0.704 (epoch ~66.7) |

Logs : `train_whisper_small_plus_2oct26.log`, `train_xlsr_plus_2oct26.log`.

## Lecture

- **XLSR** s’améliore légèrement sur le test historique (WER −2,45 pts, CER −0,96 pts).
- **Whisper Small RU** se dégrade nettement sur ce même test (WER > 100 %) : les nouvelles données / orthographe ou un overfit fort nuisent à la généralisation hors du split interne.
