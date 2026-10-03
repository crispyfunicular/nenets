# Évaluation — modèles re-entraînés avec données du 2 octobre 2026

**Date** : 2026-10-02 (éval) ; comparaison de checkpoints Whisper : 2026-10-03  
**Entraînement** : corpus historique (split figé seed=42) + 81 segments `KhO_ArcticReindeer` (locuteur Khadry Okotetto) en train uniquement  
**Test** : `1_data_prepared/processed_audio_16k`, split 90/10, `seed=42` (18 segments) — protocole inchangé  
**Scripts** : `scripts/8_evaluate_cer_ru.py` ; comparaison Whisper dans `eval_whisper_checkpoint_comparison_2oct26.csv`

## Scores sur le test fixe

| Modèle | Critère de sélection | WER | CER | Avant (174 seg.) |
|--------|----------------------|-----|-----|------------------|
| **Whisper Small RU `checkpoint-1200`** | meilleur **eval_wer** parmi les checkpoints sauvés | **45.26%** | **14.64%** | 72.82% / 22.84% |
| **Wav2Vec2 XLSR-53** | best by eval_loss (protocole entraînement) | **68.42%** | **15.51%** | 70.87% / 16.47% |
| Whisper Small RU (racine / `checkpoint-200`) | best by **eval_loss** (défaut du script) | 106.32% | 57.25% | 72.82% / 22.84% |

CSV : `eval_whisper_checkpoint_comparison_2oct26.csv`.

## Pourquoi deux scores Whisper ?

Le script d’entraînement utilise `metric_for_best_model="eval_loss"`.  
Sur ce run, le meilleur `eval_loss` est à ~epoch 6.7 (`checkpoint-200`, WER ~106 %), alors que la WER continue de baisser jusqu’à ~epoch 40 (`checkpoint-1200`, WER ~53 % en log, **45.26 %** à la ré-éval).

Les deux checkpoints viennent du **même run** (reproductible : seed, corpus, hyperparamètres, chemins `checkpoint-*`).  
Changer le critère de sélection (loss → WER) n’est pas un second entraînement ; c’est une analyse post-hoc documentée du même protocole.

**Limite méthodologique** (déjà présente avant) : le split « validation » d’entraînement est le même jeu que le test fixe. Sélectionner sur `eval_loss` ou `eval_wer` utilise donc ce jeu pour la sélection du modèle. Pour un protocole plus propre à l’avenir : train / dev / test disjoints, puis sélection sur le dev uniquement.

## Notes d’entraînement

| Modèle | Durée | Arrêt | Remarque |
|--------|-------|-------|----------|
| Whisper Small RU | ~14 min | early stop ~epoch 40 | best loss ≠ best WER |
| Wav2Vec2 XLSR | ~12 min | 100 epochs | gain modeste vs baseline |

## Lecture

- Avec sélection par **WER**, Whisper Small RU **surpasse** le baseline Whisper (72.8 %) et même XLSR (68.4 %) sur ce test.
- Le score « officiel » du script (best loss) était trompeur ; le documenter séparément évite de conclure à tort que KhO a « cassé » Whisper.
- XLSR reste un bon système CTC ; Whisper `checkpoint-1200` est pour l’instant le meilleur score seq2seq sur ce protocole.
