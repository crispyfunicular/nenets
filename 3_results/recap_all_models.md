# Récapitulatif — ASR Nénètse : Modèles et Résultats

**Corpus historique** : ~15 minutes (174 segments)  
**Ajout 2026-10-02** : 81 segments `KhO_ArcticReindeer` (Khadry Okotetto, ~2,5 min) en train uniquement  
**Évaluation** : split 90/10 sur les 174 segments, `seed=42` (18 test) — protocole inchangé  
**Date** : 2026-10-03

## Tableau comparatif

| Modèle | Type | Fine-tuning / sélection | WER | CER |
|--------|------|-------------------------|-----|-----|
| **Whisper Small RU `checkpoint-1200`** | Seq2seq | gold + KhO ; best **eval_wer** | **45.26%** | **14.64%** |
| **Wav2Vec2 XLSR** | CTC | gold + KhO ; best eval_loss | **68.42%** | **15.51%** |
| Wav2Vec2 XLSR (avant) | CTC | Corpus 174 seg. | 70.87% | 16.47% |
| Whisper Small RU (best **eval_loss**) | Seq2seq | gold + KhO ; défaut script | 106.32% | 57.25% |
| Whisper Small (russe, avant) | Seq2seq | Corpus 174 seg. | 72.82% | 22.84% |
| Whisper Small (sans langue) | Seq2seq | Corpus 174 seg. | 174.76% | 102.39% |
| Whisper Large v3 (russe) | Seq2seq | Corpus 174 seg. | 88.42% | 26.38% |
| **Omnilingual ZS** (7B) | Zero-shot | Aucun | 142.86% | 64.34% |

> Comparaison Whisper loss vs WER : `eval_whisper_checkpoint_comparison_2oct26.csv`.  
> Val d’entraînement = test fixe dans ce protocole (limite déjà présente pour la sélection par loss).

## Notes

- Le mauvais score Whisper « officiel » (106 %) venait du critère **eval_loss**, pas d’un échec du run.
- `checkpoint-1200` (fin de run) bat le baseline Whisper et XLSR sur ce test.
- Pour la suite : documenter le critère de sélection ; idéalement introduire un **dev** disjoint du test.
