# ASR Nenets v2 results (14 Aug 2026)

Automatic evaluation on the frozen 17-segment test set.

| Run | WER | CER |
|-----|-----|-----|
| v1 (published) | 70.87% | 16.47% |
| v2 (gold + training data 2) | 86.73% | 21.31% |

- `RESULTATS.txt` — summary
- `recognised_texts.txt` — gold (REF) vs model hypothesis (HYP) per segment

Local-only (gitignored): `2_models/wav2vec2-large-xlsr-nenets-v2/`, `1_data_prepared/`, `3_results/*.csv`, `train_v2.log` on MoDyCo at `~/morgane/nenets/`.
