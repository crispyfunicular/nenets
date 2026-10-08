# Inventory experiment runs — summary (Oct 2026)

Frozen eval holdout: **17 segments** (~54.5 s), identical for all runs (`splits/eval_holdout.txt`).
Checkpoint selection / early stopping: **eval WER**.
Hypotheses: `3_results/eval_predictions/{xlsr,whisper}_run{N}/`.

## Training data (cumulative)

| Run | Train segments | Train duration | Content added |
|-----|---------------:|---------------:|---------------|
| 1 | 150 | ~8.2 min | Gold MapTask + PearStory |
| 2 | 294 | ~13.5 min | + KhO Arctic (81) + Bus/Park (63) |
| 3 | 346 | ~17.0 min | + training data 2 (52) |
| 4 | 701 | ~28.2 min | + DyLaCo Text1 (355) |

## Metrics on frozen eval (exported best-checkpoint hypotheses)

| Run | XLSR WER | XLSR CER | Whisper WER | Whisper CER |
|----:|----------:|---------:|------------:|------------:|
| 1 | 77.5% | 18.2% | 73.5% | 23.7% |
| 2 | 63.3% | 15.6% | 72.5% | 21.0% |
| 3 | 69.4% | 15.9% | 79.6% | 21.6% |
| 4 | 58.2% | 14.1% | 79.6% | 22.6% |

## Best eval WER during training (trainer logs)

| Run | XLSR best eval_wer | Whisper best eval_wer |
|----:|-------------------:|----------------------:|
| 1 | 82.7% | 72.5% |
| 2 | 68.4% | 70.4% |
| 3 | 74.5% | 78.6% |
| 4 | 67.3% | 76.5% |

## Notes

- Exported WER/CER are recomputed from `eval_predictions.txt` against the frozen test `metadata.csv`.
- These scores are **not** directly comparable to the historical seed=42 split results in `recap_all_models.md`.
- Model weights remain on MoDyCo under `2_models/` (not in git).
