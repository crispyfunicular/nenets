# Inventory experiment runs — summary (Oct 2026)

Frozen eval holdout: **17 segments** (~54.5 s), identical for all runs (`splits/eval_holdout.txt`).
Checkpoint selection / early stopping: **eval WER** (patience 5, threshold 0.01).
Hypotheses: `3_results/eval_predictions/`.

## Training data (cumulative)

| Run | Train segments | Train duration | Content added |
|-----|---------------:|---------------:|---------------|
| 1 | 150 | ~8.2 min | Gold MapTask + PearStory |
| 2 | 294 | ~13.5 min | + KhO Arctic (81) + Bus/Park (63) |
| 3 | 346 | ~17.0 min | + training data 2 (52) |
| 4 | 701 | ~28.2 min | + DyLaCo Text1 (355) |

Speed perturbation (SP): train ×3 with `ffmpeg` `atempo=0.9` / `atempo=1.1` (same transcriptions; test unchanged).
SP corpora: `1_data_prepared/experiment_runs_sp/run{N}/` (derived only; sources untouched).

## Baseline — WER / CER (exported best-checkpoint hyps)

| Run | XLSR WER / CER | Whisper WER / CER |
|----:|----------------:|------------------:|
| 1 | 77.6% / 18.2% | 73.5% / 23.7% |
| 2 | 63.3% / 15.6% | 72.4% / 21.0% |
| 3 | 69.4% / 15.9% | 79.6% / 21.6% |
| 4 | 58.2% / 14.1% | 79.6% / 22.6% |

## Speed perturbation — WER / CER (exported hyps)

| Run | XLSR SP WER / CER | Whisper SP WER / CER |
|----:|------------------:|---------------------:|
| 1 | 64.3% / 16.3% | 75.5% / 24.7% |
| 2 | 62.2% / 14.9% | 70.4% / 18.4% |
| 3 | 65.3% / 14.1% | 72.4% / 18.6% |
| 4 | failed | 70.4% / 17.8% |

## Best eval WER during training (trainer logs)

| Run | XLSR | XLSR SP | Whisper | Whisper SP |
|----:|-----:|--------:|--------:|-----------:|
| 1 | 82.7% | 73.5% | 72.5% | 76.5% |
| 2 | 68.4% | 69.4% | 70.4% | 69.4% |
| 3 | 74.5% | 73.5% | 78.6% | 72.5% |
| 4 | 67.3% | 100.0% | 76.5% | 69.4% |

## Notes

- XLSR SP run 4 **failed** (eval WER stuck ~100%, early stop; export OOM). No `xlsr_sp_run4` predictions.
- For SP run 4 audio only: non-16 kHz wavs under `experiment_runs_sp/run4/` were resampled to 16 kHz (`scripts/14_resample_sp_corpus_16k.py`); `experiment_runs/` sources unchanged.
- Scores are **not** comparable to the historical seed=42 split.
- Model weights remain on MoDyCo (`2_models/`, not in git).
