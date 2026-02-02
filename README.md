# Nenets ASR Pipeline (Wav2Vec2)

This repository contains a full end-to-end Automatic Speech Recognition pipeline for the Nenets language.

## Project Structure
- **0_data_raw/**: Input files (Audio and TextGrids).
- **1_data_prepared/**: Processed audio segments and metadata.
- **2_models/**: Saved model checkpoints and final fine-tuned weights.

## Workflow

### A. Training & Evaluation
1. **Prepare Data**: Ensure segments and `metadata.csv` are in `1_data_prepared/train_segments/`.
2. **Train**: Run `3_train_wav2vec_long.py` (Fine-tunes XLSR-53 for 100 epochs).
3. **Test**: Run `4_inference.py` to compare AI predictions against human references.

### B. Transcribing New Audio
1. **VAD**: Run `1_generate_vad.py` to detect speech in raw files.
2. **Segment**: Run `2_cut_raw_audio.py` to slice the audio.
3. **Transcribe**: Run `5_transcribe_new.py` to generate automated Nenets transcriptions.