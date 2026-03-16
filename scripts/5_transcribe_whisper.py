import os
import torch
import pandas as pd
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa

# --- CONFIGURATION ---
# Uses the fine-tuned Whisper model (separate from the Wav2Vec2 model)
MODEL_PATH = "2_models/whisper-small-nenets"
# Input: Same inference segments as Wav2Vec2
AUDIO_DIR = "1_data_prepared/inference_segments"
# Output: Separate file from Wav2Vec2 transcriptions
OUTPUT_FILE = "3_results/nenets_whisper_transcriptions.csv"

def main():
    print(f"--- LOADING WHISPER MODEL: {MODEL_PATH} ---")
    processor = WhisperProcessor.from_pretrained(MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)

    # Disable forced decoder IDs for unsupported languages
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # List all .wav files
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
    results = []

    print(f"--- TRANSCRIBING {len(audio_files)} FILES WITH WHISPER ---")

    for i, file_name in enumerate(audio_files):
        path = os.path.join(AUDIO_DIR, file_name)

        try:
            # Load audio at 16kHz
            speech, sr = librosa.load(path, sr=16000)

            # Extract Mel features
            input_features = processor(
                speech, return_tensors="pt", sampling_rate=16000
            ).input_features.to(device)

            # Generate transcription (autoregressive decoding)
            with torch.no_grad():
                predicted_ids = model.generate(input_features, max_new_tokens=225)

            # Decode
            transcription = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()

            # Clean up multiple spaces
            transcription = " ".join(transcription.split())

            results.append({
                "file_name": file_name,
                "prediction": transcription
            })

            print(f"[{i+1}/{len(audio_files)}] {file_name}")

        except Exception as e:
            print(f"[{i+1}/{len(audio_files)}] Error: {file_name}: {e}")

    # Save results (semicolon-separated, same format as W2V output)
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False, sep=";")
    print(f"\n Done! Whisper transcriptions saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
