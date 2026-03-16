import os
import torch
import pandas as pd
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa

MODEL_PATH = "2_models/whisper-large-v3-nenets-ru"
AUDIO_DIR = "1_data_prepared/inference_segments"
OUTPUT_FILE = "3_results/nenets_whisper_large_ru_transcriptions.csv"

def main():
    print(f"--- LOADING WHISPER MODEL: {MODEL_PATH} ---")
    processor = WhisperProcessor.from_pretrained(MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )

    # Apply Russian forced decoder ids
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="russian", task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.suppress_tokens = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
    results = []

    print(f"--- TRANSCRIBING {len(audio_files)} FILES WITH WHISPER LARGE (RU) ---")

    for i, file_name in enumerate(audio_files):
        path = os.path.join(AUDIO_DIR, file_name)

        try:
            speech, sr = librosa.load(path, sr=16000)

            input_features = processor(
                speech, return_tensors="pt", sampling_rate=16000
            ).input_features.to(torch.bfloat16).to(device)

            with torch.no_grad():
                predicted_ids = model.generate(input_features, max_new_tokens=225)

            transcription = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()

            transcription = " ".join(transcription.split())

            results.append({
                "file_name": file_name,
                "prediction": transcription
            })

            print(f"[{i+1}/{len(audio_files)}] {file_name}")

        except Exception as e:
            print(f"[{i+1}/{len(audio_files)}] Error: {file_name}: {e}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False, sep=";")
    print(f"\n Done! Whisper transcriptions saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
