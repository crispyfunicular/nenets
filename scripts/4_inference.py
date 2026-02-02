import torch
import pandas as pd
import torchaudio
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# --- CONFIGURATION ---
MODEL_PATH = "2_models/wav2vec2-large-xlsr-nenets"
DATASET_PATH = "1_data_prepared/processed_audio_16k"
# Output: The "Report Card" for the visio
OUTPUT_FILE = "3_results/nenets_test_results.csv"

def main():
    print("--- LOADING MODEL ---")
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)

    print("--- LOADING DATASET MANUALLY ---")
    # 1. Load the CSV file directly with Pandas
    csv_path = os.path.join(DATASET_PATH, METADATA_FILE)
    if not os.path.exists(csv_path):
        # Fallback: try looking for metadata.jsonl or similar if csv missing
        raise FileNotFoundError(f"Could not find metadata file at {csv_path}")
        
    df = pd.read_csv(csv_path, sep=None, engine='python', encoding='utf-8-sig') # Auto-detect separator
    print(f"Loaded metadata with {len(df)} rows.")

    # 2. Identify columns dynamically
    # Look for audio column (path, file_name, file, etc.)
    audio_col = next((col for col in ["filename","file_name", "path", "file", "audio"] if col in df.columns), None)
    # Look for text column (sentence, text, transcription, etc.)
    text_col = next((col for col in ["sentence", "text", "transcription"] if col in df.columns), None)

    if not audio_col:
        raise ValueError(f"Could not find audio column. Available columns: {df.columns}")
    
    print(f"Columns found -> Audio: '{audio_col}', Text: '{text_col}'")

    # 3. Filter for Test set (Optional: take last 10% as test)
    # Since we don't have a split column, we just take the last 10% of rows
    num_test = int(len(df) * 0.1)
    if num_test < 1: num_test = 1
    test_df = df.tail(num_test).reset_index(drop=True)
    
    print(f"--- TRANSCRIBING {len(test_df)} FILES ---")
    
    results = []

    for idx, row in test_df.iterrows():
        try:
            # Construct full path to audio
            filename = row[audio_col]
            full_audio_path = os.path.join(DATASET_PATH, filename)
            
            # Load Audio using TorchAudio
            speech_array, sampling_rate = torchaudio.load(full_audio_path)
            
            # Resample if necessary (Model expects 16000Hz)
            if sampling_rate != 16000:
                resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
                speech_array = resampler(speech_array)
            
            # Convert to mono if stereo
            if speech_array.shape[0] > 1:
                speech_array = torch.mean(speech_array, dim=0, keepdim=True)

            # Process input
            input_values = processor(speech_array.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True).input_values

            # Inference
            with torch.no_grad():
                logits = model(input_values.to(device)).logits

            # Decode
            pred_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(pred_ids)[0]
            
            # Store result
            results.append({
                "File": filename,
                "Reference": row[text_col] if text_col else "",
                "Prediction": transcription
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

        if (idx+1) % 5 == 0:
            print(f"Processed: {idx+1}/{len(test_df)}")

    # --- SAVE ---
    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_FILE, index=False, sep=";")
    print(f"\nCompleted! Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()