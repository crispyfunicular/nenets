import os
import torch
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa

# --- CONFIGURATION ---
MODEL_PATH = "2_models/wav2vec2-large-xlsr-nenets"
# Input: The segments we just cut with Script 2
AUDIO_DIR = "1_data_prepared/inference_segments" 
# Output: The final gift for Nikolett
OUTPUT_FILE = "3_results/nenets_unlabeled_transcriptions.csv"

def main():
    print(f"--- LOADING MODEL: {MODEL_PATH} ---")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # List all .wav files in the target directory
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
    results = []

    print(f"--- TRANSCRIBING {len(audio_files)} NEW UNLABELED FILES ---")
    
    for i, file_name in enumerate(audio_files):
        path = os.path.join(AUDIO_DIR, file_name)
        
        try:
            # Load audio (Wav2Vec2 requires 16kHz)
            speech, sr = librosa.load(path, sr=16000)
            
            # Pre-processing
            input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values.to(device)
            
            # Inference
            with torch.no_grad():
                logits = model(input_values).logits
            
            # Decoding
            pred_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(pred_ids)[0]
            
            results.append({
                "file_name": file_name, 
                "prediction": transcription
            })
            
            print(f"[{i+1}/{len(audio_files)}] Successfully processed: {file_name}")
            
        except Exception as e:
            print(f"[{i+1}/{len(audio_files)}] Error processing {file_name}: {e}")

    # Save results to a new CSV file
    df = pd.DataFrame(results)
    # Using semicolon separator for easy Excel opening
    df.to_csv(OUTPUT_FILE, index=False, sep=";")
    print(f"\n Done! Transcriptions saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()