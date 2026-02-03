import os
import textgrid
import soundfile as sf
import librosa
import csv
import shutil

# --- CONFIGURATION ---
# Input folder containing raw .wav files (to be transcribed)
AUDIO_FOLDER = "0_raw_data/untranscribed_audio"
TEXTGRID_FOLDER = "1_data_prepared/inference_textgrids"

# Output: This is where the machine-only dataset will live
OUTPUT_DATASET = "1_data_prepared/inference_segments"
METADATA_FILE = "metadata_inference.csv"

# Target sampling rate for the model (Wav2Vec2 usually requires 16000Hz)
TARGET_SR = 16000

def ensure_clean_folder(folder):
    """
    Creates the directory if it does not exist.
    If it exists, it clears all contents to ensure a fresh start.
    """
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def main():
    # 1. Prepare the output directory
    print(f"Preparing output folder: {OUTPUT_DATASET}...")
    ensure_clean_folder(OUTPUT_DATASET)
    
    csv_path = os.path.join(OUTPUT_DATASET, METADATA_FILE)
    
    print(f"{'SOURCE FILE':<30} | {'STATUS'}")
    print("-" * 60)

    # 2. Initialize the CSV file
    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        
        # Header: We must include 'transcription' even if empty, for compatibility
        csv_writer.writerow(["filename", "duration", "transcription"])

        # Check if the TextGrid folder exists
        if not os.path.exists(TEXTGRID_FOLDER):
            print(f"ERROR: The folder {TEXTGRID_FOLDER} does not exist.")
            return

        # Get list of TextGrid files
        files = [f for f in os.listdir(TEXTGRID_FOLDER) if f.endswith('.TextGrid')]
        if not files:
            print(f"No .TextGrid files found in {TEXTGRID_FOLDER}")
            return

        # 3. Iterate through files
        for filename in files:
            base_name = os.path.splitext(filename)[0]
            tg_path = os.path.join(TEXTGRID_FOLDER, filename)
            
            # Locate the corresponding audio file in the raw_audio folder
            wav_path = os.path.join(AUDIO_FOLDER, f"{base_name}.wav")
            
            if not os.path.exists(wav_path):
                print(f"{filename:<30} | ERROR: Audio file not found in raw_audio")
                continue

            try:
                # Load audio and resample to 16kHz on the fly using librosa
                # This avoids the need for a separate pre-processing step
                data, sr = librosa.load(wav_path, sr=TARGET_SR)
                
                # Load the TextGrid
                tg = textgrid.TextGrid.fromFile(tg_path)
                
                # We assume the segmentation is in the first Tier (IntervalTier)
                tier = tg[0]
                
                count = 0
                for interval in tier:
                    # Filter out silence intervals
                    # (Adjust depending on how your VAD labels silence, e.g. "silence", "<p:>", or empty)
                    if interval.mark and interval.mark.lower() == "silence":
                        continue
                        
                    # Filter out very short segments (less than 1.5 seconds) to avoid artifacts
                    if (interval.maxTime - interval.minTime) < 2.0:
                        continue
                    
                    # Convert timestamps to sample indices
                    start_sample = int(interval.minTime * sr)
                    end_sample = int(interval.maxTime * sr)
                    
                    # Extract the audio segment
                    segment_audio = data[start_sample:end_sample]
                    
                    # Define the output filename for this segment
                    seg_name = f"{base_name}_seg{count+1:03d}.wav"
                    seg_path = os.path.join(OUTPUT_DATASET, seg_name)
                    
                    # Save the segment using SoundFile
                    sf.write(seg_path, segment_audio, sr)
                    
                    # Write metadata row
                    # Note: Transcription is left empty ("") for inference
                    duration = interval.maxTime - interval.minTime
                    csv_writer.writerow([seg_name, f"{duration:.2f}", ""])
                    
                    count += 1
                    
                print(f"{filename:<30} | SUCCESS: {count} segments generated")
                
            except Exception as e:
                print(f"{filename:<30} | ERROR: {e}")

    print("-" * 60)
    print("PROCESSING COMPLETE.")
    print(f"Output files are located in: {OUTPUT_DATASET}")

if __name__ == "__main__":
    main()