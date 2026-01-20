import os
import textgrid
import soundfile as sf
import csv
import shutil

# --- CONFIGURATION ---
# Folder containing the original source audio files (.wav)
AUDIO_FOLDER = "raw_audio"

# Folder containing the existing aligned TextGrids (.TextGrid)
# These files must have the SAME filename as the audio (e.g., file1.wav & file1.TextGrid)
TEXTGRID_FOLDER = "original_textgrids"

# Output folder where the sliced audio clips and the metadata CSV will be saved
OUTPUT_DATASET = "dataset_final"

# Name of the output metadata file (CSV format)
METADATA_FILE = "metadata.csv"

# IMPORTANT: The specific name of the tier containing the sentences to extract.
# You must check this name in Praat (e.g., "yrk", "sentence", "transcription").
TARGET_TIER_NAME = "yrk"

def ensure_clean_folder(folder):
    """
    Creates the directory if it doesn't exist.
    If it already exists, it DELETES it first to ensure a fresh start
    and avoid mixing old files with new ones.
    """
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def main():
    # 1. Prepare the output directory
    print(f"Preparing output folder: {OUTPUT_DATASET}...")
    ensure_clean_folder(OUTPUT_DATASET)
    
    # 2. Initialize the CSV metadata file
    # We use utf-8 encoding to support special characters (Cyrillic/Nenets)
    csv_path = os.path.join(OUTPUT_DATASET, METADATA_FILE)
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8-sig')
    csv_writer = csv.writer(csv_file, delimiter=',')
    
    # Write the header row
    csv_writer.writerow(["filename", "duration", "transcription"])

    print(f"{'SOURCE FILE':<30} | {'STATUS'}")
    print("-" * 60)

    # 3. Iterate through all WAV files in the raw_audio folder
    for filename in os.listdir(AUDIO_FOLDER):
        if not filename.lower().endswith('.wav'):
            continue
            
        base_name = os.path.splitext(filename)[0]
        wav_path = os.path.join(AUDIO_FOLDER, filename)
        
        # Construct the expected path for the corresponding TextGrid
        tg_path = os.path.join(TEXTGRID_FOLDER, f"{base_name}.TextGrid")
        
        # Check if the TextGrid actually exists
        if not os.path.exists(tg_path):
            print(f"{filename:<30} | SKIPPED (No matching TextGrid found)")
            continue

        try:
            # --- STEP A: Load Audio ---
            # Using soundfile library for robust reading of WAV files
            data, sr = sf.read(wav_path)
            
            # --- STEP B: Load TextGrid ---
            tg = textgrid.TextGrid.fromFile(tg_path)
            
            # --- STEP C: Find the correct Tier ---
            # We search by name (TARGET_TIER_NAME) because the tier index (1, 2, 3) might change.
            target_tier = None
            for tier in tg:
                if tier.name == TARGET_TIER_NAME:
                    target_tier = tier
                    break
            
            # If the tier is not found, log an error and skip this file
            if target_tier is None:
                print(f"{filename:<30} | ERROR: Tier '{TARGET_TIER_NAME}' not found in TextGrid")
                continue

            count = 0
            
            # --- STEP D: Loop through intervals in the tier ---
            for interval in target_tier:
                # Clean up the text (remove leading/trailing spaces)
                text = interval.mark.strip()
                
                # Filter: If the interval is empty (silence), skip it.
                if not text:
                    continue
                
                # Convert timestamps (seconds) to sample indices (integers)
                start_sample = int(interval.minTime * sr)
                end_sample = int(interval.maxTime * sr)
                
                # Safety check: Skip extremely short segments (< 1000 samples) to avoid errors
                if end_sample - start_sample < 1000: 
                    continue

                # Extract the audio segment (slicing the numpy array)
                segment_audio = data[start_sample:end_sample]
                
                # Define the new filename
                # Format: originalname_seg001.wav
                seg_name = f"{base_name}_seg{count+1:03d}.wav"
                seg_path = os.path.join(OUTPUT_DATASET, seg_name)
                
                # Save the new audio clip
                sf.write(seg_path, segment_audio, sr)
                
                # Write the entry to the CSV file
                # We save: filename, duration (in seconds), and the EXISTING TRANSCRIPTION
                duration = interval.maxTime - interval.minTime
                csv_writer.writerow([seg_name, f"{duration:.2f}", text])
                
                count += 1
            
            print(f"{filename:<30} | SUCCESS ({count} segments extracted)")

        except Exception as e:
            # Error handling block
            print(f"{filename:<30} | ERROR: {e}")

    # 4. Clean up and finish
    csv_file.close()
    print("-" * 60)
    print("PROCESSING COMPLETE.")
    print(f"1. Audio clips are saved in: {OUTPUT_DATASET}")
    print(f"2. Metadata (CSV) is saved as: {os.path.join(OUTPUT_DATASET, METADATA_FILE)}")

if __name__ == "__main__":
    main()