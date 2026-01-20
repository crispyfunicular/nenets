import os
import textgrid
import soundfile as sf
import csv

# --- CONFIGURATION ---
AUDIO_FOLDER = "processed_audio_16k"      # Input folder: Clean/normalized wav files
TEXTGRID_FOLDER = "output_textgrids"      # Input folder: TextGrids (corrected or raw)
OUTPUT_DATASET = "dataset_final"          # Output folder: Where sliced clips will go
METADATA_FILE = "metadata.csv"            # Output file: The CSV for the researcher

def ensure_folder(folder):
    """Creates the directory if it does not exist."""
    if not os.path.exists(folder):
        os.makedirs(folder)

def main():
    ensure_folder(OUTPUT_DATASET)
    
    # Prepare the CSV file (Columns: filename, duration, transcription)
    csv_path = os.path.join(OUTPUT_DATASET, METADATA_FILE)
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file, delimiter=',')
    # Header row
    csv_writer.writerow(["filename", "duration", "transcription"])

    print(f"{'SOURCE FILE':<30} | {'SEGMENTS CREATED'}")
    print("-" * 55)

    # Iterate over all TextGrid files in the folder
    for filename in os.listdir(TEXTGRID_FOLDER):
        if not filename.endswith('.TextGrid'):
            continue
            
        base_name = os.path.splitext(filename)[0]
        
        # Define file paths
        tg_path = os.path.join(TEXTGRID_FOLDER, filename)
        wav_path = os.path.join(AUDIO_FOLDER, f"{base_name}_16k.wav")
        
        # Check if the corresponding audio file exists
        if not os.path.exists(wav_path):
            print(f"WARNING: Audio file not found for {filename}")
            continue

        # 1. Load Audio (using SoundFile for stability) and TextGrid
        try:
            data, sr = sf.read(wav_path)
            tg = textgrid.TextGrid.fromFile(tg_path)
            
            # We assume the annotation is in the first tier (IntervalTier)
            tier = tg[0]
            
            count = 0
            # Loop through all intervals in the tier
            for i, interval in enumerate(tier):
                # Filter: Skip empty intervals or silences
                # (We keep only what is marked as "SPEECH" or any non-empty text)
                if interval.mark in ["", None]:
                    continue
                
                # Convert timestamps (seconds) to sample indices
                start_sample = int(interval.minTime * sr)
                end_sample = int(interval.maxTime * sr)
                
                # Extract the audio slice
                segment_audio = data[start_sample:end_sample]
                
                # Define segment filename
                # Example: yrk_MapTask_004_seg001.wav
                seg_name = f"{base_name}_seg{count+1:03d}.wav"
                seg_path = os.path.join(OUTPUT_DATASET, seg_name)
                
                # Save the segment
                sf.write(seg_path, segment_audio, sr)
                
                # Add entry to the CSV file
                duration = interval.maxTime - interval.minTime
                csv_writer.writerow([seg_name, f"{duration:.2f}", ""])
                
                count += 1
                
            print(f"{filename:<30} | {count} clips extracted")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    csv_file.close()
    print("-" * 55)
    print("PROCESSING COMPLETE.")
    print(f"1. Audio clips are in: {OUTPUT_DATASET}")
    print(f"2. Metadata file is:   {os.path.join(OUTPUT_DATASET, METADATA_FILE)}")

if __name__ == "__main__":
    main()