import os
import torch
import textgrid
import torchaudio

# --- CONFIGURATION ---
# Input: Raw audio files
AUDIO_FOLDER = "0_raw_data/untranscribed_audio"
# Output: Where the VAD TextGrids will be saved
TEXTGRID_FOLDER = "1_data_prepared/inference_textgrids"

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def main():
    print(f"Loading VAD model (Silero)...")
    # Load the Silero VAD model from torch hub
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  onnx=False)
    
    (get_speech_timestamps, _, read_audio, _, _) = utils
    
    ensure_folder(TEXTGRID_FOLDER)
    
    print(f"{'FILE':<35} | {'STATUS'}")
    print("-" * 60)
    
    # Process each wav file
    for filename in os.listdir(AUDIO_FOLDER):
        if not filename.endswith(".wav"):
            continue
            
        filepath = os.path.join(AUDIO_FOLDER, filename)
        base_name = os.path.splitext(filename)[0]
        
        try:
            # 1. Read Audio
            # The VAD model requires 16k mono
            wav = read_audio(filepath, sampling_rate=16000)
            
            # 2. Get speech timestamps (VAD)
            speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000, threshold=0.9)
            
            # 3. Create TextGrid
            # Get accurate duration of the original file
            info = torchaudio.info(filepath)
            duration = info.num_frames / info.sample_rate
            
            # FIX: Check if the last speech segment exceeds the file duration
            # This happens due to resampling rounding errors
            if speech_timestamps:
                last_end_seconds = speech_timestamps[-1]['end'] / 16000
                if last_end_seconds > duration:
                    # Extend duration slightly to accommodate the segment
                    duration = last_end_seconds + 0.001
            
            tg = textgrid.TextGrid()
            tier = textgrid.IntervalTier(name="speech", minTime=0, maxTime=duration)
            
            # Fill the tier with speech segments
            for segment in speech_timestamps:
                start = segment['start'] / 16000
                end = segment['end'] / 16000
                
                # Double safety: clip the end to the duration
                if end > duration:
                    end = duration
                if start >= end:
                    continue
                    
                tier.add(start, end, "speech")
                
            tg.append(tier)
            
            # 4. Save TextGrid
            out_path = os.path.join(TEXTGRID_FOLDER, f"{base_name}.TextGrid")
            with open(out_path, 'w') as f:
                tg.write(f)
                
            print(f"{filename:<35} | SUCCESS ({len(speech_timestamps)} segments)")
            
        except Exception as e:
            print(f"{filename:<35} | ERROR: {e}")

if __name__ == "__main__":
    main()