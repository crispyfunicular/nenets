# Nenets ASR Data Preparation Pipeline
This toolkit provides scripts to pre-process audio data for training the **Wav2Vec2** model. It handles the segmentation of long audio recordings into short, sentence-level clips and generates the necessary metadata CSV files.

## Project StructureBefore running the scripts, ensure your folders are organized as follows:Plaintextnenets/
```bash
├── raw_audio/                  # [INPUT] Place your .wav files here
├── original_textgrids/         # [INPUT] Place your human-corrected .TextGrid files here
├── dataset_final/              # [OUTPUT] This is where the sliced audio and CSV will appear
│
├── 1_extract_training_data.py  # Script for Case A (Files with transcriptions)
├── 2_generate_vad_draft.py     # Script for Case B (Raw audio without transcriptions)
│
├── venv/                       # Python virtual environment
└── requirements.txt            # List of dependencies
```
## Setup
Ensure your Python environment is active and dependencies are installed.
```bash
# Activate virtual environment (Linux/WSL)
source venv/bin/activate

# Install required libraries
pip install torch torchaudio soundfile textgrid
```
## Workflow
There are two main use cases for this pipeline:
### Case A: Preparing Training Data (Gold Standard)
**Use this when you already have the Audio + the corrected TextGrid.**
- Place the ```.wav``` file in ```raw_audio/```.
- Place the corresponding ```.TextGrid``` file in ```original_textgrids/```.
  - Note: The filenames must match (e.g., ```file1.wav``` and ```file1.TextGrid```).
- Run the extraction script:
```bash
python 1_extract_training_data.py
```
- **Result**: The script will check the ```dataset_final/``` folder. It contains:
  - Sliced audio files (e.g., file1_seg001.wav).
  - ```metadata.csv```: A file containing filenames, duration, and the transcription extracted from the TextGrid.

### Case B: Processing New/Raw Audio
**Use this when you receive a new recording and need to start from scratch.**
- Place the new ```.wav``` file in ```raw_audio/```.
- Run the VAD (Voice Activity Detection) script:
```bash
python 2_generate_vad_draft.py
```
*(Note: This is the script formerly known as OLD_script.py)*
- **Result**: A draft ```.TextGrid``` will be generated in ```output_textgrids/```.
- **Next Steps**:
  - Open the draft TextGrid in Praat.
  - Listen to the segments, adjust boundaries, and write the transcription.
  - Save the corrected TextGrid.Move it to ```original_textgrids/``` and proceed to **Case A**.

## Technical Notes
- **Sampling Rate**: All audio is automatically converted/saved as **16kHz** mono, which is required for Wav2Vec2.
- **Target Tier**: By default, the script looks for a tier named **"yrk"** in the TextGrid. If you change your annotation template, please update the ```TARGET_TIER_NAME``` variable inside 1_extract_training_data.py.
- **Engine**: The scripts use ```soundfile``` for robust I/O operations to ensure compatibility with Linux/WSL environments.  