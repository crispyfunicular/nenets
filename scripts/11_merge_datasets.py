import os
import re
import csv
import shutil

import librosa
import soundfile as sf

# --- CONFIGURATION ---
TRAIN_SRC = "OmnilingualZS/corpus_entrainement"
TEST_SRC = "OmnilingualZS/corpus_evaluation"
NEW_SRC = "1_data_prepared/training_data_2_segments"
OUTPUT_DIR = "1_data_prepared/processed_audio_16k_combined"
TARGET_SR = 16000

README_TRAIN_ROW_RE = re.compile(
    r"\|\s*`([^`]+)`\s*\|\s*([0-9.]+)\s*\|\s*`([^`]*)`\s*\|"
)
README_TEST_ROW_RE = re.compile(
    r"\|\s*`([^`]+)`\s*\|\s*`([^`]*)`\s*\|"
)


def ensure_clean_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def parse_readme_transcriptions(readme_path, split):
    """Extract filename → transcription from OmnilingualZS README tables."""
    with open(readme_path, encoding="utf-8") as handle:
        text = handle.read()

    mapping = {}
    if split == "train":
        for filename, _duration, transcription in README_TRAIN_ROW_RE.findall(text):
            mapping[filename] = transcription
    else:
        for filename, transcription in README_TEST_ROW_RE.findall(text):
            mapping[filename] = transcription
    if not mapping:
        raise ValueError(f"No transcriptions parsed from {readme_path}")
    return mapping


def resample_copy(src_wav, dest_wav):
    audio, _sr = librosa.load(src_wav, sr=TARGET_SR, mono=True)
    sf.write(dest_wav, audio, TARGET_SR)
    return len(audio) / TARGET_SR


def write_metadata(split_dir, rows):
    csv_path = os.path.join(split_dir, "metadata.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file_name", "transcription"])
        writer.writerows(rows)
    return csv_path


def copy_gold_split(src_dir, dest_dir, transcriptions, split_name):
    rows = []
    missing_wav = []
    missing_text = []
    wav_names = sorted(
        f for f in os.listdir(src_dir) if f.lower().endswith(".wav")
    )
    for wav_name in wav_names:
        if wav_name not in transcriptions:
            missing_text.append(wav_name)
            continue
        duration = resample_copy(
            os.path.join(src_dir, wav_name),
            os.path.join(dest_dir, wav_name),
        )
        rows.append((wav_name, transcriptions[wav_name]))
        print(f"  [{split_name}] {wav_name} ({duration:.2f}s)")

    for filename in transcriptions:
        if not os.path.exists(os.path.join(src_dir, filename)):
            missing_wav.append(filename)

    if missing_text:
        print(f"WARNING: {len(missing_text)} {split_name} wavs without README text")
        for name in missing_text:
            print(f"  - {name}")
    if missing_wav:
        print(f"WARNING: {len(missing_wav)} {split_name} README rows without wav")
        for name in missing_wav:
            print(f"  - {name}")
    return rows


def copy_new_segments(src_dir, dest_dir):
    metadata_path = os.path.join(src_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"{metadata_path} not found. Run scripts/10_prepare_training_data_2.py first."
        )

    rows = []
    with open(metadata_path, encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            wav_name = row["file_name"]
            src_wav = os.path.join(src_dir, wav_name)
            if not os.path.exists(src_wav):
                print(f"WARNING: missing new segment {wav_name}")
                continue
            shutil.copy2(src_wav, os.path.join(dest_dir, wav_name))
            rows.append((wav_name, row["transcription"]))
            print(f"  [new ] {wav_name} ({row.get('duration', '?')}s)")
    return rows


def main():
    print("=" * 60)
    print("  MERGE DATASETS (frozen original test split)")
    print("=" * 60)

    train_dir = os.path.join(OUTPUT_DIR, "train")
    test_dir = os.path.join(OUTPUT_DIR, "test")
    ensure_clean_folder(OUTPUT_DIR)
    os.makedirs(train_dir)
    os.makedirs(test_dir)

    train_texts = parse_readme_transcriptions(
        os.path.join(TRAIN_SRC, "README.md"), "train"
    )
    test_texts = parse_readme_transcriptions(
        os.path.join(TEST_SRC, "README.md"), "test"
    )
    print(f"README train rows: {len(train_texts)}")
    print(f"README test rows:  {len(test_texts)}")

    print("\nCopying original gold train...")
    train_rows = copy_gold_split(TRAIN_SRC, train_dir, train_texts, "gold")

    print("\nCopying training data 2 segments into train...")
    train_rows.extend(copy_new_segments(NEW_SRC, train_dir))

    print("\nCopying original gold test (frozen)...")
    test_rows = copy_gold_split(TEST_SRC, test_dir, test_texts, "test")

    write_metadata(train_dir, train_rows)
    write_metadata(test_dir, test_rows)

    print("-" * 60)
    print(f"Train: {len(train_rows)} segments")
    print(f"Test:  {len(test_rows)} segments (frozen original split)")
    print(f"Output: {OUTPUT_DIR}")
    if len(test_rows) != 17:
        print(f"WARNING: expected 17 frozen test files, got {len(test_rows)}")


if __name__ == "__main__":
    main()
