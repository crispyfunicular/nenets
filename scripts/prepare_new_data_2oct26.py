"""
Build a training copy that keeps the historical 90/10 split and adds
KhO_ArcticReindeer speech intervals to the training set only.

Originals are never modified:
  - 0_raw_data/new_data_2oct26/
  - 1_data_prepared/processed_audio_16k/
"""

import csv
import os
import re
import shutil

import librosa
import soundfile as sf
from datasets import load_dataset

OLD_DATASET = "1_data_prepared/processed_audio_16k"
NEW_AUDIO = "0_raw_data/new_data_2oct26/KhO_ArcticReindeer.wav"
NEW_TEXTGRID = "0_raw_data/new_data_2oct26/KhO_ArcticReindeer.TextGrid"
OUTPUT_DIR = "1_data_prepared/processed_audio_16k_plus_2oct26"

TARGET_SR = 16000
SPLIT_SEED = 42
SPLIT_TEST_SIZE = 0.1

# Align the new transcriptions with the existing corpus (CTC tokens).
ORTHOGRAPHY = {
    "\u04A3": "\u04C8",  # ң -> ӈ
    "\u02BC": "'",       # ʼ -> '
    "\u201D": '"',       # ” -> "
    "\u201C": '"',       # “ -> "
    "\u02EE": '"',       # ˮ -> "
}

LATIN_RE = re.compile(r"[A-Za-z]")
INTERVAL_RE = re.compile(
    r"xmin = ([0-9.]+)\s+xmax = ([0-9.]+)\s+text = \"(.*)\"",
)


def normalize_transcription(text):
    for src, dst in ORTHOGRAPHY.items():
        text = text.replace(src, dst)
    return text


def read_speech_intervals(textgrid_path):
    raw = open(textgrid_path, encoding="utf-8").read()
    intervals = []
    for xmin, xmax, text in INTERVAL_RE.findall(raw):
        text = text.strip()
        if not text or text == "<p>" or text.startswith("<"):
            continue
        intervals.append((float(xmin), float(xmax), text))
    return intervals


def write_metadata(folder, rows):
    path = os.path.join(folder, "metadata.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["file_name", "duration", "transcription"]
        )
        writer.writeheader()
        writer.writerows(rows)


def historical_split(dataset_path):
    """Same split as the training scripts: audiofolder then seed 42, 10% test."""
    dataset = load_dataset("audiofolder", data_dir=dataset_path)
    if "test" in dataset:
        raise RuntimeError(
            f"{dataset_path} already has a test split; expected a flat audiofolder."
        )
    split = dataset["train"].train_test_split(
        test_size=SPLIT_TEST_SIZE, seed=SPLIT_SEED
    )
    # Drop the audio column before returning names so files are not decoded twice.
    names = {}
    texts = {}
    for part in ("train", "test"):
        names[part] = split[part]["audio"]
        # audiofolder exposes the transcript under the CSV column name
        col = next(
            c
            for c in ("transcription", "sentence", "text")
            if c in split[part].column_names
        )
        texts[part] = split[part][col]
    return names, texts


def main():
    print(f"Preparing {OUTPUT_DIR}")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    train_dir = os.path.join(OUTPUT_DIR, "train")
    test_dir = os.path.join(OUTPUT_DIR, "test")
    os.makedirs(train_dir)
    os.makedirs(test_dir)

    print("Reproducing the historical 90/10 split (seed=42)...")
    audio_cols, texts = historical_split(OLD_DATASET)

    rows = {"train": [], "test": []}
    for part, dest in (("train", train_dir), ("test", test_dir)):
        for audio, transcription in zip(audio_cols[part], texts[part]):
            src = audio["path"]
            name = os.path.basename(src)
            shutil.copy2(src, os.path.join(dest, name))
            duration = round(float(audio["array"].shape[0]) / audio["sampling_rate"], 2)
            rows[part].append(
                {
                    "file_name": name,
                    "duration": f"{duration:.2f}",
                    "transcription": transcription,
                }
            )

    print(
        f"  Copied existing corpus: {len(rows['train'])} train / {len(rows['test'])} test"
    )

    print("Cutting new TextGrid speech intervals into the training set...")
    audio, _ = librosa.load(NEW_AUDIO, sr=TARGET_SR, mono=True)
    latin_hits = []
    count = 0
    for start, end, text in read_speech_intervals(NEW_TEXTGRID):
        count += 1
        normalized = normalize_transcription(text)
        if LATIN_RE.search(normalized):
            latin_hits.append((count, normalized))
        name = f"KhO_ArcticReindeer_seg{count:03d}.wav"
        segment = audio[int(start * TARGET_SR) : int(end * TARGET_SR)]
        sf.write(os.path.join(train_dir, name), segment, TARGET_SR)
        rows["train"].append(
            {
                "file_name": name,
                "duration": f"{len(segment) / TARGET_SR:.2f}",
                "transcription": normalized,
            }
        )

    write_metadata(train_dir, rows["train"])
    write_metadata(test_dir, rows["test"])

    print(f"  Added {count} new training segments")
    print(
        f"  Final: {len(rows['train'])} train / {len(rows['test'])} test"
    )
    if latin_hits:
        print("  Latin letters left unchanged:")
        for index, text in latin_hits:
            print(f"    seg{index:03d}: {text}")
    else:
        print("  No residual Latin letters.")


if __name__ == "__main__":
    main()
