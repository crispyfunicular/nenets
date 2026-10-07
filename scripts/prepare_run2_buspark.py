"""
Extract utterance segments for Run2 extras (BusStop 001/002 + Park)
from Speech-to-Text/002 TextGrids. Train-only inventory; does not touch
the historical test split.

Source (after unzipping 002.zip):
  0_raw_data/speech_to_text_002/002/
Output:
  1_data_prepared/run2_buspark_segments/
"""

import csv
import os
import re
import shutil
import unicodedata

import librosa
import soundfile as sf

SOURCE_DIR = "0_raw_data/speech_to_text_002/002"
OUTPUT_DIR = "1_data_prepared/run2_buspark_segments"
METADATA_FILE = "metadata.csv"
TARGET_SR = 16000
MIN_DURATION = 0.4

# Only the three texts Aleksandra asked to add to Run2 (Arctic already inventoried).
TARGETS = (
    "yrk_thea_BusStop_001",
    "yrk_thea_BusStop_002",
    "yrk_thea_Park",
)

SENTENCE_TIER_NAMES = {
    "sentences",
    "sentence",
    "corrected-slevel",
    "analysis",
}

MODIFIER_PRIME = "\u02b9"
MODIFIER_DOUBLE_PRIME = "\u02ba"
MODIFIER_APOSTROPHE = "\u02bc"
MODIFIER_DOUBLE_APOSTROPHE = "\u02ee"

PHONETIC_PAREN_RE = re.compile(r"\(([^)]{1,3})\)")
SQUARE_BRACKET_RE = re.compile(r"\[([^]]+)\]")
REMAINING_PAREN_RE = re.compile(r"\([^)]*\)")
ELLIPSIS_RE = re.compile(r"\.{2,}")
LATIN_RE = re.compile(r"[A-Za-z]")


def ensure_clean_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def parse_textgrid(path):
    with open(path, encoding="utf-8") as handle:
        content = handle.read()

    items = re.split(r"item\s*\[\d+\]\s*:", content)[1:]
    tiers = []
    for item in items:
        name_m = re.search(r'name\s*=\s*"([^"]*)"', item)
        name = name_m.group(1) if name_m else ""
        intervals = []
        for block in re.split(r"intervals\s*\[\d+\]\s*:", item)[1:]:
            xmin_m = re.search(r"xmin\s*=\s*([0-9.eE+-]+)", block)
            xmax_m = re.search(r"xmax\s*=\s*([0-9.eE+-]+)", block)
            text_m = re.search(r'text\s*=\s*"(.*)"', block)
            if not (xmin_m and xmax_m and text_m):
                continue
            intervals.append(
                {
                    "xmin": float(xmin_m.group(1)),
                    "xmax": float(xmax_m.group(1)),
                    "text": text_m.group(1),
                }
            )
        tiers.append({"name": name, "intervals": intervals})
    return tiers


def find_sentence_tier(tiers):
    for tier in tiers:
        if tier["name"].strip().lower() in SENTENCE_TIER_NAMES:
            return tier
    return None


def is_skip_interval(raw_text):
    text = raw_text.strip()
    if not text:
        return True
    lowered = text.lower()
    if lowered.startswith("<"):
        return True
    if lowered in {"pause", "sil", "silence"}:
        return True
    return False


def _unwrap_phonetic_paren(match):
    inner = match.group(1).strip()
    if " " not in inner and 1 <= len(inner) <= 2:
        return inner
    return match.group(0)


def lowercase_cyrillic(text):
    return "".join(ch.lower() if "\u0400" <= ch <= "\u04ff" else ch for ch in text)


# Latin lookalikes occasionally typed inside Cyrillic Nenets words.
LATIN_LOOKALIKES = str.maketrans(
    {
        "a": "а",
        "e": "е",
        "o": "о",
        "p": "р",
        "c": "с",
        "x": "х",
        "y": "у",
        "A": "А",
        "E": "Е",
        "O": "О",
        "P": "Р",
        "C": "С",
        "X": "Х",
        "Y": "У",
    }
)


def fix_latin_lookalikes(text):
    """Replace Latin lookalikes only in tokens that also contain Cyrillic."""
    parts = []
    for token in re.split(r"(\s+)", text):
        if has_cyrillic(token) and LATIN_RE.search(token):
            parts.append(token.translate(LATIN_LOOKALIKES))
        else:
            parts.append(token)
    return "".join(parts)


def clean_transcription(raw_text):
    text = SQUARE_BRACKET_RE.sub(r"\1", raw_text)
    text = PHONETIC_PAREN_RE.sub(_unwrap_phonetic_paren, text)
    text = REMAINING_PAREN_RE.sub(" ", text)
    text = (
        text.replace(MODIFIER_PRIME, "'")
        .replace(MODIFIER_DOUBLE_PRIME, '"')
        .replace(MODIFIER_APOSTROPHE, "'")
        .replace(MODIFIER_DOUBLE_APOSTROPHE, '"')
        .replace("\u04a3", "\u04c8")  # ң -> ӈ
        .replace("\u201d", '"')
        .replace("\u201c", '"')
    )
    text = ELLIPSIS_RE.sub(" ", text)
    text = lowercase_cyrillic(text)
    text = fix_latin_lookalikes(text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" ,;")


def has_cyrillic(text):
    return any("\u0400" <= ch <= "\u04ff" for ch in text)


def main():
    if not os.path.isdir(SOURCE_DIR):
        raise FileNotFoundError(f"Source folder not found: {SOURCE_DIR}")

    ensure_clean_folder(OUTPUT_DIR)
    csv_path = os.path.join(OUTPUT_DIR, METADATA_FILE)
    latin_hits = []
    kept = 0
    skipped = 0
    total_dur = 0.0
    inventory_rows = []

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "file_name",
                "source",
                "start",
                "end",
                "duration",
                "transcription_raw",
                "transcription",
            ]
        )

        for stem in TARGETS:
            wav_path = os.path.join(SOURCE_DIR, f"{stem}.wav")
            tg_path = os.path.join(SOURCE_DIR, f"{stem}.TextGrid")
            if not os.path.isfile(wav_path) or not os.path.isfile(tg_path):
                raise FileNotFoundError(f"Missing pair for {stem}")

            audio, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
            audio_dur = len(audio) / TARGET_SR
            tier = find_sentence_tier(parse_textgrid(tg_path))
            if tier is None:
                raise RuntimeError(f"No sentence-level tier in {tg_path}")

            count = 0
            for interval in tier["intervals"]:
                raw = interval["text"]
                start = max(0.0, interval["xmin"])
                end = min(audio_dur, interval["xmax"])
                duration = end - start

                if is_skip_interval(raw) or duration < MIN_DURATION:
                    skipped += 1
                    continue

                cleaned = clean_transcription(raw)
                if not cleaned or not has_cyrillic(cleaned):
                    skipped += 1
                    continue

                count += 1
                seg_name = f"{stem}_seg{count:03d}.wav"
                if LATIN_RE.search(cleaned):
                    latin_hits.append((seg_name, cleaned))

                start_sample = int(start * TARGET_SR)
                end_sample = int(end * TARGET_SR)
                sf.write(
                    os.path.join(OUTPUT_DIR, seg_name),
                    audio[start_sample:end_sample],
                    TARGET_SR,
                )
                writer.writerow(
                    [
                        seg_name,
                        stem,
                        f"{start:.4f}",
                        f"{end:.4f}",
                        f"{duration:.2f}",
                        raw,
                        cleaned,
                    ]
                )
                inventory_rows.append(
                    {
                        "file_name": seg_name,
                        "source": f"{stem}.wav",
                        "duration": duration,
                    }
                )
                kept += 1
                total_dur += duration

            print(f"{stem}: {count} segments (tier={tier['name']})")

    tsv_path = os.path.join(OUTPUT_DIR, "spreadsheet_run2.tsv")
    with open(tsv_path, "w", encoding="utf-8") as handle:
        handle.write(
            "Extracted file\tSource corpus\tSource text\t"
            "Split (train/dev/test)\tFile duration (sec)\tRun\tIssues\n"
        )
        for row in inventory_rows:
            dur = f"{row['duration']:.2f}".replace(".", ",")
            handle.write(
                f"{row['file_name']}\tThEA\t{row['source']}\ttrain\t{dur}\tRun2\t\n"
            )

    print(f"Kept: {kept} segments, {total_dur:.1f}s ({total_dur / 60:.2f} min)")
    print(f"Skipped: {skipped} intervals")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Spreadsheet TSV: {tsv_path}")
    if latin_hits:
        print("Latin letters left in transcription:")
        for name, text in latin_hits:
            print(f"  {name}: {text}")


if __name__ == "__main__":
    main()
