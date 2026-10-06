import os
import re
import csv
import shutil
import unicodedata

import librosa
import soundfile as sf

# --- CONFIGURATION ---
SOURCE_DIR = "training data 2"
OUTPUT_DIR = "1_data_prepared/training_data_2_segments"
METADATA_FILE = "metadata.csv"
TARGET_SR = 16000
MIN_DURATION = 0.4
SENTENCE_TIER_NAMES = {"sentences", "sentence"}
SKIP_MARKERS = {"<p>", "<n>", "<rus>", "<pause>"}

# Nenets glottal-stop letters used in the new TextGrids → MapTask/PearStory ASCII
MODIFIER_PRIME = "\u02b9"          # ʹ
MODIFIER_DOUBLE_PRIME = "\u02ba"   # ʺ
MODIFIER_APOSTROPHE = "\u02bc"     # ʼ
MODIFIER_DOUBLE_APOSTROPHE = "\u02ee"  # ˮ

PHONETIC_PAREN_RE = re.compile(r"\(([^)]{1,3})\)")
SQUARE_BRACKET_RE = re.compile(r"\[([^]]+)\]")
REMAINING_PAREN_RE = re.compile(r"\([^)]*\)")
ELLIPSIS_RE = re.compile(r"\.{2,}")
NON_ALNUM_SLUG_RE = re.compile(r"[^a-z0-9]+")


def ensure_clean_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def normalize_stem(name):
    """Collapse spaces/underscores and curly apostrophes for WAV↔TextGrid pairing."""
    stem = os.path.splitext(name)[0].lower()
    stem = unicodedata.normalize("NFKC", stem)
    for ch in ("'", "'", "'", "'", "ʼ"):
        stem = stem.replace(ch, "")
    stem = re.sub(r"[\s_]+", "", stem)
    return stem


def slugify(stem):
    stem = unicodedata.normalize("NFKC", stem).lower()
    for ch in ("'", "'", "'", "'", "ʼ"):
        stem = stem.replace(ch, "")
    return NON_ALNUM_SLUG_RE.sub("_", stem).strip("_")


def parse_textgrid(path):
    """Minimal Praat TextGrid parser (long text format)."""
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
            intervals.append({
                "xmin": float(xmin_m.group(1)),
                "xmax": float(xmax_m.group(1)),
                "text": text_m.group(1),
            })
        tiers.append({"name": name, "intervals": intervals})
    return tiers


def find_sentence_tier(tiers):
    for tier in tiers:
        if tier["name"].strip().lower() in SENTENCE_TIER_NAMES:
            return tier
    return None


def pair_wavs_and_textgrids(source_dir):
    wavs = [f for f in os.listdir(source_dir) if f.lower().endswith(".wav")]
    tgs = [f for f in os.listdir(source_dir) if f.lower().endswith(".textgrid")]
    tg_by_stem = {normalize_stem(name): name for name in tgs}

    pairs = []
    unmatched = []
    for wav_name in sorted(wavs):
        key = normalize_stem(wav_name)
        tg_name = tg_by_stem.get(key)
        if tg_name is None:
            unmatched.append(wav_name)
            continue
        pairs.append((wav_name, tg_name))
    return pairs, unmatched


def is_skip_interval(raw_text):
    text = raw_text.strip()
    if not text:
        return True
    lowered = text.lower()
    if lowered in SKIP_MARKERS:
        return True
    if lowered.startswith("<") and lowered.endswith(">"):
        return True
    return False


def _unwrap_phonetic_paren(match):
    inner = match.group(1).strip()
    if " " not in inner and 1 <= len(inner) <= 2:
        return inner
    return match.group(0)


def lowercase_cyrillic(text):
    return "".join(ch.lower() if "\u0400" <= ch <= "\u04ff" else ch for ch in text)


def clean_transcription(raw_text):
    """Align new TextGrid orthography with the original MapTask/PearStory gold."""
    text = SQUARE_BRACKET_RE.sub(r"\1", raw_text)
    text = PHONETIC_PAREN_RE.sub(_unwrap_phonetic_paren, text)
    text = REMAINING_PAREN_RE.sub(" ", text)
    text = (
        text.replace(MODIFIER_PRIME, "'")
        .replace(MODIFIER_DOUBLE_PRIME, '"')
        .replace(MODIFIER_APOSTROPHE, "'")
        .replace(MODIFIER_DOUBLE_APOSTROPHE, '"')
    )
    text = ELLIPSIS_RE.sub(" ", text)
    text = lowercase_cyrillic(text)
    text = re.sub(r"\s*,\s*,+", ",", text)
    text = re.sub(r"\s*,\s*\.", ".", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" ,;")
    return text


def has_cyrillic(text):
    return any("\u0400" <= ch <= "\u04ff" for ch in text)


def main():
    print("=" * 60)
    print("  PREPARE TRAINING DATA 2 (sentence tier)")
    print("=" * 60)

    if not os.path.isdir(SOURCE_DIR):
        raise FileNotFoundError(f"Source folder not found: {SOURCE_DIR}")

    ensure_clean_folder(OUTPUT_DIR)
    csv_path = os.path.join(OUTPUT_DIR, METADATA_FILE)

    pairs, unmatched = pair_wavs_and_textgrids(SOURCE_DIR)
    if unmatched:
        print("Unmatched WAV files:")
        for name in unmatched:
            print(f"  - {name}")

    kept = 0
    skipped = 0
    total_dur = 0.0

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "file_name",
            "source",
            "start",
            "end",
            "duration",
            "transcription_raw",
            "transcription",
        ])

        print(f"{'SOURCE':<40} | {'STATUS'}")
        print("-" * 70)

        for wav_name, tg_name in pairs:
            wav_path = os.path.join(SOURCE_DIR, wav_name)
            tg_path = os.path.join(SOURCE_DIR, tg_name)
            source_stem = os.path.splitext(wav_name)[0]
            slug = slugify(source_stem)

            audio, _sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
            audio_dur = len(audio) / TARGET_SR

            tiers = parse_textgrid(tg_path)
            sentence_tier = find_sentence_tier(tiers)
            if sentence_tier is None:
                print(f"{wav_name:<40} | ERROR: no sentences tier")
                continue

            count = 0
            for interval in sentence_tier["intervals"]:
                raw = interval["text"]
                start = max(0.0, interval["xmin"])
                end = min(audio_dur, interval["xmax"])
                duration = end - start

                if is_skip_interval(raw):
                    skipped += 1
                    continue
                if duration < MIN_DURATION:
                    skipped += 1
                    continue

                cleaned = clean_transcription(raw)
                if not cleaned or not has_cyrillic(cleaned):
                    skipped += 1
                    continue

                count += 1
                seg_name = f"td2_{slug}_seg{count:03d}.wav"
                start_sample = int(start * TARGET_SR)
                end_sample = int(end * TARGET_SR)
                sf.write(
                    os.path.join(OUTPUT_DIR, seg_name),
                    audio[start_sample:end_sample],
                    TARGET_SR,
                )
                writer.writerow([
                    seg_name,
                    source_stem,
                    f"{start:.4f}",
                    f"{end:.4f}",
                    f"{duration:.2f}",
                    raw,
                    cleaned,
                ])
                kept += 1
                total_dur += duration

            print(f"{wav_name:<40} | {count} segments ({tg_name})")

    print("-" * 70)
    print(f"Kept: {kept} segments, {total_dur:.1f}s ({total_dur / 60:.2f} min)")
    print(f"Skipped: {skipped} intervals")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Metadata: {csv_path}")


if __name__ == "__main__":
    main()
