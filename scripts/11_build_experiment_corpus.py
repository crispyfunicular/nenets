"""
Build train/test audiofolders from the training-data inventory CSV.

Eval holdout = rows with Split == dev (frozen list also in splits/eval_holdout.txt).
Train for run N = rows with Run in {Run1..RunN}, Split == train, not excluded.

This replaces the old seed=42 random split so the next finetuning runs use the
same files as the spreadsheet (incl. PearStory seg006/063 in eval, seg005/051
in train).
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

INVENTORY = "Tundra Nenets data and metadata - training data.csv"
GOLD_META = "1_data_prepared/processed_audio_16k/metadata.csv"
GOLD_DIR = "1_data_prepared/processed_audio_16k"
EVAL_LIST = "splits/eval_holdout.txt"
EXCLUDE_LIST = "splits/exclude_no_content.txt"

# Extra train-only segment packs (local prepared dirs).
EXTRA_PACKS = {
    "Run2": [
        "1_data_prepared/processed_audio_16k_plus_2oct26/train",  # KhO_* only used from here
        "1_data_prepared/run2_buspark_segments",
    ],
    "Run3": [
        "1_data_prepared/training_data_2_segments",
    ],
    "Run4": [
        "1_data_prepared/yrk_dylaco_Text1_preprocessed/yrk_dylaco_Text1_preprocessed",
    ],
}

OUTPUT_ROOT = "1_data_prepared/experiment_runs"


def read_name_list(path: str) -> list[str]:
    names = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line)
    return names


def canon(name: str) -> str:
    return name.replace("yrk_thea_", "yrk_")


def load_gold_meta() -> dict[str, dict]:
    rows = {}
    with open(GOLD_META, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows[row["file_name"]] = row
    return rows


def load_inventory():
    with open(INVENTORY, encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_metadata(folder: Path, rows: list[dict]):
    """Write metadata compatible with HF audiofolder (file_name as plain string)."""
    path = folder / "metadata.csv"
    # newline='\n' + csv module avoids pandas StringDtype -> Arrow large_string,
    # which breaks datasets' check that file_name == Value("string").
    with path.open("w", newline="\n", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["file_name", "duration", "transcription"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "file_name": str(row["file_name"]),
                    "duration": f"{float(row['duration']):.2f}",
                    "transcription": str(row["transcription"]),
                }
            )


def copy_gold(name: str, dest_dir: Path, gold_meta: dict) -> dict:
    src = Path(GOLD_DIR) / name
    if not src.is_file():
        raise FileNotFoundError(f"Missing gold wav: {src}")
    shutil.copy2(src, dest_dir / name)
    meta = gold_meta[name]
    return {
        "file_name": name,
        "duration": f"{float(meta['duration']):.2f}",
        "transcription": meta["transcription"],
    }


def load_pack_meta(pack_dir: Path) -> dict[str, dict]:
    meta_path = pack_dir / "metadata.csv"
    if not meta_path.is_file():
        return {}
    with meta_path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    out = {}
    for row in rows:
        name = row.get("file_name") or row.get("filename")
        if not name:
            continue
        # training_data_2 uses 'transcription'; buspark too; some only duration
        text = row.get("transcription") or row.get("sentence") or ""
        dur = row.get("duration")
        out[name] = {"file_name": name, "duration": dur, "transcription": text}
    return out


def add_extra_run2(train_dir: Path, rows: list[dict], used: set[str]):
    """Add KhO Arctic + BusStop/Park segments to train."""
    # KhO from plus_2oct26 train folder
    kho_dir = Path("1_data_prepared/processed_audio_16k_plus_2oct26/train")
    kho_meta = load_pack_meta(kho_dir)
    for name, meta in sorted(kho_meta.items()):
        if not name.startswith("KhO_ArcticReindeer_"):
            continue
        if name in used:
            continue
        src = kho_dir / name
        if not src.is_file():
            continue
        shutil.copy2(src, train_dir / name)
        rows.append(
            {
                "file_name": name,
                "duration": f"{float(meta['duration']):.2f}",
                "transcription": meta["transcription"],
            }
        )
        used.add(name)

    bus_dir = Path("1_data_prepared/run2_buspark_segments")
    bus_meta = load_pack_meta(bus_dir)
    for name, meta in sorted(bus_meta.items()):
        if name in used:
            continue
        src = bus_dir / name
        if not src.is_file():
            continue
        shutil.copy2(src, train_dir / name)
        rows.append(
            {
                "file_name": name,
                "duration": f"{float(meta['duration']):.2f}",
                "transcription": meta["transcription"],
            }
        )
        used.add(name)


def add_extra_run3(train_dir: Path, rows: list[dict], used: set[str]):
    td2 = Path("1_data_prepared/training_data_2_segments")
    if not td2.is_dir():
        print("  WARN: training_data_2_segments missing — Run3 extras skipped")
        return
    meta = load_pack_meta(td2)
    for name, m in sorted(meta.items()):
        if name in used:
            continue
        src = td2 / name
        if not src.is_file():
            continue
        shutil.copy2(src, train_dir / name)
        rows.append(
            {
                "file_name": name,
                "duration": f"{float(m['duration']):.2f}",
                "transcription": m["transcription"],
            }
        )
        used.add(name)


def add_extra_run4(train_dir: Path, rows: list[dict], used: set[str]):
    dylaco = Path(
        "1_data_prepared/yrk_dylaco_Text1_preprocessed/yrk_dylaco_Text1_preprocessed"
    )
    if not dylaco.is_dir():
        print("  WARN: DyLaCo segments missing — Run4 extras skipped")
        return
    meta = load_pack_meta(dylaco)
    # Map Text1_NNNN -> text001_NNNN for inventory consistency optional;
    # keep on-disk names from the pack.
    for name, m in sorted(meta.items()):
        if not name.endswith(".wav"):
            continue
        if name in used:
            continue
        src = dylaco / name
        if not src.is_file():
            continue
        shutil.copy2(src, train_dir / name)
        text = m.get("transcription") or ""
        dur = m.get("duration")
        if dur is None:
            # fall back later if needed
            dur = "0"
        rows.append(
            {
                "file_name": name,
                "duration": f"{float(dur):.2f}",
                "transcription": text,
            }
        )
        used.add(name)


def build_run(run_n: int, gold_meta: dict, inventory: list[dict], exclude: set[str]):
    out = Path(OUTPUT_ROOT) / f"run{run_n}"
    if out.exists():
        shutil.rmtree(out)
    train_dir = out / "train"
    test_dir = out / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    allowed_runs = {f"Run{i}" for i in range(1, run_n + 1)}
    eval_names = []
    train_names = []

    for row in inventory:
        name = canon(row["Extracted file"].strip())
        split = row["Split (train/dev/test)"].strip()
        run = row["Run"].strip()
        if split == "dev" or run == "NA":
            if name not in eval_names:
                eval_names.append(name)
            continue
        if split != "train":
            continue
        if name in exclude:
            continue
        # Gold MapTask/PearStory rows are Run1; extras may be Run2/3/4
        if run in allowed_runs and name.startswith(("yrk_MapTask_", "yrk_PearStory_")):
            train_names.append(name)

    # Sanity: required swap
    for must_eval in ("yrk_PearStory_seg006.wav", "yrk_PearStory_seg063.wav"):
        if must_eval not in eval_names:
            raise RuntimeError(f"{must_eval} missing from eval holdout")
    for must_train in ("yrk_PearStory_seg005.wav", "yrk_PearStory_seg051.wav"):
        if must_train not in train_names:
            raise RuntimeError(f"{must_train} missing from Run1 train")
        if must_train in eval_names:
            raise RuntimeError(f"{must_train} must not be in eval")

    test_rows = [copy_gold(n, test_dir, gold_meta) for n in sorted(eval_names)]
    train_rows = [copy_gold(n, train_dir, gold_meta) for n in sorted(set(train_names))]
    used = {r["file_name"] for r in train_rows} | {r["file_name"] for r in test_rows}

    if run_n >= 2:
        add_extra_run2(train_dir, train_rows, used)
    if run_n >= 3:
        add_extra_run3(train_dir, train_rows, used)
    if run_n >= 4:
        add_extra_run4(train_dir, train_rows, used)

    write_metadata(train_dir, train_rows)
    write_metadata(test_dir, test_rows)

    print(
        f"run{run_n}: train={len(train_rows)} test={len(test_rows)} -> {out}"
    )
    pear_test = sorted(n for n in eval_names if "PearStory_seg00" in n or "PearStory_seg05" in n or "PearStory_seg06" in n)
    print(f"  eval PearStory swap files: {[n for n in eval_names if n in ('yrk_PearStory_seg005.wav','yrk_PearStory_seg006.wav','yrk_PearStory_seg051.wav','yrk_PearStory_seg063.wav')]}")
    return out


def sync_omnilingual_zs(gold_meta: dict, eval_names: list[str]):
    """Make OmnilingualZS packs match the frozen eval holdout."""
    train_root = Path("OmnilingualZS/corpus_entrainement")
    eval_root = Path("OmnilingualZS/corpus_evaluation")
    all_gold = sorted(gold_meta)
    exclude = set(read_name_list(EXCLUDE_LIST))
    train_names = [n for n in all_gold if n not in eval_names and n not in exclude]

    for folder, names in ((eval_root, eval_names), (train_root, train_names)):
        # remove wavs not in target
        for wav in folder.glob("*.wav"):
            if wav.name not in names:
                wav.unlink()
        for name in names:
            dest = folder / name
            if not dest.exists():
                shutil.copy2(Path(GOLD_DIR) / name, dest)

    def write_readme(folder: Path, names: list[str], title: str):
        lines = [
            f"# {title}",
            "",
            "| Fichier Audio | Transcription de Référence |",
            "| :--- | :--- |",
        ]
        for name in sorted(names):
            text = gold_meta[name]["transcription"].replace("|", "\\|")
            lines.append(f"| `{name}` | `{text}` |")
        lines.append("")
        (folder / "README.md").write_text("\n".join(lines), encoding="utf-8")

    write_readme(eval_root, eval_names, "Corpus d'évaluation (dev/test holdout)")
    write_readme(train_root, train_names, "Corpus d'entraînement (gold, hors holdout)")
    print(
        f"OmnilingualZS synced: train={len(train_names)} eval={len(eval_names)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        default="1,2,3,4",
        help="Comma-separated run numbers to build (default: 1,2,3,4)",
    )
    parser.add_argument(
        "--sync-omni",
        action="store_true",
        help="Also rewrite OmnilingualZS train/eval packs to match holdout",
    )
    args = parser.parse_args()

    eval_list = read_name_list(EVAL_LIST)
    exclude = set(read_name_list(EXCLUDE_LIST))
    gold_meta = load_gold_meta()
    inventory = load_inventory()

    # Cross-check CSV dev vs frozen list
    csv_dev = sorted(
        {
            canon(r["Extracted file"].strip())
            for r in inventory
            if r["Split (train/dev/test)"].strip() == "dev"
        }
    )
    if csv_dev != sorted(eval_list):
        raise RuntimeError(
            "CSV dev set != splits/eval_holdout.txt:\n"
            f"  only CSV: {set(csv_dev) - set(eval_list)}\n"
            f"  only file: {set(eval_list) - set(csv_dev)}"
        )

    for run_n in [int(x) for x in args.runs.split(",") if x.strip()]:
        build_run(run_n, gold_meta, inventory, exclude)

    if args.sync_omni:
        sync_omnilingual_zs(gold_meta, eval_list)

    # Convenience: point legacy plus_2oct26 at run2 layout (copy)
    run2 = Path(OUTPUT_ROOT) / "run2"
    legacy = Path("1_data_prepared/processed_audio_16k_plus_2oct26")
    if run2.is_dir():
        if legacy.exists():
            shutil.rmtree(legacy)
        shutil.copytree(run2, legacy)
        print(f"Updated legacy path {legacy} <- {run2}")


if __name__ == "__main__":
    main()
