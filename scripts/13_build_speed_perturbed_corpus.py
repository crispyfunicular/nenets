#!/usr/bin/env python3
"""
Build speed-perturbed experiment corpora (atempo 0.9 + 1.1) for inventory runs.

For each run N under experiment_runs/runN/:
  train/ → originals + *_0.9x.wav + *_1.1x.wav (same transcriptions)
  test/  → originals only (frozen holdout, no perturbation)

Output: 1_data_prepared/experiment_runs_sp/run{N}/

Note: Aleksandra's note had a typo (atempo=0.1 for slow-down). We use atempo=0.9
for the 0.9× files and atempo=1.1 for the 1.1× files (ffmpeg filter).

Usage (repo root, ideally on MoDyCo where wavs already exist):
  python scripts/13_build_speed_perturbed_corpus.py
  python scripts/13_build_speed_perturbed_corpus.py --runs 1 2 --jobs 8
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SRC_ROOT = Path("1_data_prepared/experiment_runs")
DST_ROOT = Path("1_data_prepared/experiment_runs_sp")
TEMPOS = (("0.9", 0.9), ("1.1", 1.1))


def stem_suffix(name: str, label: str) -> str:
    p = Path(name)
    return f"{p.stem}_{label}x{p.suffix}"


def ffmpeg_atempo(src: Path, dst: Path, tempo: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-filter:a",
        f"atempo={tempo}",
        str(dst),
    ]
    subprocess.check_call(cmd)


def read_meta(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_meta(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
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


def copy_split(src_dir: Path, dst_dir: Path, rows: list[dict]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        name = row["file_name"]
        src = src_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing audio: {src}")
        shutil.copy2(src, dst_dir / name)


def build_run(run: int, jobs: int, force: bool) -> None:
    src = SRC_ROOT / f"run{run}"
    dst = DST_ROOT / f"run{run}"
    if not (src / "train" / "metadata.csv").is_file():
        raise FileNotFoundError(f"Missing source corpus: {src}/train/metadata.csv")
    if not (src / "test" / "metadata.csv").is_file():
        raise FileNotFoundError(f"Missing source corpus: {src}/test/metadata.csv")

    if dst.exists():
        if force:
            shutil.rmtree(dst)
        else:
            print(f"[skip] {dst} exists (use --force to rebuild)")
            return

    train_rows = read_meta(src / "train" / "metadata.csv")
    test_rows = read_meta(src / "test" / "metadata.csv")

    train_dst = dst / "train"
    test_dst = dst / "test"
    copy_split(src / "train", train_dst, train_rows)
    copy_split(src / "test", test_dst, test_rows)

    # Perturb train only
    tasks = []
    for row in train_rows:
        src_wav = src / "train" / row["file_name"]
        for label, tempo in TEMPOS:
            out_name = stem_suffix(row["file_name"], label)
            tasks.append((src_wav, train_dst / out_name, tempo, label, row))

    print(f"run{run}: perturbing {len(train_rows)} train files × 2 ({len(tasks)} ffmpeg jobs, jobs={jobs})")
    errors = []

    def _one(item):
        src_wav, out_wav, tempo, label, row = item
        try:
            ffmpeg_atempo(src_wav, out_wav, tempo)
            return None
        except Exception as exc:  # noqa: BLE001 — collect and report
            return f"{src_wav.name} @ {tempo}: {exc}"

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_one, t) for t in tasks]
        done = 0
        for fut in as_completed(futures):
            err = fut.result()
            done += 1
            if err:
                errors.append(err)
            if done % 50 == 0 or done == len(tasks):
                print(f"  ffmpeg {done}/{len(tasks)}")

    if errors:
        for e in errors[:20]:
            print("ERROR:", e, file=sys.stderr)
        raise RuntimeError(f"{len(errors)} ffmpeg failures for run{run}")

    out_train = list(train_rows)
    for row in train_rows:
        dur = float(row["duration"])
        text = row["transcription"]
        for label, tempo in TEMPOS:
            out_train.append(
                {
                    "file_name": stem_suffix(row["file_name"], label),
                    "duration": f"{dur / tempo:.2f}",
                    "transcription": text,
                }
            )

    write_meta(train_dst / "metadata.csv", out_train)
    write_meta(test_dst / "metadata.csv", test_rows)
    print(
        f"run{run}: train={len(out_train)} (3×{len(train_rows)}) "
        f"test={len(test_rows)} -> {dst}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--jobs", type=int, default=8, help="parallel ffmpeg workers")
    parser.add_argument("--force", action="store_true", help="rebuild even if dest exists")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")

    if not SRC_ROOT.is_dir():
        raise SystemExit(f"Missing {SRC_ROOT} — build experiment_runs first")

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        build_run(run, jobs=args.jobs, force=args.force)
    print("Done.")


if __name__ == "__main__":
    main()
