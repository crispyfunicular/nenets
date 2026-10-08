#!/usr/bin/env python3
"""
Resample wavs under experiment_runs_sp/ to 16 kHz (in place).

Does NOT touch experiment_runs/, raw packs, or any other source data —
only the derived speed-perturbed corpora.

Usage:
  python scripts/14_resample_sp_corpus_16k.py --runs 4
  python scripts/14_resample_sp_corpus_16k.py --runs 1 2 3 4 --jobs 8
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("1_data_prepared/experiment_runs_sp")
TARGET_SR = 16000


def probe_sr(path: Path) -> int | None:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=sample_rate",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            text=True,
        ).strip()
        return int(out) if out else None
    except Exception:
        return None


def resample_one(path: Path) -> str:
    sr = probe_sr(path)
    if sr == TARGET_SR:
        return "skip"
    if sr is None:
        return f"fail:probe:{path.name}"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=path.parent) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.check_call(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-ar",
                str(TARGET_SR),
                "-ac",
                "1",
                str(tmp_path),
            ]
        )
        tmp_path.replace(path)
        return f"ok:{sr}->{TARGET_SR}"
    except Exception as exc:  # noqa: BLE001
        if tmp_path.exists():
            tmp_path.unlink()
        return f"fail:{path.name}:{exc}"


def process_run(run: int, jobs: int) -> None:
    train = ROOT / f"run{run}" / "train"
    test = ROOT / f"run{run}" / "test"
    if not train.is_dir():
        raise FileNotFoundError(train)
    files = sorted(train.glob("*.wav")) + sorted(test.glob("*.wav"))
    print(f"run{run}: scanning {len(files)} wavs under {ROOT / f'run{run}'} only")
    ok = skip = fail = 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(resample_one, p): p for p in files}
        done = 0
        for fut in as_completed(futs):
            done += 1
            res = fut.result()
            if res == "skip":
                skip += 1
            elif res.startswith("ok:"):
                ok += 1
            else:
                fail += 1
                print(" ", res, file=sys.stderr)
            if done % 200 == 0 or done == len(files):
                print(f"  {done}/{len(files)} (resampled={ok} already16k={skip} fail={fail})")
    if fail:
        raise RuntimeError(f"run{run}: {fail} failures")
    print(f"run{run}: done resampled={ok} already16k={skip}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=int, default=[4])
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if not ROOT.is_dir():
        raise SystemExit(f"Missing {ROOT} (derived SP corpora only)")
    for run in args.runs:
        process_run(run, args.jobs)


if __name__ == "__main__":
    main()
