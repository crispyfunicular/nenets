"""
Export eval-set hypotheses from a finetuned checkpoint (one prediction per line).

Used for bootstrap confidence intervals and qualitative review.
Order = sorted eval filenames (same as splits/eval_holdout.txt).

Usage:
  python scripts/12_export_eval_predictions.py \\
      --model-dir 2_models/wav2vec2-large-xlsr-nenets-run1 \\
      --model-type xlsr --run 1

  python scripts/12_export_eval_predictions.py \\
      --model-dir 2_models/whisper-small-nenets-ru-run1 \\
      --model-type whisper --run 1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


def resolve_dataset(run: int | None, dataset_path: str | None) -> str:
    if dataset_path:
        return dataset_path
    if run is None:
        run = int(os.environ.get("RUN", "1"))
    return f"1_data_prepared/experiment_runs/run{run}"


def _load_wav_mono16k(path: Path):
    """Load wav as float32 mono @16k without depending on datasets/torchcodec."""
    import numpy as np

    try:
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        import librosa

        audio, sr = librosa.load(str(path), sr=16000, mono=True)
        return np.asarray(audio, dtype=np.float32)

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio


def load_eval(dataset_path: str):
    """Return (paths, names) for the eval split, sorted by filename."""
    test_dir = Path(dataset_path) / "test"
    meta = test_dir / "metadata.csv"
    if not meta.is_file():
        raise RuntimeError(f"Missing {meta}")
    import pandas as pd

    df = pd.read_csv(meta, encoding="utf-8-sig")
    name_col = next(
        c for c in ("file_name", "filename", "file", "path") if c in df.columns
    )
    names = [Path(str(x)).name for x in df[name_col].tolist()]
    # Stable order for CI / manual inspection
    names = sorted(names)
    paths = [test_dir / n for n in names]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing wavs under {test_dir}: {missing[:3]}")
    return paths, names


def predict_xlsr(model_dir: str, paths, device):
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for path in paths:
            audio = _load_wav_mono16k(path)
            inputs = processor(
                audio, return_tensors="pt", sampling_rate=16000
            ).input_values.to(device)
            logits = model(inputs).logits
            ids = torch.argmax(logits, dim=-1)
            preds.append(processor.batch_decode(ids)[0].strip())
    return preds


def predict_whisper(model_dir: str, paths, device):
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(device)
    forced = processor.get_decoder_prompt_ids(language="russian", task="transcribe")
    model.config.forced_decoder_ids = forced
    model.generation_config.forced_decoder_ids = forced
    model.generation_config.suppress_tokens = []
    model.eval()
    preds = []
    with torch.no_grad():
        for path in paths:
            audio = _load_wav_mono16k(path)
            feats = processor(
                audio, return_tensors="pt", sampling_rate=16000
            ).input_features.to(device)
            ids = model.generate(feats, max_new_tokens=225)
            preds.append(
                processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            )
    return preds


def write_single_column(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    # One prediction per line, no header — single column for bootstrap CIs.
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-type", choices=("xlsr", "whisper"), required=True)
    parser.add_argument("--run", type=int, default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument(
        "--output",
        default=None,
        help="Default: <model-dir>/eval_predictions.txt",
    )
    args = parser.parse_args()

    dataset_path = resolve_dataset(args.run, args.dataset_path)
    out = Path(args.output or os.path.join(args.model_dir, "eval_predictions.txt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Model:   {args.model_dir} ({args.model_type})")
    print(f"Eval:    {dataset_path}/test")
    print(f"Device:  {device}")

    paths, names = load_eval(dataset_path)
    print(f"Samples: {len(paths)}")

    if args.model_type == "xlsr":
        preds = predict_xlsr(args.model_dir, paths, device)
    else:
        preds = predict_whisper(args.model_dir, paths, device)

    write_single_column(out, preds)
    # Sidecar with filenames (not required for CIs, useful for inspection)
    side = out.with_name(out.stem + "_files.txt")
    write_single_column(side, names)

    print(f"Wrote {len(preds)} predictions -> {out}")
    print(f"Wrote filename order       -> {side}")


if __name__ == "__main__":
    main()
