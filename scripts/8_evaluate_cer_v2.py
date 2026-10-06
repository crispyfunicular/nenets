import os
import torch
import jiwer
from datasets import load_dataset, Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

DATASET_PATH = "1_data_prepared/processed_audio_16k_combined"
WAV2VEC2_V1 = "2_models/wav2vec2-large-xlsr-nenets"
WAV2VEC2_V2 = "2_models/wav2vec2-large-xlsr-nenets-v2"


def load_wav2vec(model_path, device):
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
    model.eval()
    return processor, model


def transcribe(model, processor, audio, device):
    input_val = processor(
        audio, return_tensors="pt", sampling_rate=16000
    ).input_values.to(device)
    with torch.no_grad():
        logits = model(input_val).logits
        pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0].strip()


def maybe_load(label, path, device):
    if not os.path.isdir(path):
        print(f"Skipping {label}: {path} not found")
        return None
    try:
        processor, model = load_wav2vec(path, device)
        print(f"Loaded {label}: {path}")
        return {"label": label, "processor": processor, "model": model, "preds": []}
    except Exception as exc:
        print(f"Could not load {label} ({path}): {exc}")
        return None


def main():
    print("=" * 60)
    print("  EVALUATION WER/CER — Wav2Vec2 v1 vs v2 (frozen test)")
    print("=" * 60)

    print(f"\nLoading test split from {DATASET_PATH}...")
    dataset = load_dataset("audiofolder", data_dir=DATASET_PATH)
    if "test" not in dataset:
        raise ValueError(
            f"Expected a frozen test split in {DATASET_PATH}. "
            "Run scripts/11_merge_datasets.py first."
        )
    test_ds = dataset["test"]
    text_col = next(
        (col for col in ["transcription", "sentence", "text"]
         if col in test_ds.column_names),
        "sentence",
    )
    print(f"Test samples: {len(test_ds)}")
    if len(test_ds) != 17:
        print(f"WARNING: expected 17 frozen test files, got {len(test_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models = [
        maybe_load("Wav2Vec2 v1", WAV2VEC2_V1, device),
        maybe_load("Wav2Vec2 v2", WAV2VEC2_V2, device),
    ]
    models = [m for m in models if m is not None]
    if not models:
        raise FileNotFoundError(
            "No Wav2Vec2 checkpoints found. Train v2 with "
            "scripts/3_train_wav2vec_v2.py (and optionally place v1 in "
            f"{WAV2VEC2_V1})."
        )

    references = []
    test_ds = test_ds.cast_column("audio", Audio(sampling_rate=16000))
    print("\nRunning inference...")
    for i in range(len(test_ds)):
        sample = test_ds[i]
        audio = sample["audio"]["array"]
        references.append(sample[text_col])
        for entry in models:
            pred = transcribe(entry["model"], entry["processor"], audio, device)
            entry["preds"].append(pred)

    print("\n--- RESULTS (frozen 17-segment test) ---")
    scores = {}
    for entry in models:
        wer = jiwer.wer(references, entry["preds"])
        cer = jiwer.cer(references, entry["preds"])
        scores[entry["label"]] = (wer, cer)
        print(f"{entry['label']:<14} -> WER: {wer:.4f} | CER: {cer:.4f}")

    if "Wav2Vec2 v1" in scores and "Wav2Vec2 v2" in scores:
        v1_wer, v1_cer = scores["Wav2Vec2 v1"]
        v2_wer, v2_cer = scores["Wav2Vec2 v2"]
        print("\n--- DELTA (v2 - v1, negative = improvement) ---")
        print(f"WER: {v2_wer - v1_wer:+.4f}")
        print(f"CER: {v2_cer - v1_cer:+.4f}")
        if v2_cer <= v1_cer and v2_wer <= v1_wer:
            print("v2 is better or equal on both metrics.")
        elif v2_cer > v1_cer or v2_wer > v1_wer:
            print(
                "v2 is worse on at least one metric; extra data may have "
                "injected noise (code-switching / orthography)."
            )


if __name__ == "__main__":
    main()
