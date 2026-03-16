import os
import torch
import jiwer
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration, WhisperProcessor,
    Wav2Vec2ForCTC, Wav2Vec2Processor
)

DATASET_PATH = "1_data_prepared/processed_audio_16k"
WHISPER_MODEL = "2_models/whisper-large-v3-nenets-ru"
WAV2VEC2_MODEL = "2_models/wav2vec2-large-xlsr-nenets"

def main():
    print("Loading test dataset...")
    dataset = load_dataset("audiofolder", data_dir=DATASET_PATH)
    if "test" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    test_ds = dataset["test"]

    text_col = next(
        (col for col in ["transcription", "sentence", "text"]
         if col in test_ds.column_names),
        "sentence"
    )

    print(f"Test samples: {len(test_ds)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Whisper
    try:
        w_proc = WhisperProcessor.from_pretrained(WHISPER_MODEL)
        w_model = WhisperForConditionalGeneration.from_pretrained(
            WHISPER_MODEL,
            torch_dtype=torch.bfloat16,
        ).to(device)
        forced_decoder_ids = w_proc.get_decoder_prompt_ids(language="russian", task="transcribe")
        w_model.config.forced_decoder_ids = forced_decoder_ids
        w_model.generation_config.forced_decoder_ids = forced_decoder_ids
        w_model.generation_config.suppress_tokens = []
        w_model.eval()
        whisper_ok = True
    except Exception as e:
        print(f"Could not load Whisper Large: {e}")
        whisper_ok = False

    references = []
    w_preds = []

    print("Running inference...")
    test_ds = test_ds.cast_column("audio", Audio(sampling_rate=16000))
    for i in range(len(test_ds)):
        sample = test_ds[i]
        audio = sample["audio"]["array"]
        ref = sample[text_col]
        references.append(ref)

        with torch.no_grad():
            if whisper_ok:
                input_feat = w_proc(audio, return_tensors="pt", sampling_rate=16000).input_features.to(torch.bfloat16).to(device)
                pred_ids = w_model.generate(input_feat, max_new_tokens=225)
                w_pred = w_proc.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
                w_preds.append(w_pred)

    print("\n--- RESULTS ---")
    if whisper_ok:
        w_wer = jiwer.wer(references, w_preds)
        w_cer = jiwer.cer(references, w_preds)
        print(f"Whisper Large (RU) -> WER: {w_wer:.4f} | CER: {w_cer:.4f}")

if __name__ == '__main__':
    main()
