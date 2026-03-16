import os
import json
import jiwer
import pandas as pd
from datasets import load_dataset, Audio
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
import numpy as np

# --- CONFIGURATION ---
# Input: Same Gold Standard data as Wav2Vec2
DATASET_PATH = "1_data_prepared/processed_audio_16k"
# Output: Separate folder for the Whisper model (does NOT overwrite Wav2Vec2)
OUTPUT_DIR = "2_models/whisper-small-nenets"
# Base pre-trained model
MODEL_ID = "openai/whisper-small"

# --- EARLY STOPPING ---
# Stop training when eval_loss stops improving to prevent overfitting.
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_THRESHOLD = 0.01

def main():
    print("=" * 60)
    print("  WHISPER FINE-TUNING PIPELINE (Nenets)")
    print("=" * 60)

    # --- FIX: Normalize CSV columns (same as W2V script) ---
    metadata_path = os.path.join(DATASET_PATH, "metadata.csv")
    if os.path.exists(metadata_path):
        print("Normalizing metadata.csv columns...")
        df_temp = pd.read_csv(metadata_path, sep=None, engine='python', encoding='utf-8-sig')
        if 'filename' in df_temp.columns:
            df_temp = df_temp.rename(columns={'filename': 'file_name'})
            df_temp.to_csv(metadata_path, index=False)
            print("Successfully renamed 'filename' to 'file_name'")

    # ---------------------------------------------------------
    # 1. LOAD DATASET
    # ---------------------------------------------------------
    print("\n[1/7] Loading dataset...")
    dataset = load_dataset("audiofolder", data_dir=DATASET_PATH)

    if "test" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    text_col = next(
        (col for col in ["transcription", "sentence", "text"]
         if col in dataset["train"].column_names),
        "sentence"
    )

    print(f"  Training samples: {len(dataset['train'])}")
    print(f"  Validation samples: {len(dataset['test'])}")
    print(f"  Text column: '{text_col}'")

    # ---------------------------------------------------------
    # 2. PROCESSOR INITIALIZATION
    # ---------------------------------------------------------
    print("\n[2/7] Loading Whisper processor...")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_ID, language=None, task="transcribe")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=None, task="transcribe")

    # ---------------------------------------------------------
    # 3. PREPROCESSING
    # ---------------------------------------------------------
    print("\n[3/7] Preprocessing audio and text labels...")

    # Ensure audio is 16kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def prepare_dataset(batch):
        audio = batch["audio"]

        # Compute log-Mel input features from the audio array
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # Encode target text to label ids
        batch["labels"] = tokenizer(batch[text_col]).input_ids
        return batch

    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names["train"],
        num_proc=1,
    )

    # ---------------------------------------------------------
    # 4. DATA COLLATOR
    # ---------------------------------------------------------
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # Split inputs and labels since they have different padding requirements
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # Tokenize and pad labels
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # Replace padding with -100 so it's ignored in loss computation
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            # Remove BOS token if the model prepends it during generation
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # ---------------------------------------------------------
    # 5. METRICS
    # ---------------------------------------------------------
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # Decode predictions and references
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = jiwer.wer(label_str, pred_str)
        return {"wer": wer}

    # ---------------------------------------------------------
    # 6. MODEL INITIALIZATION
    # ---------------------------------------------------------
    print("\n[4/7] Loading pre-trained Whisper model...")

    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

    # CRITICAL for unsupported languages like Nenets:
    # Disable forced decoder IDs (language/task tokens) so the model
    # doesn't try to force a known language.
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    # Enable gradient checkpointing to save VRAM
    model.config.use_cache = False  # Required when using gradient checkpointing

    print(f"  Model parameters: {model.num_parameters() / 1e6:.0f}M")

    # ---------------------------------------------------------
    # 7. TRAINING ARGUMENTS
    # ---------------------------------------------------------
    print("\n[5/7] Configuring training...")

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        # --- Batch size: small to fit in 8GB VRAM ---
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch = 2 * 4 = 8
        per_device_eval_batch_size=2,
        # --- Training schedule ---
        num_train_epochs=100,
        learning_rate=1e-5,  # Lower LR than W2V (Whisper is already well-trained)
        warmup_steps=50,
        # --- Evaluation & checkpoints ---
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # --- Memory optimization ---
        fp16=True,
        gradient_checkpointing=True,
        # --- Generation during eval ---
        predict_with_generate=True,
        generation_max_length=225,
        # --- Logging ---
        logging_steps=25,
    )

    # ---------------------------------------------------------
    # 8. TRAINER EXECUTION
    # ---------------------------------------------------------
    print("\n[6/7] Initializing trainer...")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
        )],
    )

    print("\n[7/7] Starting training...")
    print("-" * 60)
    trainer.train()

    # ---------------------------------------------------------
    # 9. FINAL SAVE
    # ---------------------------------------------------------
    print(f"\nSaving best model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("=" * 60)
    print("  WHISPER TRAINING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
