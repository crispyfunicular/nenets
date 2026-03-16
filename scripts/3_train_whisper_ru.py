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
DATASET_PATH = "1_data_prepared/processed_audio_16k"
OUTPUT_DIR = "2_models/whisper-small-nenets-ru"
MODEL_ID = "openai/whisper-small"

EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_THRESHOLD = 0.01

def main():
    print("=" * 60)
    print("  WHISPER FINE-TUNING PIPELINE (Nenets) - RUSSIAN CONFIG")
    print("=" * 60)

    metadata_path = os.path.join(DATASET_PATH, "metadata.csv")
    if os.path.exists(metadata_path):
        print("Normalizing metadata.csv columns...")
        df_temp = pd.read_csv(metadata_path, sep=None, engine='python', encoding='utf-8-sig')
        if 'filename' in df_temp.columns:
            df_temp = df_temp.rename(columns={'filename': 'file_name'})
            df_temp.to_csv(metadata_path, index=False)
            print("Successfully renamed 'filename' to 'file_name'")

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

    print("\n[2/7] Loading Whisper processor...")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_ID, language="russian", task="transcribe")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="russian", task="transcribe")

    print("\n[3/7] Preprocessing audio and text labels...")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = tokenizer(batch[text_col]).input_ids
        return batch

    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names["train"],
        num_proc=1,
    )

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = jiwer.wer(label_str, pred_str)
        return {"wer": wer}

    print("\n[4/7] Loading pre-trained Whisper model...")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

    # Apply Russian forced decoder ids
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="russian", task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.suppress_tokens = []

    model.config.use_cache = False  # Required when using gradient checkpointing

    print(f"  Model parameters: {model.num_parameters() / 1e6:.0f}M")

    print("\n[5/7] Configuring training...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,
        num_train_epochs=100,
        learning_rate=1e-5,
        warmup_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        gradient_checkpointing=True,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
    )

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

    print(f"\nSaving best model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("=" * 60)
    print("  WHISPER TRAINING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
