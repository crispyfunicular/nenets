import os
import sys
import json
import jiwer
import pandas as pd
from pathlib import Path
from datasets import load_dataset, Audio
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_audiofolder_compat

hf_audiofolder_compat.enable()

# --- CONFIGURATION ---
# Inventory-defined splits (see scripts/11_build_experiment_corpus.py).
# RUN=1..4 selects cumulative training data; eval holdout is always the same.
RUN = int(os.environ.get("RUN", "1"))
DATASET_PATH = os.environ.get(
    "DATASET_PATH", f"1_data_prepared/experiment_runs/run{RUN}"
)
# Output: Where the brain of the AI is stored
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR", f"2_models/wav2vec2-large-xlsr-nenets-run{RUN}"
)
# Base pre-trained model (Facebook XLSR-53)
MODEL_ID = "facebook/wav2vec2-large-xlsr-53"

# --- EARLY STOPPING / BEST CHECKPOINT ---
# Select and early-stop on eval WER (lower is better), not eval_loss.
# Override via env for long SP runs that need more steps before WER moves.
EARLY_STOPPING_PATIENCE = int(os.environ.get("EARLY_STOPPING_PATIENCE", "5"))
EARLY_STOPPING_THRESHOLD = float(os.environ.get("EARLY_STOPPING_THRESHOLD", "0.01"))

def main():
    print("Starting training pipeline (LONG VERSION)...")

    # --- FIX: Rename column in CSV before loading ---
    metadata_path = os.path.join(DATASET_PATH, "metadata.csv")
    if os.path.exists(metadata_path):
        print("Normalizing metadata.csv columns...")
        # On lit avec utf-8-sig pour virer le BOM Windows si présent
        df_temp = pd.read_csv(metadata_path, sep=None, engine='python', encoding='utf-8-sig')
        if 'filename' in df_temp.columns:
            df_temp = df_temp.rename(columns={'filename': 'file_name'})
            df_temp.to_csv(metadata_path, index=False)
            print("Successfully renamed 'filename' to 'file_name' in metadata.csv")

    # 1. LOAD DATASET — train/test dirs from inventory (no seed=42 resplit)
    print(f"Dataset: {DATASET_PATH} (RUN={RUN})")
    dataset = load_dataset("audiofolder", data_dir=DATASET_PATH)
    if "test" not in dataset:
        raise RuntimeError(
            f"{DATASET_PATH} has no test/ split. "
            "Run: python scripts/11_build_experiment_corpus.py --sync-omni"
        )
    
    text_col = next((col for col in ["transcription", "sentence", "text"] if col in dataset["train"].column_names), "sentence")

    # 2. VOCABULARY & TOKENIZER
    def extract_all_chars(batch):
        all_text = " ".join(batch[text_col])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=dataset.column_names["train"])
    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    
    with open("vocab.json", "w", encoding="utf-8") as vocab_file:
        json.dump(vocab_dict, vocab_file)

    tokenizer = Wav2Vec2CTCTokenizer("vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # 3. PREPROCESSING
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["labels"] = processor.tokenizer(batch[text_col]).input_ids
        return batch

    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=1)

    # 4. DATA COLLATOR & METRICS
    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, padding=self.padding, return_tensors="pt")
            labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        
        # Utilisation directe de jiwer (plus robuste)
        wer = jiwer.wer(label_str, pred_str)
        return {"wer": wer}

    # 5. MODEL INIT
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID, 
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    model.freeze_feature_encoder()

    # 6. TRAINING ARGUMENTS
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        num_train_epochs=100,
        fp16=True,
        gradient_checkpointing=True, 
        save_steps=200,
        eval_steps=200,
        logging_steps=50,
        learning_rate=1e-4,
        warmup_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=0,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
        )],
    )

    print(
        f"Starting training... (early_stopping_patience={EARLY_STOPPING_PATIENCE}, "
        f"threshold={EARLY_STOPPING_THRESHOLD})"
    )
    trainer.train()
    
    print(f"Saving best model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    # Free GPU before a separate export process (shared MoDyCo GPU).
    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Point 4: single-column eval hypotheses for bootstrap CIs
    print("Exporting eval predictions (best checkpoint)...")
    import subprocess

    export_env = os.environ.copy()
    # Prefer CPU for export if another job already holds most of the GPU.
    if torch.cuda.is_available():
        free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
        if free_mb < 4000:
            export_env["CUDA_VISIBLE_DEVICES"] = ""
            print(f"Export on CPU (only {free_mb:.0f} MiB GPU free)")

    subprocess.check_call(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "12_export_eval_predictions.py"),
            "--model-dir",
            OUTPUT_DIR,
            "--model-type",
            "xlsr",
            "--dataset-path",
            DATASET_PATH,
        ],
        env=export_env,
    )
    print("Process completed successfully.")

if __name__ == "__main__":
    main()