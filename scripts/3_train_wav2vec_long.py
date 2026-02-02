import os
import json
import jiwer
import pandas as pd
from datasets import load_dataset, Audio, load_metric
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
)
import numpy as np

# --- CONFIGURATION ---
# Input: The "Gold Standard" data prepared from Nikolett's TextGrids
DATASET_PATH = "1_data_prepared/processed_audio_16k" 
# Output: Where the brain of the AI is stored
OUTPUT_DIR = "2_models/wav2vec2-large-xlsr-nenets"

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

    # 1. LOAD DATASET (Maintenant ça ne plantera plus)
    dataset = load_dataset("audiofolder", data_dir=DATASET_PATH)
    
    if "test" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    
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
        with processor.as_target_processor():
            batch["labels"] = processor(batch[text_col]).input_ids
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
            with self.processor.as_target_processor():
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
        MODEL_NAME, 
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    model.freeze_feature_extractor()

    # 6. TRAINING ARGUMENTS
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        group_by_length=True,
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
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=processor.feature_extractor,
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving best model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Process completed successfully.")

if __name__ == "__main__":
    main()