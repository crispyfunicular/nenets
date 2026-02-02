import os
import json
import numpy as np
import torch
import pandas as pd
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from dataclasses import dataclass
from typing import Dict, List, Union
import evaluate

# --- Configuration ---

# Directory containing the training data (the 'metadata.csv' and audio files.) (Gold Standard).
DATASET_PATH = "dataset_final" 

# Output directory for the trained model: where the fine-tuned model and checkpoints will be saved.
OUTPUT_DIR = "wav2vec2-large-xlsr-nenets"

# Base pre-trained model (Facebook XLSR-53).
MODEL_ID = "facebook/wav2vec2-large-xlsr-53"

# --- Hyperparameters (Server Config) ---
# Set to 4 to fit within the available ~13GB VRAM on the server.
# Increasing this might cause CUDA Out Of Memory errors.
BATCH_SIZE = 4  
LEARNING_RATE = 3e-4
NUM_EPOCHS = 30 

def main():
    print("Starting training pipeline...")
    print(f"Data source: {DATASET_PATH}")
    
    # ---------------------------------------------------------
    # 1. DATA LOADING AND PREPARATION
    # ---------------------------------------------------------

    # Verify metadata existence
    metadata_path = os.path.join(DATASET_PATH, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Could not find {metadata_path}.")

    # Load the dataset using Hugging Face's 'datasets' library
    dataset = load_dataset("csv", data_files=metadata_path, split="train")
    
    # Ensure all audio paths are absolute system paths
    def update_path(batch):
        batch["audio"] = os.path.join(os.path.abspath(DATASET_PATH), batch["filename"])
        return batch

    dataset = dataset.map(update_path)

    # Critical Step: Resample audio to 16kHz.
    # Wav2Vec2 models are pre-trained on 16kHz audio; inputs must match this rate.
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Split dataset: 90% for training, 10% for validation/testing
    dataset = dataset.train_test_split(test_size=0.1)
    print(f"Training samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['test'])}")


    # ---------------------------------------------------------
    # 2. VOCABULARY CREATION (Character Level)
    # ---------------------------------------------------------

    print("Building vocabulary...")

    # Extract all unique characters from the transcriptions to build the alphabet 
    def extract_all_chars(batch):
        all_text = " ".join(batch["transcription"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocab_train = extract_all_chars(dataset["train"])
    vocab_test = extract_all_chars(dataset["test"])
    
    # Merge vocabularies to ensure all characters are covered
    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    
    # Add special tokens required for CTC (Connectionist Temporal Classification) decoding
    # Replace the standard space with a visible delimiter '|'
    vocab_dict["|"] = vocab_dict[" "] 
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    
    # Save the vocabulary to a JSON file for future inference
    with open("vocab.json", "w", encoding="utf-8") as vocab_file:
        json.dump(vocab_dict, vocab_file)

    # ---------------------------------------------------------
    # 3. PROCESSOR INITIALIZATION
    # ---------------------------------------------------------
    
    # The processor wraps the Feature Extractor (Audio -> Vectors) and Tokenizer (Vectors -> Text)
    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


    # ---------------------------------------------------------
    # 4. PREPROCESSING (Vectorization)
    # ---------------------------------------------------------

    print("Preprocessing audio and text labels...")

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["labels"] = processor(text=batch["transcription"]).input_ids
        return batch

    encoded_dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names, num_proc=1)


    # ---------------------------------------------------------
    # 5. DATA COLLATOR (Dynamic Padding)
    # ---------------------------------------------------------
    # Handles batch preparation by dynamically padding audio and text to the length of the longest sample in the batch.@dataclass
    class DataCollatorCTCWithPadding:
        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.feature_extractor.pad(
                input_features,
                padding=self.padding,
                return_tensors="pt",
            )
            labels_batch = self.processor.tokenizer.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}


    # ---------------------------------------------------------
    # 6. MODEL INITIALIZATION
    # ---------------------------------------------------------

    print("Loading pre-trained model...")

    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    # Optimization: Freeze the feature encoder (CNN layers).
    # Since these layers are already well-trained on massive datasets, freezing them saves memory
    # and prevents catastrophic forgetting of low-level acoustic features.
    model.freeze_feature_extractor()


    # ---------------------------------------------------------
    # 7. TRAINING ARGUMENTS
    # ---------------------------------------------------------

    training_args = TrainingArguments(
        output_dir="wav2vec2-large-xlsr-nenets-long",
        group_by_length=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        
        # --- LES CHANGEMENTS SONT ICI ---
        evaluation_strategy="steps",
        num_train_epochs=100,
        save_steps=500, 
        eval_steps=500,
        
        load_best_model_at_end=True, 
        metric_for_best_model="loss",
        save_total_limit=2,
        # -------------------------------
        
        fp16=True,
        gradient_checkpointing=True, 
        logging_steps=50,
        learning_rate=1e-4,
        warmup_steps=100,
    )


    # ---------------------------------------------------------
    # 8. TRAINER EXECUTION
    # ---------------------------------------------------------
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=processor.feature_extractor,
    )

    print("Starting training...")
    trainer.train()
    

    # ---------------------------------------------------------
    # 9. FINAL SAVE
    # ---------------------------------------------------------
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Process completed successfully.")


if __name__ == "__main__":
    main()