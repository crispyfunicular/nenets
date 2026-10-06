import os
import json
import jiwer
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Union

from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# --- CONFIGURATION ---
DATASET_PATH = "1_data_prepared/processed_audio_16k_combined"
OUTPUT_DIR = "2_models/wav2vec2-large-xlsr-nenets-v2"
MODEL_ID = "facebook/wav2vec2-large-xlsr-53"
VOCAB_PATH = os.path.join(OUTPUT_DIR, "vocab.json")

EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_THRESHOLD = 0.01


def main():
    print("=" * 60)
    print("  WAV2VEC2 FINE-TUNING v2 (original gold + training data 2)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/7] Loading dataset...")
    dataset = load_dataset("audiofolder", data_dir=DATASET_PATH)
    if "test" not in dataset:
        raise ValueError(
            f"Expected a frozen test split in {DATASET_PATH}. "
            "Run scripts/11_merge_datasets.py first."
        )

    text_col = next(
        (col for col in ["transcription", "sentence", "text"]
         if col in dataset["train"].column_names),
        "sentence",
    )
    print(f"  Training samples: {len(dataset['train'])}")
    print(f"  Frozen test samples: {len(dataset['test'])}")
    print(f"  Text column: '{text_col}'")

    print("\n[2/7] Building vocabulary from train+test...")

    def extract_all_chars(batch):
        all_text = " ".join(batch[text_col])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=dataset.column_names["train"],
    )
    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(VOCAB_PATH, "w", encoding="utf-8") as vocab_file:
        json.dump(vocab_dict, vocab_file, ensure_ascii=False)
    print(f"  Vocab size: {len(vocab_dict)} -> {VOCAB_PATH}")

    tokenizer = Wav2Vec2CTCTokenizer(
        VOCAB_PATH, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    print("\n[3/7] Preprocessing audio and labels...")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        batch["labels"] = processor.tokenizer(batch[text_col]).input_ids
        return batch

    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names["train"],
        num_proc=1,
    )

    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True

        def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            batch = self.processor.feature_extractor.pad(
                input_features, padding=self.padding, return_tensors="pt"
            )
            labels_batch = self.processor.tokenizer.pad(
                label_features, padding=self.padding, return_tensors="pt"
            )
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = jiwer.wer(label_str, pred_str)
        return {"wer": wer}

    print("\n[4/7] Loading pre-trained XLSR-53...")
    use_cuda = torch.cuda.is_available()
    print(f"  Device: {'cuda' if use_cuda else 'cpu'}")
    if use_cuda:
        train_bs, accum = 4, 2
    else:
        # CPU: keep effective batch size 8, avoid fp16 (CUDA-only)
        train_bs, accum = 1, 8
        print("  No GPU detected — training on CPU (fp16 disabled, batch=1 x accum=8).")

    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    else:
        model.freeze_feature_extractor()

    print("\n[5/7] Configuring training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=train_bs,
        gradient_accumulation_steps=accum,
        eval_strategy="steps",
        num_train_epochs=100,
        fp16=use_cuda,
        gradient_checkpointing=True,
        save_steps=200,
        eval_steps=200,
        logging_steps=25 if not use_cuda else 50,
        learning_rate=1e-4,
        warmup_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        report_to=[],
    )

    print("\n[6/7] Initializing trainer...")
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

    print("\n[7/7] Starting training...")
    print("-" * 60)
    trainer.train()

    print(f"\nSaving best model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("=" * 60)
    print("  WAV2VEC2 v2 TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
