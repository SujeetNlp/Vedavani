#!/usr/bin/env python3
"""train.py – fine-tune Whisper on a mapped DatasetDict.

Required CLI arguments:
    --dataset_dir   Path to the *mapped* DatasetDict saved by preprocess.py
    --model_path    Base Whisper checkpoint (local dir or HF hub id)
    --output_dir    Where to write checkpoints / logs / final processor

Example
    python train.py \
        --dataset_dir /data/mapped_dataset \
        --model_path  openai/whisper-medium \
        --output_dir  /results/whisper_finetuned
"""

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import evaluate
from datasets import load_from_disk
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# ――― constants ―――
MAX_DURATION_SECONDS = 30.0
SAMPLE_RATE = 16000
MAX_INPUT_LENGTH = int(MAX_DURATION_SECONDS * SAMPLE_RATE)
MAX_LABEL_LENGTH = 448


# ――― helpers ―――

def filter_inputs(input_length: int) -> bool:
    return 0 < input_length < MAX_INPUT_LENGTH


def filter_labels(labels_length: int) -> bool:
    return labels_length < MAX_LABEL_LENGTH


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # audio
        inputs = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")

        # labels
        label_feats = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_feats, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# ――― metric fn ―――
metric_wer = evaluate.load("wer")
metric_cer = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)
    cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}


# ――― main ―――

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper")
    parser.add_argument("--dataset_dir", required=True, help="Path to mapped DatasetDict (output_dir2 from preprocess.py)")
    parser.add_argument("--model_path", required=True, help="HF model id or local directory for Whisper checkpoint")
    parser.add_argument("--output_dir", required=True, help="Where to save checkpoints, logs, and the final processor")
    # optional quick knobs
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=1200)
    parser.add_argument("--eval_steps", type=int, default=300)
    args = parser.parse_args()

    # ――― load dataset ―――
    ds = load_from_disk(args.dataset_dir)
    ds = ds.filter(filter_inputs, input_columns=["input_length"])
    ds = ds.filter(filter_labels, input_columns=["labels_length"])

    # ――― processor/model ―――
    global tokenizer  # used in compute_metrics
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_path)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_path, language="sanskrit", task="transcribe")
    processor = WhisperProcessor.from_pretrained(args.model_path, language="sanskrit", task="transcribe")

    model = WhisperForConditionalGeneration.from_pretrained(args.model_path, ignore_mismatched_sizes=True)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="sanskrit", task="transcribe")
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = model.config.forced_decoder_ids
    model.generation_config.suppress_tokens = []

    # move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ――― training setup ―――
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.eval_steps // 10,
        report_to=["tensorboard"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["valid"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # save processor so it travels with the model
    processor.save_pretrained(args.output_dir)

    trainer.train()


if __name__ == "__main__":
    main()