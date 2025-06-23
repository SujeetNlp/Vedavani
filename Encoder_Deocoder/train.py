import os
import torch
import argparse
from datasets import load_from_disk
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

import evaluate

# ------------------------------
# Data Collator Definition
# ------------------------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# ------------------------------
# Metrics Computation
# ------------------------------
def compute_metrics(pred, tokenizer):
    metric_wer = evaluate.load("wer")
    metric_cer = evaluate.load("cer")

    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)
    cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}

# ------------------------------
# Main Function
# ------------------------------
def main(args):
    # Load dataset
    dataset = load_from_disk(args.dataset_path)

    # Filter
    max_input_length = 30.0 * 16000
    dataset = dataset.filter(lambda x: 0 < x["input_length"] < max_input_length)
    dataset = dataset.filter(lambda x: x["labels_length"] < 448)

    # Load processor, tokenizer, model
    processor = WhisperProcessor.from_pretrained(args.model_path, language="sanskrit", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained(args.model_path, language="sanskrit", task="transcribe")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path, ignore_mismatched_sizes=True)

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="sanskrit", task="transcribe")
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = model.config.forced_decoder_ids
    model.generation_config.suppress_tokens = []

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=15000 * 8,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=10000,
        eval_steps=10000,
        logging_steps=25,
        report_to=["tensorboard"],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        fp16_full_eval=False,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor),
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer),
        tokenizer=feature_extractor,
    )

    processor.save_pretrained(training_args.output_dir)

    # Train
    trainer.train()

# ------------------------------
# Argument Parsing
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/sujeet-pg/whisper/saved_data_after_map_devnagri",
        help="Path to the processed dataset directory",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/sujeet-pg/whisper/sanskrit_models/whisper-medium-sa_alldata_multigpu",
        help="Path to the pre-trained Whisper model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/sujeet-pg/whisper/indicwhisper_devns",
        help="Directory to save model checkpoints and logs",
    )

    args = parser.parse_args()
    main(args)