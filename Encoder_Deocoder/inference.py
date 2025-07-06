"""inference.py – generate transcriptions with a fine‑tuned Whisper model.

Required CLI arguments
    --dataset_dir    Path to *raw* DatasetDict saved by preprocess.py (output_dir1)
    --orig_out_txt   Output file to save the ground‑truth sentences (one per line)
    --pred_out_txt   Output file to save the model transcriptions (one per line)

Optional arguments
    --model_path     Fine‑tuned Whisper checkpoint (local dir or HF id). Default: "openai/whisper-medium"
    --split          Dataset split to run on (default: "test")
    --max_samples    Limit number of samples (useful for quick sanity checks)

Example
    python inference.py \
        --dataset_dir /data/raw_dataset \
        --orig_out_txt references.txt \
        --pred_out_txt predictions.txt \
        --model_path  /results/whisper_finetuned \
        --split test
"""

import argparse
import os
from typing import List

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def load_model(model_path: str):
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, device


def transcribe_dataset(dataset, processor, model, device, max_samples: int | None = None) -> List[str]:
    predictions = []
    for idx, item in tqdm(enumerate(dataset), total=min(len(dataset), max_samples or len(dataset))):
        if max_samples is not None and idx >= max_samples:
            break
        waveform = item["audio"]["array"]
        with torch.no_grad():
            inputs = processor(waveform, sampling_rate=16_000, return_tensors="pt").input_features.to(device)
            forced_ids = processor.get_decoder_prompt_ids(language="sanskrit", task="transcribe")
            generated_ids = model.generate(inputs, forced_decoder_ids=forced_ids)
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        predictions.append(transcription)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a Whisper checkpoint")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--orig_out_txt", required=True)
    parser.add_argument("--pred_out_txt", required=True)
    parser.add_argument("--model_path", default="openai/whisper-small")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    # ――― dataset ―――
    ds = load_from_disk(args.dataset_dir)
    if args.split not in ds:
        raise ValueError(f"Split '{args.split}' not found in dataset.")
    split_ds = ds[args.split]

    # ――― model & processor ―――
    processor, model, device = load_model(args.model_path)

    # ――― transcription ―――
    pred_texts = transcribe_dataset(split_ds, processor, model, device, args.max_samples)
    orig_texts = [item["sentence"] for item in split_ds][: len(pred_texts)]

    # ――― write outputs ―――
    with open(args.orig_out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(orig_texts))
    with open(args.pred_out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(pred_texts))

    print(f"Wrote {len(pred_texts)} predictions to {args.pred_out_txt}\n Wrote references to {args.orig_out_txt}")


if __name__ == "__main__":
    main()