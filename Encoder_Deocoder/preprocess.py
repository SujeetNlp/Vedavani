#!/usr/bin/env python3
"""preprocess.py

Run:
    python preprocess.py \
        --train_csv /path/to/train.csv \
        --test_csv  /path/to/test.csv  \
        --valid_csv /path/to/validation.csv \
        --audio_dir /path/to/Audio__files \
        --output_dir1 /path/to/raw_dataset \
        --output_dir2 /path/to/mapped_dataset
"""

import argparse
import os

import librosa
import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperTokenizer

# ――― default Whisper model (change if needed) ―――
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")


def load_and_process(csv_path: str, audio_dir: str, split_name: str, max_duration: float = 30.0) -> Dataset:
    """Return a HF Dataset with raw audio + sentence from a CSV split."""

    df = pd.read_csv(csv_path)  # col‑0: filename, col‑1: transcript, col‑2: length (ignored)
    data = {"audio": [], "sentence": []}

    pbar = tqdm(df.itertuples(index=False), total=len(df), unit="files", desc=split_name)
    for row in pbar:
        audio_path = os.path.join(audio_dir, row[0])
        sentence = row[1]

        try:
            waveform, sr = librosa.load(audio_path, sr=16000)
        except FileNotFoundError:
            pbar.write(f"missing: {audio_path}")
            continue

        if librosa.get_duration(y=waveform, sr=sr) > max_duration:
            continue

        data["audio"].append({"path": audio_path, "array": waveform, "sampling_rate": sr})
        data["sentence"].append(sentence)

    return Dataset.from_dict(data)


def prepare_dataset(batch):
    """Add Whisper‑style log‑Mel features + label ids."""

    audio = batch["audio"]
    batch["input_length"] = len(audio["array"])
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    batch["labels_length"] = len(tokenizer(batch["sentence"], add_special_tokens=False).input_ids)
    return batch


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess ASR CSVs and save raw + mapped DatasetDicts")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--valid_csv", required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--output_dir1", required=True, help="Directory for raw DatasetDict")
    ap.add_argument("--output_dir2", required=True, help="Directory for mapped DatasetDict")
    args = ap.parse_args()

    # ――― build raw DatasetDict ―――
    train_ds = load_and_process(args.train_csv, args.audio_dir, "train")
    test_ds  = load_and_process(args.test_csv,  args.audio_dir, "test")
    valid_ds = load_and_process(args.valid_csv, args.audio_dir, "valid")
    raw_ds_dict = DatasetDict({"train": train_ds, "test": test_ds, "valid": valid_ds})

    # save raw
    raw_ds_dict.save_to_disk(args.output_dir1)
    print(f"Raw dataset saved to {args.output_dir1}")

    # ――― map + save mapped DatasetDict ―――
    mapped_ds_dict = raw_ds_dict.map(
        prepare_dataset,
        remove_columns=raw_ds_dict["train"].column_names,
        num_proc=1,
    )
    mapped_ds_dict.save_to_disk(args.output_dir2)
    print(f"Mapped dataset saved to {args.output_dir2}")


if __name__ == "__main__":
    main()