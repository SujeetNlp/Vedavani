import os
import argparse
import librosa
from tqdm import tqdm
from datasets import Dataset, DatasetDict
import logging
from transformers import WhisperFeatureExtractor, WhisperTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_lines(audio_path, text_path):
    with open(audio_path, "r", encoding="utf-8") as f_audio, open(text_path, "r", encoding="utf-8") as f_text:
        return f_audio.readlines(), f_text.readlines()


def process_split(audio_lines, text_lines, split_name):
    data = {"audio": [], "sentence": []}
    progress_bar = tqdm(zip(audio_lines, text_lines), total=len(audio_lines), desc=f"Processing {split_name}")

    for audio_line, text_line in progress_bar:
        audio_path = audio_line.strip()
        try:
            waveform, sample_rate = librosa.load(audio_path, sr=16000)
            duration = librosa.get_duration(y=waveform, sr=sample_rate)
            if duration > 30:
                continue
        except FileNotFoundError:
            logger.warning(f"[{split_name}] File not found: {audio_path}")
            continue

        audio_feature = {
            "path": audio_path,
            "array": waveform,
            "sampling_rate": sample_rate
        }
        data["audio"].append(audio_feature)
        data["sentence"].append(text_line.strip())

    return Dataset.from_dict(data)


def prepare_dataset(batch, feature_extractor, tokenizer):
    audio = batch["audio"]
    batch["input_length"] = len(audio["array"])
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    batch["labels_length"] = len(batch["labels"])
    return batch


def main(args):
    # Load tokenizer and feature extractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Sanskrit", task="transcribe")

    # Read lines from file paths
    train_audio_lines, train_text_lines = read_lines(args.train_audio_path, args.train_text_path)
    test_audio_lines, test_text_lines = read_lines(args.test_audio_path, args.test_text_path)
    valid_audio_lines, valid_text_lines = read_lines(args.valid_audio_path, args.valid_text_path)

    # Process each split
    train_dataset = process_split(train_audio_lines, train_text_lines, "train")
    test_dataset = process_split(test_audio_lines, test_text_lines, "test")
    valid_dataset = process_split(valid_audio_lines, valid_text_lines, "valid")

    # Save raw dataset
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
        "valid": valid_dataset
    })
    dataset_dict.save_to_disk(args.raw_dataset_path)
    logger.info(f" Raw datasets saved to: {args.raw_dataset_path}")

    # Apply feature extraction and tokenization
    dataset_dict = dataset_dict.map(
        lambda batch: prepare_dataset(batch, feature_extractor, tokenizer),
        remove_columns=dataset_dict["train"].column_names,
        num_proc=1
    )
    dataset_dict.save_to_disk(args.processed_dataset_path)
    logger.info(f" Processed dataset saved to: {args.processed_dataset_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ASR dataset for Whisper model")

    parser.add_argument("--train_audio_path", type=str, required=True, help="Path to train audio list file")
    parser.add_argument("--train_text_path", type=str, required=True, help="Path to train text file")
    parser.add_argument("--test_audio_path", type=str, required=True, help="Path to test audio list file")
    parser.add_argument("--test_text_path", type=str, required=True, help="Path to test text file")
    parser.add_argument("--valid_audio_path", type=str, required=True, help="Path to validation audio list file")
    parser.add_argument("--valid_text_path", type=str, required=True, help="Path to validation text file")

    parser.add_argument("--raw_dataset_path", type=str, required=True, help="Where to save raw dataset")
    parser.add_argument("--processed_dataset_path", type=str, required=True, help="Where to save processed dataset")

    args = parser.parse_args()
    main(args)