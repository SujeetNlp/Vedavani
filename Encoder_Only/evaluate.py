# ------------------ Imports ------------------
import os
import torch
import librosa
import pandas as pd
import argparse
import logging
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Union, Optional
from datasets import Dataset, load_metric
from torch.utils.data import DataLoader
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM,
    Wav2Vec2ForCTC
)
from pyctcdecode import build_ctcdecoder

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------ Functions ------------------
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = librosa.load(batch["path"], sr=16000)
    batch["speech"] = speech_array
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["text"]
    return batch

def prepare_dataset(batch, processor):
    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

@dataclass
class DataCollatorCTCWithPadding:
    processor: Union[Wav2Vec2Processor, Wav2Vec2ProcessorWithLM]
    padding: Union[bool, str] = True
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding,
                                   pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding,
                                              max_length=self.max_length_labels,
                                              pad_to_multiple_of=self.pad_to_multiple_of_labels,
                                              return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

def compute_metrics(outputs, labels, processor, wer_metric, cer_metric, results_df, use_lm):
    logits = outputs.logits
    if use_lm:
        pred_str = processor.batch_decode(logits.cpu().numpy()).text
    else:
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = processor.batch_decode(pred_ids)
    labels[labels == -100] = processor.tokenizer.pad_token_id
    label_str = processor.tokenizer.batch_decode(labels, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    batch_results_df = pd.DataFrame({"Original String": label_str, "Predicted String": pred_str, "WER": wer, "CER": cer})
    results_df = pd.concat([results_df, batch_results_df], ignore_index=True)
    return wer, cer, results_df

# ------------------ Main Function ------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    df_test = pd.read_csv(args.csv_path)
    df_test = df_test[df_test['length'] < 30]
    df_test['path'] = args.audio_path_prefix + df_test['audio_file']
    test_data = Dataset.from_pandas(df_test)

    tokenizer = Wav2Vec2CTCTokenizer(args.vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True
    )

    if args.arpa_path:
        vocab_dict = tokenizer.get_vocab()
        sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
        decoder = build_ctcdecoder(labels=list(sorted_vocab_dict.keys()), kenlm_model_path=args.arpa_path)
        processor = Wav2Vec2ProcessorWithLM(feature_extractor=feature_extractor, tokenizer=tokenizer, decoder=decoder)
        use_lm = True
    else:
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        use_lm = False

    test_data = test_data.map(speech_file_to_array_fn, remove_columns=test_data.column_names)
    test_dataset = test_data.map(lambda x: prepare_dataset(x, processor), remove_columns=test_data.column_names, batch_size=32, batched=True)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collator)

    wer_metric = load_metric("wer")
    cer_metric = load_metric("cer")
    results_df = pd.DataFrame(columns=["Original String", "Predicted String", "WER", "CER"])

    logger.info("Loading model...")
    model = Wav2Vec2ForCTC.from_pretrained(args.model_dir).to(device)

    model.eval()
    eval_loss, wer, cer = 0.0, 0.0, 0.0
    logger.info("Starting evaluation...")

    for eval_batch in tqdm(test_loader, desc="Evaluating"):
        with torch.no_grad():
            eval_inputs = {k: v.to(device) for k, v in eval_batch.items()}
            eval_outputs = model(**eval_inputs)
            wer_t, cer_t, results_df = compute_metrics(eval_outputs, eval_inputs["labels"], processor, wer_metric, cer_metric, results_df, use_lm)
            wer += wer_t
            cer += cer_t
            eval_loss += eval_outputs.loss.item()

    eval_loss /= len(test_loader)
    wer /= len(test_loader)
    cer /= len(test_loader)

    logger.info(f"Final Evaluation -> Loss: {eval_loss:.4f}, WER: {wer:.4f}, CER: {cer:.4f}")

    summary_row = pd.DataFrame({"Original String": ["Average"], "Predicted String": [""], "WER": [wer], "CER": [cer]})
    results_df = pd.concat([results_df, summary_row], ignore_index=True)
    results_df.to_csv(args.save_path, index=False)
    logger.info(f"Saved results to {args.save_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Wav2Vec2 Inference with optional LM")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--audio_path_prefix", type=str, required=True, help="Prefix path to audio files")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocab.json")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to load model from")
    parser.add_argument("--arpa_path", type=str, default=None, help="Optional KenLM .arpa file path")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save inference CSV")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    args = parser.parse_args()

    main(args)
