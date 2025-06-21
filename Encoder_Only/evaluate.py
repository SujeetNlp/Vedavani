# ------------------ Imports ------------------
import os
import torch
import pandas as pd
import argparse
import logging
from tqdm import tqdm
from datasets import Dataset, load_metric
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM,
    Wav2Vec2ForCTC
)
from pyctcdecode import build_ctcdecoder

from utils import speech_file_to_array_fn, prepare_dataset, DataCollatorCTCWithPadding
from metrics import compute_metrics
from config import Config

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------ Main Function ------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    df_test = pd.read_csv(args.csv_path)
    df_test = df_test[df_test['length'] < Config.max_audio_length]
    df_test['path'] = args.audio_path_prefix + df_test['audio_file']
    test_data = Dataset.from_pandas(df_test)

    tokenizer = Wav2Vec2CTCTokenizer(args.vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=Config.sampling_rate, padding_value=0.0, do_normalize=True, return_attention_mask=True
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
    test_dataset = test_data.map(lambda x: prepare_dataset(x, processor), remove_columns=test_data.column_names, batch_size=args.batch_size, batched=True)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collator)

    wer_metric = load_metric("wer", trust_remote_code=True)
    cer_metric = load_metric("cer", trust_remote_code=True)
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
