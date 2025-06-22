# ------------------ Imports ------------------
import os
import torch
import pandas as pd
import argparse
import logging
from tqdm import tqdm
from datasets import Dataset, load_metric
from torch.utils.data import DataLoader
from transformers import (
    Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor,
    Wav2Vec2ForCTC, AdamW, get_linear_schedule_with_warmup
)
from torch.cuda.amp import GradScaler, autocast

from config import Config
from utils import speech_file_to_array_fn, prepare_dataset, DataCollatorCTCWithPadding
from metrics import compute_metrics

# ------------------ Logger ------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ------------------ Main Function ------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    df_train = pd.read_csv(args.train_csv)
    df_val = pd.read_csv(args.val_csv)

    df_train['path'] = args.audio_path_prefix + df_train['audio_file']
    df_val['path'] = args.audio_path_prefix + df_val['audio_file']

    df_train = df_train[df_train['length'] < Config.max_audio_length]
    df_val = df_val[df_val['length'] < Config.max_audio_length]

    train_data = Dataset.from_pandas(df_train)
    val_data = Dataset.from_pandas(df_val)

    tokenizer = Wav2Vec2CTCTokenizer(args.vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=Config.sampling_rate,
        padding_value=0.0, do_normalize=True, return_attention_mask=True
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(os.path.join(args.output_dir, "wav2vec2_processor"))

    train_data = train_data.map(lambda x: speech_file_to_array_fn(x), remove_columns=train_data.column_names)
    val_data = val_data.map(lambda x: speech_file_to_array_fn(x), remove_columns=val_data.column_names)

    train_data = train_data.map(lambda x: prepare_dataset(x, processor), remove_columns=train_data.column_names, batched=True, batch_size=args.train_batch_size)
    val_data = val_data.map(lambda x: prepare_dataset(x, processor), remove_columns=val_data.column_names, batched=True, batch_size=args.eval_batch_size)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    wer_metric = load_metric("wer", trust_remote_code=True)
    cer_metric = load_metric("cer", trust_remote_code=True)

    model = Wav2Vec2ForCTC.from_pretrained(
        args.pretrained_model,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True
    )

    model.gradient_checkpointing_enable()
    model.freeze_feature_extractor()
    model.to(device)

    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, collate_fn=data_collator)
    val_loader = DataLoader(val_data, batch_size=args.eval_batch_size, collate_fn=data_collator)

    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=Config.warmup_steps,
                                                num_training_steps=Config.num_train_epochs * len(train_loader))
    scaler = GradScaler()

    log_file = os.path.join(args.output_dir, "epoch_info.txt")
    with open(log_file, "w") as file:
        for epoch in range(Config.num_train_epochs):
            model.train()
            for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
                inputs = {k: v.to(device) for k, v in batch.items()}
                with autocast(enabled=Config.use_fp16):
                    outputs = model(**inputs)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                if (step + 1) % Config.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                if (step + 1) % Config.logging_steps == 0:
                    logger.info(f"[Epoch {epoch}, Step {step}] Loss: {loss.item():.4f}")

            # Evaluation
            model.eval()
            eval_loss = wer = cer = 0.0
            for batch in val_loader:
                with torch.no_grad():
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**inputs)
                    wer_t, cer_t = compute_metrics(outputs, inputs["labels"], processor, wer_metric, cer_metric)
                    wer += wer_t
                    cer += cer_t
                    eval_loss += outputs.loss.item()

            wer /= len(val_loader)
            cer /= len(val_loader)
            eval_loss /= len(val_loader)

            logger.info(f"[Eval @ Epoch {epoch}] Loss = {eval_loss:.4f}, WER = {wer:.4f}, CER = {cer:.4f}")
            file.write(f"Epoch {epoch}, Eval Loss = {eval_loss}, WER = {wer}, CER = {cer}\n")

            save_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)


# ------------------ Entry Point ------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Wav2Vec2 ASR Model")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--audio_path_prefix", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, default=Config.model_name)
    parser.add_argument("--train_batch_size", type=int, default=Config.train_batch_size)
    parser.add_argument("--eval_batch_size", type=int, default=Config.eval_batch_size)

    args = parser.parse_args()
    main(args)
