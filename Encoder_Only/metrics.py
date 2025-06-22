import torch
import pandas as pd

def compute_metrics(outputs, labels, processor, wer_metric, cer_metric, results_df=None, use_lm=False):
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

    if results_df is not None:
        batch_results_df = pd.DataFrame({
            "Original String": label_str,
            "Predicted String": pred_str,
            "WER": wer,
            "CER": cer
        })
        results_df = pd.concat([results_df, batch_results_df], ignore_index=True)
        return wer, cer, results_df

    return wer, cer
