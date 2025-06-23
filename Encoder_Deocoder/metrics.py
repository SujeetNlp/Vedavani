import evaluate
def compute_metrics(pred, tokenizer):
    metric_wer = evaluate.load("wer")
    metric_cer = evaluate.load("cer")

    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    return {
        "wer": 100 * metric_wer.compute(predictions=pred_str, references=label_str),
        "cer": 100 * metric_cer.compute(predictions=pred_str, references=label_str),
    }