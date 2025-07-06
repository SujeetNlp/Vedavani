"""results.py – Evaluate CER and WER 

Run:
    python results.py \
        --original /path/to/test_orig.txt \
        --predicted /path/to/test_predicted.txt
"""

import argparse
import evaluate

def main():
    parser = argparse.ArgumentParser(description="Compute WER and CER from predictions.")
    parser.add_argument("--original", required=True, help="Path to original reference text file")
    parser.add_argument("--predicted", required=True, help="Path to model-predicted text file")
    args = parser.parse_args()

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    with open(args.original, 'r') as f:
        references = f.read().strip().splitlines()

    with open(args.predicted, 'r') as f:
        predictions = f.read().strip().splitlines()

    if len(references) != len(predictions):
        print(f"Mismatched line count: {len(references)} refs vs {len(predictions)} preds")
        return

    wer = 100 * wer_metric.compute(predictions=predictions, references=references)
    cer = 100 * cer_metric.compute(predictions=predictions, references=references)

    print(f" WER: {wer:.2f}%")
    print(f" CER: {cer:.2f}%")

if __name__ == "__main__":
    main()