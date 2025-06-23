import argparse
import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def transcribe_dataset(dataset_path, model_path, output_transcript_path, output_original_path, max_items=None):
    # Load dataset
    dataset_dict = load_from_disk(dataset_path)
    test_dataset = dataset_dict["test"]

    # Load model and processor
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transcriptions_list = []
    orig_list = []

    print("----------------------start---------------------------")
    print(dataset_dict)

    # Inference loop
    for index, item in tqdm(enumerate(test_dataset), total=len(test_dataset)):
        if max_items is not None and index >= max_items:
            break

        waveform = item["audio"]["array"]
        sent = item["sentence"]
        orig_list.append(sent)

        # Preprocess audio
        input_features = processor(waveform, sampling_rate=16000, return_tensors="pt").input_features.to(device)

        # Get forced decoder IDs
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="sanskrit", task="transcribe")
        forced_decoder_ids = torch.tensor(forced_decoder_ids, dtype=torch.long).to(device)

        # Generate transcription
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        transcriptions_list.append(transcription)
        print(transcription)

    print("----------------------saving---------------------------")

    # Save predicted transcriptions
    with open(output_transcript_path, "w", encoding="utf-8") as f:
        for item in transcriptions_list:
            f.write(f"{item}\n")

    # Save original sentences
    with open(output_original_path, "w", encoding="utf-8") as f:
        for item in orig_list:
            f.write(f"{item}\n")

    print("----------------------complete---------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe using Whisper model")

    parser.add_argument("--dataset_path", type=str, required=True, help="Path to saved dataset directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Whisper model directory")
    parser.add_argument("--output_transcript_path", type=str, required=True, help="Path to save transcriptions")
    parser.add_argument("--output_original_path", type=str, required=True, help="Path to save original sentences")
    parser.add_argument("--max_items", type=int, default=None, help="Max items to process (for testing)")

    args = parser.parse_args()

    transcribe_dataset(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        output_transcript_path=args.output_transcript_path,
        output_original_path=args.output_original_path,
        max_items=args.max_items,
    )