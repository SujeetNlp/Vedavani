import torch
import librosa
from dataclasses import dataclass
from typing import List, Dict, Union, Optional
from config import Config

# Convert audio file path to waveform array
def speech_file_to_array_fn(batch):
    speech_array, _ = librosa.load(batch["path"], sr=Config.sampling_rate)
    batch["speech"] = speech_array
    batch["sampling_rate"] = Config.sampling_rate
    batch["target_text"] = batch["text"]
    return batch

def prepare_dataset(batch, processor):
    batch["input_values"] = processor(batch["speech"], sampling_rate=Config.sampling_rate).input_values
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

@dataclass
class DataCollatorCTCWithPadding:
    processor: any
    padding: Union[bool, str] = True
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features, padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt"
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features, padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt"
            )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch
