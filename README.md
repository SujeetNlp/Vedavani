# Vedavani: A Benchmark Corpus for ASR on Vedic Sanskrit Poetry (ACL‑2025)

[![arXiv](https://img.shields.io/badge/PDF-arXiv-blue)](https://arxiv.org/pdf/2506.00145v1)
[![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/SujeetNlp/Vedavani)

Code and dataset for the ACL 2025 paper ["Vedavani: A Benchmark Corpus for ASR on Vedic Sanskrit Poetry"](https://arxiv.org/pdf/2506.00145v1), by
Sujeet Kumar, Pretam Ray, Abhinay Beerukuri, Shrey Kamoji, Manoj Balaji Jagadeeshan, and Pawan Goyal.

Vedavani introduces a 54‑hour annotated speech corpus from the Rig Veda (≈20 k verses) and Atharva Veda (≈10 k verses); this is the first ASR study targeting Vedic Sanskrit poetry, capturing its complex prosody and phonetics. We benchmark multiple SOTA models—including multilingual variants, Whisper, and IndicWhisper—showing strong performance of IndicWhisper 

## Dataset Access

The complete **Vedavani** dataset—including audio files and annotated transcripts—is hosted on Google Drive. You can download it using the following link:

👉 [Download the Vedavani dataset (Google Drive)](https://drive.google.com/drive/folders/1bDE8Vlm9Be-Lf2Tab12SWQ5vrwfvTf0S?usp=sharing)

Please note:
- The dataset is organized into train, validation, and test splits.
- Transcripts are provided in Devanagari script and contain prosodic markers aligned with audio.
- The audio covers ~54 hours across 30,779 Vedic verse samples.


## Installation
Install dependencies listed in requirements.txt:
```bash
pip install -r requirements.txt
```

## Usage

### Encoder-Only Models
Move to "Encoder_Only" directory

#### Train Encoder-Only Model 

Make sure you're logged into Hugging Face before training:

```bash
huggingface-cli login
    # Training the model using the provided dataset and pretrained checkpoint
python train.py \
  --train_csv /path/to/train.csv \
  --val_csv /path/to/validation.csv \
  --audio_path_prefix /path/to/Audio_files/ \
  --vocab_path /path/to/vocab.json \
  --output_dir /path/to/output_dir \
  --pretrained_model facebook/wav2vec2-base \
  --train_batch_size 8 \
  --eval_batch_size 8
```

#### Evaluate the Encoder-Only Model
```bash
python evaluate.py \
  --csv_path /path/to/test.csv \
  --audio_path_prefix /path/to/Audio_files/ \
  --vocab_path /path/to/vocab.json \
  --model_dir /path/to/checkpoints/epoch_20 \
  --save_path /path/to/inference/resuts.csv \
  --batch_size 32

```

