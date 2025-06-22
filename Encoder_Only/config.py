class Config:
    num_train_epochs = 30
    train_batch_size = 4
    eval_batch_size = 2
    learning_rate = 1e-5
    warmup_steps = 300
    logging_steps = 1400
    save_steps = 1400
    eval_steps = 1400
    gradient_accumulation_steps = 2
    sampling_rate = 16000
    max_audio_length = 30  # seconds
    use_fp16 = True
    model_name = "facebook/mms-300m"
