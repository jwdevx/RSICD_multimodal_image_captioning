Expect output in this folder:
    ├── best_model/
    │   │
    │   ├── config.json (Configuration file for the decoder)
    │   ├── preprocessor_config.json (Preprocessing config from BlipProcessor)
    │   ├── pytorch_model.bin (Model weights of the fine-tuned decoder (encoder is frozen))
    │   ├── special_tokens_map.json (Mapping of special tokens (e.g. pad, eos))
    │   ├── tokenizer_config.json (Tokenizer config from Hugging Face)
    │   │
    │   ├── merges.txt (Tokenizer vocab + merge rules (if using BPE/Byte-level models))
    │   └── vocab.json (Tokenizer vocab + merge rules (if using BPE/Byte-level models))
    │
    ├── training_log.txt (Console + file logs for each epoch: loss, warnings, generation samples, etc)
    ├── generated_captions_epoch*.json (1 .. n)
    ├── sample_logs_epoch1.txt (In-depth sample logging)

Pretraining the ability to
    - 📉 loss and accuracy curves
    - 📊 Plot accuracy curve
    - 🧠 Manual compare caption between ground truth and generated