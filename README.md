# RSICD multimodal image captioning
Remote Sensing Image Captioning Project for UNSW Comp9444


# Original Repo
Lu et al.(2017) https://github.com/201528014227051/RSICD_optimal

# Tree

```txt
.
├── LICENSE
├── README.md
├── data
│   ├── processed
│   │   ├── README.md
│   │   ├── captions_test2017.json
│   │   ├── captions_train2017.json
│   │   └── captions_val2017.json
│   └── raw
│       └── README.md
├── outputs
│   └── mlat
│       └── 00493_transformer.png
├── requirements.txt
└── scripts
    ├── data
    │   └── data_augment
    │       ├── captions_train2017.json
    │       ├── captions_train_augmented.json
    │       ├── data_augment.py
    │       ├── data_augment_score.py
    │       ├── images
    │       └── original
    └── models
        └── blip1
            ├── 1_train_5e7_v1.py
            ├── 1_train_5e7_v2.py
            ├── 2_eval.py
            ├── 3_caption.py
            └── blip1.ipynb
```