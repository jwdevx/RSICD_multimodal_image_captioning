# RSICD multimodal image captioning
Remote Sensing Image Captioning Project for UNSW Comp9444


# Original Repo
Lu et al. (2017) https://github.com/201528014227051/RSICD_optimal


# Final Best Metric on RSICD

| Model       | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE_L | CIDEr  | SPICE  |
|-------------|--------|--------|--------|--------|--------|---------|--------|--------|
| VLAD + RNN  | 0.493  | 0.3091 | 0.2209 | 0.1677 | 0.1996 |  0.4242 | 1.0392 |   -    |
| VIT + GPT2  | 0.5832 | 0.3456 | 0.2118 | 0.1371 | 0.3413 |  0.3306 | 0.3846 | 0.2124 |
| BLIP1-Base  | 0.6809 | 0.5071 | 0.3865 | 0.3027 | 0.2579 |  0.4794 | 0.5864 | 0.2671 |
| BLIP1-Large | 0.7387 | 0.5773 | 0.4584 | 0.3719 | 0.2999 |  0.5397 | 0.8822 | 0.2917 |
| SkyEye GPT  | 0.8773 | 0.777  | 0.689  | 0.6199 | 0.3623 |  0.6354 | 0.8937 |   -    |

- VLAD + RNN implemented by Lu et al.(2017)
- VIT + GPT2, own implementation
- BLIP1-Base, own implementation
- BLIP1-Large, own implementation
- SkyEye GPT, current best benchmark (2025)

# Future Works
- see notebook/ powerpoint for full writeup on why BITA approach could be better