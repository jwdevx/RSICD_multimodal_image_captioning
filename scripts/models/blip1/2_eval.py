#!/usr/bin/env python3

# !pip install -q pycocotools git+https://github.com/salaniz/pycocoevalcap

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch, json, os
from pathlib import Path
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

# === Paths ===
OUTPUT_DIR = Path("outputs/blip1_rsicd_github_approach_full_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = OUTPUT_DIR / "best_model"
TEST_JSON = "data/processed/captions_test2017.json"
IMAGE_DIR = "data/RSICD_captions/images"

# NEW FILENAMES FOR BEAM SEARCH
OUT_COCO = OUTPUT_DIR / "test_generated_captions_formatted_beam5.json"
OUT_DEBUG = OUTPUT_DIR / "test_generated_captions_debug_beam5.json"

# === Load model & processor ===
processor = BlipProcessor.from_pretrained(MODEL_DIR)
model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR).cuda().eval()

# === Load test metadata ===
with open(TEST_JSON) as f:
    test_data = json.load(f)

id_to_file = {img["id"]: img["file_name"] for img in test_data["images"]}

annotations = {}
for ann in test_data["annotations"]:
    annotations.setdefault(ann["image_id"], []).append(ann["caption"])

debug_results = []
coco_results = []

# === Generate captions with beam search ===
for image_id, file_name in tqdm(id_to_file.items(), desc="Generating captions with beam search"):
    img = Image.open(os.path.join(IMAGE_DIR, file_name)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        caption = processor.decode(out[0], skip_special_tokens=True)

    # Append to both outputs
    debug_results.append({
        "image_id": image_id,
        "file_name": file_name,
        "generated": caption,
        "ground_truth": annotations.get(image_id, [])
    })

    coco_results.append({
        "image_id": image_id,
        "caption": caption
    })

# === Save outputs ===
with open(OUT_DEBUG, "w") as f:
    json.dump(debug_results, f, indent=2)

with open(OUT_COCO, "w") as f:
    json.dump(coco_results, f, indent=2)

print("✅ Done! Saved:")
print(f"  - COCO format: {OUT_COCO}")
print(f"  - Full debug : {OUT_DEBUG}")

# === Evaluate ===
coco = COCO(TEST_JSON)
coco_res = coco.loadRes(str(OUT_COCO))

coco_eval = COCOEvalCap(coco, coco_res)
coco_eval.evaluate()

# === Display nicely formatted metrics ===
metrics = coco_eval.eval
print("\n📊 Evaluation Metrics:")
print("|       Model    | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE_L | CIDEr | SPICE |")
print("|----------------|--------|--------|--------|--------|--------|---------|-------|-------|")
print(f"| BLIP-1 Beam-5 | {metrics['Bleu_1']:.4f} | {metrics['Bleu_2']:.4f} | {metrics['Bleu_3']:.4f} | {metrics['Bleu_4']:.4f} | {metrics['METEOR']:.4f} | {metrics['ROUGE_L']:.4f} | {metrics['CIDEr']:.4f} | {metrics['SPICE']:.4f} |")
