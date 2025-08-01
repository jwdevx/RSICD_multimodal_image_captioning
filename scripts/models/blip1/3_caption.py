# Caption Comparisons Using Unseen Datasets
# ==============================================================================
# 📦 Install Required Packages
# ==============================================================================
# !pip install -q pycocotools git+https://github.com/salaniz/pycocoevalcap

# ==============================================================================
# 📊 Evaluate Caption on a Single RSICD Image
# ==============================================================================
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from IPython.display import display, HTML
from PIL import Image
import matplotlib.pyplot as plt
import torch
import re
import json
import os
import contextlib
import io
from difflib import SequenceMatcher
from pathlib import Path

# === Global path to prediction JSON (adjust if needed) ===
OUTPUT_DIR = Path("outputs/blip1_rsicd_github_approach_full_model")
OUT_COCO = OUTPUT_DIR / "test_generated_captions_formatted_beam5.json"


def show_specific_example_with_metric(image_filename, dataset, model, processor,
                          gt_json="data/processed/captions_test2017.json",
                          pred_json="outputs/blip1_rsicd_final/generated_captions_epoch5_formatted.json",
                          model_name="BLIP-1"):
    model.eval()
    device = next(model.parameters()).device

    # Find dataset index
    index = None
    for i, img_id in enumerate(dataset.image_ids):
        if dataset.id_to_filename[img_id] == image_filename:
            index = i
            break
    if index is None:
        print(f"Image {image_filename} not found in dataset.")
        return

    # Load input
    pixel_values, _, reference_captions = dataset[index]
    image_path = os.path.join(dataset.image_dir, image_filename)
    original_image = Image.open(image_path).convert("RGB")
    pixel_values_batch = pixel_values.unsqueeze(0).to(device)

    with torch.no_grad():
        output_ids = model.generate(pixel_values_batch, max_length=30, num_beams=1)
        generated_caption = processor.decode(output_ids[0], skip_special_tokens=True)

    # Tokenize
    gen_tokens = re.findall(r"\b\w+\b", generated_caption.lower())
    ref_tokens_all = []
    for caption in reference_captions:
        ref_tokens_all += re.findall(r"\b\w+\b", caption.lower())
    ref_word_set = set(ref_tokens_all)

    # BLEU-n highlight
    def get_ngrams(tokens, n):
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

    def bleu_n_highlight(gen_tokens, ref_tokens, n, color):
        gen_ngrams = get_ngrams(gen_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)
        matches = gen_ngrams & ref_ngrams
        result = []
        i = 0
        while i < len(gen_tokens):
            ngram = tuple(gen_tokens[i:i+n])
            if len(ngram) == n and ngram in matches:
                result.append(f"<span style='color:{color}'>{' '.join(ngram)}</span>")
                i += n
            else:
                result.append(gen_tokens[i])
                i += 1
        return ' '.join(result)

    bleu1_html = ' '.join([f"<span style='color:blue'>{w}</span>" if w in ref_word_set else w for w in gen_tokens])
    bleu2_html = bleu_n_highlight(gen_tokens, ref_tokens_all, 2, "blue")
    bleu3_html = bleu_n_highlight(gen_tokens, ref_tokens_all, 3, "blue")
    bleu4_html = bleu_n_highlight(gen_tokens, ref_tokens_all, 4, "blue")

    # ROUGE-L
    def lcs(a, b):
        sm = SequenceMatcher(None, a, b)
        return [a[i] for i, j, n in sm.get_matching_blocks() if n > 0 for i in range(i, i + n)]

    best_lcs = []
    best_caption_idx = -1
    for idx, caption in enumerate(reference_captions):
        ref = re.findall(r"\b\w+\b", caption.lower())
        lcs_seq = lcs(gen_tokens, ref)
        if len(lcs_seq) > len(best_lcs):
            best_lcs = lcs_seq
            best_caption_idx = idx

    best_lcs_set = set(best_lcs)
    rouge_gen_html = ' '.join([
        f"<span style='color:green'><b>{w}</b></span>" if w in best_lcs_set else w
        for w in gen_tokens
    ])

    # ===============================
    # Display Section
    # ===============================
    plt.figure(figsize=(5, 5))
    plt.imshow(original_image)
    plt.title("Generated Caption", fontsize=10)
    plt.axis('off')
    plt.show()

    print(f"\n🖼️ Image: {image_filename}\n")
    print("📄 Ground Truth Captions:\n")
    gt_lines = []
    for j, ref in enumerate(reference_captions):
        if j == best_caption_idx:
            ref_words = ref.split()
            highlighted = ' '.join(
                f"<span style='color:green'>{w}</span>" if w.lower() in best_lcs_set else w
                for w in ref_words
            )
            line = f"<b>{j+1}.</b> {highlighted}"
        else:
            line = f"<b>{j+1}.</b> {ref}"
        gt_lines.append(f"<div style='margin-left: 20px'>{line}</div>")
    display(HTML("".join(gt_lines)))

    print("\n🔮 Generated Caption (highlighted):")
    for label, html_caption in [
        ("BLEU1", bleu1_html),
        ("BLEU2", bleu2_html),
        ("BLEU3", bleu3_html),
        ("BLEU4", bleu4_html),
        ("ROUGE-L", rouge_gen_html),
    ]:
        display(HTML(f"<div style='margin-left: 20px'><b>{label}:</b> {html_caption}</div>"))

    # ===============================
    # Evaluation Metrics Section
    # ===============================
    with open(gt_json, 'r') as f: gt_data = json.load(f)
    with open(pred_json, 'r') as f: pred_data = json.load(f)

    filename_to_id = {img["file_name"]: img["id"] for img in gt_data["images"]}
    if image_filename not in filename_to_id:
        print(f"\n⚠️ Warning: Image not found in COCO GT file.\n")
        return
    img_id = filename_to_id[image_filename]

    single_gt = {
        "info": {},  # Required dummy field
        "licenses": [],  # Required dummy field
        "type": "captions",  # Required by COCO API
        "images": [{"id": img_id, "file_name": image_filename}],
        "annotations": [ann for ann in gt_data["annotations"] if ann["image_id"] == img_id]
    }

    single_pred = [entry for entry in pred_data if entry["image_id"] == img_id]

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = single_gt
        coco.createIndex()
        coco_res = coco.loadRes(single_pred)
        coco_eval = COCOEvalCap(coco, coco_res)
        coco_eval.evaluate()

    m = coco_eval.eval
    print("\n📊 Evaluation Metrics:")
    print("| Model   | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE_L | SPICE |")
    print("|---------|--------|--------|--------|--------|--------|---------|-------|")
    print(f"| {model_name:<7} |  {m['Bleu_1']:.2f}  |  {m['Bleu_2']:.2f}  |  {m['Bleu_3']:.2f}  |  {m['Bleu_4']:.2f}  |"
          f"  {m['METEOR']:.2f}  |  {m['ROUGE_L']:.2f}   |  {m['SPICE']:.2f} |")
    # print("-" * 80)

# 5 images to compare with the different models
show_specific_example_with_metric("00493.jpg", test_dataset, model, processor)
show_specific_example_with_metric("00834.jpg", test_dataset, model, processor)
# show_specific_example_with_metric("00695.jpg", test_dataset, model, processor)
# show_specific_example_with_metric("00407.jpg", test_dataset, model, processor)
show_specific_example_with_metric("00867.jpg", test_dataset, model, processor)
show_specific_example_with_metric("00846.jpg", test_dataset, model, processor)
# show_specific_example_with_metric("00373.jpg", test_dataset, model, processor)
show_specific_example_with_metric("00424.jpg", test_dataset, model, processor)
# show_specific_example_with_metric("00348.jpg", test_dataset, model, processor)
show_specific_example_with_metric("viaduct_328.jpg", test_dataset, model, processor)
# show_specific_example_with_metric("viaduct_329.jpg", test_dataset, model, processor)
# show_specific_example_with_metric("viaduct_351.jpg", test_dataset, model, processor)