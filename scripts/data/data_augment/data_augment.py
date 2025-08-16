#!/usr/bin/env python3

"""
RSICD Dataset Augmentation Script for Google Colab
Generates 2 augmented variants per training image with de-duplicated captions
(8,736 images x 5 captions = 43,680 caption-image pairs) * 3 = 131040 image caption pairs
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import random
import numpy as np

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ==============================================================================
# Configuration
# ==============================================================================

# # Original data paths
# original_data_args = {
#     'train_json': 'data/processed/captions_train2017.json',
#     'image_dir': 'data/RSICD_captions/images',
# }

# # New augmented data paths
# augmented_data_args = {
#     'train_json': 'data/augmentation_v1/captions_train_augmented.json',
#     'image_dir': 'data/augmentation_v1/images/',
# }

# Original data paths
original_data_args = {
    'train_json': 'captions_train2017.json',
    'image_dir': 'original',
}

# New augmented data paths
augmented_data_args = {
    'train_json': 'captions_train_augmented.json',
    'image_dir': 'images',
}
# ==============================================================================
# Define Augmentation Variants
# ==============================================================================

class FixedColorTransform:
    def __init__(self, brightness=1.0, contrast=1.0, saturation=1.0, hue=0.0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
# 20% brighter (brightness=1.2)
    def __call__(self, img):

        img = adjust_brightness(img, self.brightness)
        img = adjust_contrast(img, self.contrast)
        img = adjust_saturation(img, self.saturation)
        img = adjust_hue(img, self.hue)
        return img

class AdjustSharpnessTransform:
    def __init__(self, sharpness_factor=2.0):
        self.factor = sharpness_factor  # >1.0 sharpens, <1.0 blurs

    def __call__(self, img):
        return ImageEnhance.Sharpness(img).enhance(self.factor)

class RandomSharpness:
    """Applies random sharpening in a mild, safe range."""
    def __init__(self, min_factor=1.0, max_factor=2.0):
        self.min = min_factor
        self.max = max_factor

    def __call__(self, img):
        factor = random.uniform(self.min, self.max)
        return ImageEnhance.Sharpness(img).enhance(factor)

class AdjustLevelsTransform:
    """Photoshop-style levels adjustment (black/white point clipping)."""
    def __init__(self, black_level=40/255, white_level=210/255):
        self.black = black_level
        self.white = white_level

    def __call__(self, img):
        img_tensor = TF.to_tensor(img)
        img_tensor = (img_tensor - self.black) / (self.white - self.black)
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
        return TF.to_pil_image(img_tensor)


class Safe45Rotate:
    """Rotate 45° safely with expand=True and padded background fill."""
    def __call__(self, img):
        fill_color = tuple([int(c * 255) for c in TF.to_tensor(img).mean(dim=(1, 2))])
        return TF.rotate(img, angle=45, interpolation=TF.InterpolationMode.BILINEAR, expand=True, fill=fill_color)

class AxialRotateTransform:
    """Rotates 90 or 270 degrees randomly (true axial rotation)."""
    def __call__(self, img):
        angle = random.choice([90, 270])
        return TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)



#option 1
variant1_transform = transforms.Compose([
    AxialRotateTransform(),
    FixedColorTransform(brightness=1.2, contrast=1.0, saturation=1.5, hue=0.0),
    RandomSharpness(min_factor=1.0, max_factor=2.0),
    AdjustLevelsTransform(black_level=0/255, white_level=240/255),
    transforms.Resize((384, 384)),
    transforms.ToTensor()
])

variant2_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomResizedCrop(384, scale=(0.95, 1.0), ratio=(0.95, 1.05)),
    FixedColorTransform(brightness=1.0, contrast=1.2, saturation=1.1, hue=0.05),
    AdjustSharpnessTransform(sharpness_factor=3.5),
    AdjustLevelsTransform(black_level=10/255, white_level=210/255),
    transforms.Resize(384),
    transforms.ToTensor()
])


# Helper Functions


def create_directories():
    """Create necessary directories for augmented dataset"""
    augmented_dir = Path(augmented_data_args['image_dir'])
    augmented_dir.mkdir(parents=True, exist_ok=True)
    print(f" Created directory: {augmented_dir}")

    # Create parent directory for JSON
    json_dir = Path(augmented_data_args['train_json']).parent
    json_dir.mkdir(parents=True, exist_ok=True)
    print(f" Created directory: {json_dir}")



# Caption Redordering based on TFIDF


from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def deduplicate_and_pad_captions(captions, target_count=5):
    """
    De-duplicate captions, score them, place the most descriptive first,
    and pad to target count if needed.
    """

    # Remove duplicates while preserving order
    seen = set()
    unique_captions = []
    for caption in captions:
        if caption not in seen:
            seen.add(caption)
            unique_captions.append(caption)

    # Score unique captions using a basic TF-IDF vectorizer
    if len(unique_captions) > 1:
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(unique_captions)
        rarity_scores = tfidf_matrix.sum(axis=1).A1  # Sum of TF-IDF scores per sentence
        best_idx = int(np.argmax(rarity_scores))
        best_caption = unique_captions[best_idx]

        # Move best caption to the front
        unique_captions.pop(best_idx)
        unique_captions = [best_caption] + unique_captions

    # Pad if necessary (cyclic repeat)
    while len(unique_captions) < target_count:
        unique_captions.append(unique_captions[len(unique_captions) % len(unique_captions)])

    return unique_captions[:target_count]


# Create Original + 2 Variants


def process_and_save_image(image_path, output_path, transform):
    """Apply transformation and save image"""
    try:
        # Open image
        img = Image.open(image_path).convert('RGB')

        # Apply transformation
        img_transformed = transform(img)

        # Convert back to PIL if tensor
        if isinstance(img_transformed, torch.Tensor):
            # Convert tensor to PIL
            img_transformed = transforms.ToPILImage()(img_transformed)

        # Save image
        img_transformed.save(output_path, quality=95)
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def generate_augmented_dataset():
    """Main function to generate augmented dataset"""

    print("="*60)
    print("RSICD DATASET AUGMENTATION")
    print("="*60)

    # Create directories
    create_directories()

    # Load original training JSON
    print(f"\nLoading original training data from: {original_data_args['train_json']}")
    with open(original_data_args['train_json'], 'r') as f:
        original_data = json.load(f)

    print(f" Loaded {len(original_data['images'])} images")
    print(f" Loaded {len(original_data['annotations'])} annotations")

    # Initialize new augmented data structure
    augmented_data = {
        "images": [],
        "annotations": [],
        "info": {
            "description": "RSICD Augmented Training Set (Original + 2 Variants)",
            "version": "1.0",
            "year": 2025,
            "contributor": "RSICD Dataset - Augmented",
            "date_created": str(Path(original_data_args['train_json']).stat().st_mtime)
        },
        "licenses": original_data.get("licenses", [])
    }

    # Create mapping of image_id to captions
    image_id_to_captions = {}
    for ann in original_data['annotations']:
        if ann['image_id'] not in image_id_to_captions:
            image_id_to_captions[ann['image_id']] = []
        image_id_to_captions[ann['image_id']].append(ann['caption'])

    # Process counters
    total_images = len(original_data['images'])
    processed_images = 0
    new_annotation_id = 0

    print(f"\nProcessing {total_images} images (generating 3 versions each)...")
    print("This will create {total_images * 3} = {total_images * 3} total images")

    # Process each image
    for img_data in tqdm(original_data['images'], desc="Processing images"):
        image_id = img_data['id']
        original_filename = img_data['file_name']
        original_path = Path(original_data_args['image_dir']) / original_filename

        if not original_path.exists():
            print(f"Warning: Image not found: {original_path}")
            continue

        # Get captions for this image
        captions = image_id_to_captions.get(image_id, [])
        if not captions:
            print(f"Warning: No captions found for image {image_id}")
            continue

        # Process original captions
        original_captions = captions

        # Process augmented captions (de-duplicate and pad)
        augmented_captions = deduplicate_and_pad_captions(captions)

        # === 1. Copy Original Image ===
        original_output_path = Path(augmented_data_args['image_dir']) / original_filename
        try:
            # shutil.copy2(original_path, original_output_path)
            img = Image.open(original_path).convert("RGB")
            img_resized = transforms.Resize((384, 384))(img)
            img_resized.save(original_output_path, quality=95)

            # Add original image entry
            augmented_data['images'].append({
                "id": image_id,
                "file_name": original_filename,
                "width": 384,
                "height": 384,
                "license": img_data.get('license', 1),
                "date_captured": img_data.get('date_captured', "")
            })

            # Add original annotations (unchanged)
            for caption in original_captions:
                augmented_data['annotations'].append({
                    "id": new_annotation_id,
                    "image_id": image_id,
                    "caption": caption
                })
                new_annotation_id += 1

        except Exception as e:
            print(f"Error copying original image {original_filename}: {e}")
            continue

        #2. Generate Variant 1
        variant1_filename = original_filename.rsplit('.', 1)[0] + '_v1.' + original_filename.rsplit('.', 1)[1]
        variant1_output_path = Path(augmented_data_args['image_dir']) / variant1_filename
        variant1_image_id = int(str(image_id) + "001")

        if process_and_save_image(original_path, variant1_output_path, variant1_transform):
            # Add variant 1 image entry
            augmented_data['images'].append({
                "id": variant1_image_id,
                "file_name": variant1_filename,
                "width": 384,
                "height": 384,
                "license": img_data.get('license', 1),
                "date_captured": img_data.get('date_captured', "")
            })

            # Add variant 1 annotations (de-duplicated)
            for caption in augmented_captions:
                augmented_data['annotations'].append({
                    "id": new_annotation_id,
                    "image_id": variant1_image_id,
                    "caption": caption
                })
                new_annotation_id += 1

        # === 3. Generate Variant 2 ===
        variant2_filename = original_filename.rsplit('.', 1)[0] + '_v2.' + original_filename.rsplit('.', 1)[1]
        variant2_output_path = Path(augmented_data_args['image_dir']) / variant2_filename
        variant2_image_id = int(str(image_id) + "002")

        if process_and_save_image(original_path, variant2_output_path, variant2_transform):
            # Add variant 2 image entry
            augmented_data['images'].append({
                "id": variant2_image_id,
                "file_name": variant2_filename,
                "width": 384,
                "height": 384,
                "license": img_data.get('license', 1),
                "date_captured": img_data.get('date_captured', "")
            })

            # Add variant 2 annotations (de-duplicated)
            for caption in augmented_captions:
                augmented_data['annotations'].append({
                    "id": new_annotation_id,
                    "image_id": variant2_image_id,
                    "caption": caption
                })
                new_annotation_id += 1

        processed_images += 1

    # Save augmented JSON
    print(f"\nSaving augmented dataset JSON to: {augmented_data_args['train_json']}")
    with open(augmented_data_args['train_json'], 'w') as f:
        json.dump(augmented_data, f, indent=2)

    # ===== MINIMAL FIX: Copy validation and test data =====
    print("\nCopying validation and test data...")

    # Copy validation JSON
    val_json_src = original_data_args['train_json'].parent / 'captions_val2017.json'
    val_json_dst = Path(augmented_data_args['train_json']).parent / 'captions_val2017.json'
    if val_json_src.exists():
        shutil.copy2(val_json_src, val_json_dst)
        print(f" Copied validation JSON")

        # Copy validation images
        with open(val_json_src, 'r') as f:
            val_data = json.load(f)

        val_copied = 0
        for img_data in val_data['images']:
            src = Path(original_data_args['image_dir']) / img_data['file_name']
            dst = Path(augmented_data_args['image_dir']) / img_data['file_name']
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                val_copied += 1
        print(f" Copied {val_copied} validation images")

    # Copy test JSON
    test_json_src = original_data_args['train_json'].parent / 'captions_test2017.json'
    test_json_dst = Path(augmented_data_args['train_json']).parent / 'captions_test2017.json'
    if test_json_src.exists():
        shutil.copy2(test_json_src, test_json_dst)
        print(f" Copied test JSON")

        # Copy test images
        with open(test_json_src, 'r') as f:
            test_data = json.load(f)

        test_copied = 0
        for img_data in test_data['images']:
            src = Path(original_data_args['image_dir']) / img_data['file_name']
            dst = Path(augmented_data_args['image_dir']) / img_data['file_name']
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                test_copied += 1
        print(f" Copied {test_copied} test images")
    # ===== END OF FIX =====

    # Print summary
    print("\n" + "="*60)
    print("AUGMENTATION COMPLETE!")
    print("="*60)
    print(f" Total images created: {len(augmented_data['images'])}")
    print(f"  - Original images: {processed_images}")
    print(f"  - Variant 1 images: ~{processed_images}")
    print(f"  - Variant 2 images: ~{processed_images}")
    print(f" Total annotations: {len(augmented_data['annotations'])}")
    print(f" Expected image-caption pairs: {len(augmented_data['images']) * 5}")
    print(f" Images saved to: {augmented_data_args['image_dir']}")
    print(f" JSON saved to: {augmented_data_args['train_json']}")

    # Verify counts
    unique_images = len(set(img['file_name'] for img in augmented_data['images']))
    print(f"\n Unique image files: {unique_images}")

    # Show sample of de-duplicated captions
    print("\n Sample caption de-duplication:")
    sample_img_id = original_data['images'][0]['id'] if original_data['images'] else None
    if sample_img_id and sample_img_id in image_id_to_captions:
        original_caps = image_id_to_captions[sample_img_id]
        dedup_caps = deduplicate_and_pad_captions(original_caps)
        print(f"Original ({len(original_caps)}): {original_caps}")
        print(f"De-duplicated & padded (5): {dedup_caps}")

# Main Execution


if __name__ == "__main__":
    # Check if original data exists
    if not Path(original_data_args['train_json']).exists():
        print(f" Error: Original training JSON not found at {original_data_args['train_json']}")
        print("Please ensure your data is in the correct location.")
    elif not Path(original_data_args['image_dir']).exists():
        print(f" Error: Original image directory not found at {original_data_args['image_dir']}")
        print("Please ensure your images are in the correct location.")
    else:
        # Run augmentation
        generate_augmented_dataset()

        print("\n Augmentation complete! You can now use the augmented dataset for training.")
        print("\nTo use in your training script, update your paths to:")
        print(f"  train_json: '{augmented_data_args['train_json']}'")
        print(f"  image_dir: '{augmented_data_args['image_dir']}'")