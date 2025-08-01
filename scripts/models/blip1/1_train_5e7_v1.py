#!/usr/bin/env python3

# ==============================================================================
# Install required packages
# ==============================================================================

# !pip install -q transformers torch torchvision pillow nltk pycocoevalcap tqdm matplotlib

"""
BLIP-1 Fine-tuning for RSICD Dataset - GitHub Repository Style
==============================================================
Following the approach from: https://github.com/gurveervirk/Image-Captioning-for-Remote-Sensing-Data

Enabled
- Uses ALL 5 captions per image (not just 1 random)
- mixed precision FP16
- Much lower learning rate: 5e-7
- Trains full model (encoder + decoder)


📊 Evaluation Metrics:
| Model         | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE_L | CIDEr  | SPICE  |
|---------------|--------|--------|--------|--------|--------|---------|--------|--------|
| BLIP-1 Beam-5 | 0.6900 | 0.5246 | 0.4029 | 0.3166 | 0.2734 | 0.4888  | 0.7271 | 0.2641 |
"""

# ==============================================================================
# 📦 Standard Library Imports
# ==============================================================================
import os
import gc
import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import math
# ==============================================================================
# 📦 Third-Party Imports
# ==============================================================================
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import warnings
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ==============================================================================
# ⚙️ Global Setup: Seeds, Warnings, Memory
# ==============================================================================
# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    # Add deterministic mode for debugging
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Warning suppression and memory optimization
warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
gc.collect()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# ==============================================================================
#  Minimal Training Setup
# ==============================================================================
OUTPUT_DIR = Path("outputs/blip1/blip1_rsicd_github_approach_full_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Stripped down configuration
training_args = {
    'model_name': 'Salesforce/blip-image-captioning-base',
    'train_json': '/content/data_local/processed/captions_train2017.json',
    'val_json': '/content/data_local/processed/captions_val2017.json',
    'image_dir': '/content/data_local/RSICD_captions/images',

    'batch_size': 6,            # Increased from 4!
    'learning_rate': 5e-7,
    'weight_decay': 0.01,       # Added weight decay

    'label_smoothing': 0.0,     # No smoothing initially

    'num_epochs': 5,            # Just 5 epochs is more than enough
    'early_stopping_patience': 3,
    'resume': True,             # Set it to false if you want to start fresh
}


# ==============================================================================
# 📁 Output and Logging Setup
# ==============================================================================
# Writes to console and training_log.txt
log_file = OUTPUT_DIR / "training_log.txt"

# Clear any previous logging handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
def log_and_print(message):
    """Log to file and print to notebook"""
    logging.info(message)

# ==============================================================================
# 🔧 Sanity check: Test Pretrained Model First
# ==============================================================================

def test_pretrained_model(device):
    """Test if pretrained BLIP works before training"""
    log_and_print("\n🔍 Testing pretrained BLIP model...")

    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
    model = model.to(device)
    model.eval()

    # Test on a dummy image
    dummy_image = torch.randn(1, 3, 384, 384).to(device)

    # Test generation
    with torch.no_grad():
        generated = model.generate(dummy_image, max_length=20)
        caption = processor.decode(generated[0], skip_special_tokens=True)
        log_and_print(f"Test caption (random image): {caption}")

        # Check if output is degenerate
        if len(caption) < 3 or all(c == caption[0] for c in caption):
            log_and_print("⚠️ WARNING: Model generating garbage on random input!")

    # Test with a real image if possible
    try:
        import requests
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = processor(image, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(**inputs, max_length=20)
            real_caption = processor.decode(out[0], skip_special_tokens=True)
            log_and_print(f"Test caption (real image): {real_caption}")

    except:
        log_and_print("Could not test with real image (no internet?)")

    # Verify vocabulary alignment
    model_vocab_size = model.config.text_config.vocab_size
    processor_vocab_size = len(processor.tokenizer)
    log_and_print(f"Model vocab size: {model_vocab_size}")
    log_and_print(f"Processor vocab size: {processor_vocab_size}")

    # Check special tokens
    log_and_print(f"Pad token ID: {processor.tokenizer.pad_token_id}")
    log_and_print(f"BOS token ID: {processor.tokenizer.bos_token_id}")
    log_and_print(f"EOS token ID: {processor.tokenizer.eos_token_id}")

    log_and_print("✅ Pretrained model test complete\n")
    return True


# ==============================================================================
# Dataset - CHANGED to return ALL captions
# ==============================================================================

class RSICDDatasetMinimal(Dataset):
    """CHANGED: Flattens all caption-image pairs for efficient batching"""

    def __init__(self, json_path, image_dir, processor, split='train'):
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.split = split

        # Load annotations
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Build annotation mapping
        annotations = defaultdict(list)
        for ann in data['annotations']:
            annotations[ann['image_id']].append(ann['caption'])

        # CHANGED: Flatten all image-caption pairs
        self.samples = []
        for img in data['images']:
            if img['id'] in annotations:
                img_path = self.image_dir / img['file_name']
                captions = annotations[img['id']]
                # Create one sample per caption
                for caption in captions:
                    self.samples.append({
                        'image_path': img_path,
                        'caption': caption,
                        'image_id': img['id'],
                        'file_name': img['file_name'],
                        'all_captions': captions
                    })

        log_and_print(f"Loaded {len(self.samples)} caption-image pairs from {len(data['images'])} images for {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['image_path']

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            log_and_print(f"Failed to load {img_path}: {e}")
            return None

        # Process image and caption together
        encoding = self.processor(
            image,
            sample['caption'].lower(),
            padding="max_length",
            truncation=True,
            max_length=40,
            return_tensors="pt"
        )

        # Remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}

        # Add metadata
        encoding['image_id'] = sample['image_id']
        encoding['file_name'] = sample['file_name']
        encoding['all_captions'] = sample['all_captions']
        return encoding

# ==============================================================================
# 🔧  Training with Gradient Monitoring - FIXED without FP16
# ==============================================================================
from IPython.display import display
import matplotlib.pyplot as plt

def train_epoch_with_monitoring(model, dataloader, optimizer, scheduler, device, epoch, num_epochs, scaler):
    """CHANGED: Added FP16 support with caption-level batching"""
    model.train()
    total_loss = 0
    batch_count = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch_idx, batch in enumerate(progress_bar):
        if batch is None:
            continue

        # Move batch to device
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Debug first batch
        if epoch == 0 and batch_idx == 0:
            log_and_print(f"\n📊 First batch debug:")
            log_and_print(f"Pixel values shape: {pixel_values.shape}")
            log_and_print(f"Input IDs shape: {input_ids.shape}")
            log_and_print(f"Batch size: {len(pixel_values)}")

        # Prepare labels
        labels = input_ids.clone()
        labels[labels == model.config.pad_token_id] = -100

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            loss = outputs.loss

        # Check for anomalies
        if torch.isnan(loss) or torch.isinf(loss):
            log_and_print(f"⚠️ WARNING: Invalid loss at batch {batch_idx}: {loss.item()}")
            continue

        # Scaled backward pass
        scaler.scale(loss).backward()

        # Unscale before gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))

        if math.isnan(grad_norm):
            log_and_print("⚠️ CRITICAL: NaN gradients detected!")
            optimizer.zero_grad()
            scaler.update()
            continue

        # Scaled optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Update metrics
        current_loss = loss.item()
        total_loss += current_loss
        batch_count += 1

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'avg_loss': f'{total_loss / batch_count:.4f}',
            'grad_norm': f'{grad_norm:.2f}'
        })

        # Sample generation for monitoring
        if epoch == 0 and batch_idx % 100 == 0 and batch_idx > 0:
            model.eval()
            with torch.no_grad():
                # Generate from first image in batch
                sample_out = model.generate(
                    pixel_values[:1],
                    max_length=20,
                    num_beams=1
                )
                sample_caption = dataloader.dataset.processor.decode(sample_out[0], skip_special_tokens=True)

                # Get metadata
                image_id = batch['image_ids'][0] if 'image_ids' in batch else 'N/A'
                file_name = batch['file_names'][0] if 'file_names' in batch else 'N/A'

                log_and_print(f"\n🧪 Sample Generation at Epoch {epoch+1} Step {batch_idx}")
                log_and_print(f"🖼️ Image ID: {image_id}")
                log_and_print(f"📝 File Name: {file_name}")
                log_and_print(f"🧠 Generated: {sample_caption}\n")

            model.train()

    return total_loss / max(batch_count, 1)


# ==============================================================================
# Simple validation without label smoothing - FIXED without FP16
# ==============================================================================

def validate_simple(model, dataloader, device):
    """CHANGED: Validation with FP16 support"""
    model.eval()
    total_loss = 0
    batch_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if batch is None:
                continue

            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            labels = input_ids.clone()
            labels[labels == model.config.pad_token_id] = -100

            # Use autocast for validation too
            with torch.cuda.amp.autocast():
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True
                )

            if not torch.isnan(outputs.loss):
                total_loss += outputs.loss.item()
                batch_count += 1

    return total_loss / max(batch_count, 1)

# ==============================================================================
# Conservative Generation
# ==============================================================================
def generate_captions_conservative(model, pixel_values):
    """Very conservative generation settings"""
    return model.generate(
        pixel_values,
        max_length=30,  # Shorter
        num_beams=3,    # Greedy first
        temperature=1.0,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )

# ==============================================================================
# Minimal evaluation
# ==============================================================================
def evaluate_mini_simple(model, processor, dataloader, device, epoch, num_samples=50):
    """Simple evaluation with debugging + saving to file"""
    model.eval()
    successful_captions = []
    degenerate_count = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if batch is None:
                continue
            if len(successful_captions) >= num_samples:
                break

            pixel_values = batch['pixel_values'].to(device)

            # Generate for each image in batch
            generated_ids = generate_captions_conservative(model, pixel_values)

            # Decode all
            for j in range(len(generated_ids)):
                if len(successful_captions) >= num_samples:
                    break

                caption = processor.decode(generated_ids[j], skip_special_tokens=True)

                # Check if degenerate
                if len(caption) > 0 and not all(c == caption[0] for c in caption):
                    successful_captions.append(caption)
                else:
                    degenerate_count += 1
                    if degenerate_count <= 5:
                        log_and_print(f"Degenerate output: '{caption}'")

    log_and_print(f"Generated {len(successful_captions)} valid captions, {degenerate_count} degenerate")

    # 💾 Save valid captions to file
    output_path = OUTPUT_DIR / f"generated_captions_epoch{epoch+1}.json"
    with open(output_path, "w") as f:
        json.dump(successful_captions, f, indent=2)
    log_and_print(f"📝 Saved generated captions to {output_path}")

    return successful_captions



# ==============================================================================
# BLEU Score Evaluation
# ==============================================================================
def evaluate_mini_bleu(model, processor, dataloader, device, num_samples=50):
    """Evaluate BLEU-1 to BLEU-4 on small validation subset"""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        log_and_print("❌ NLTK not available, skipping BLEU evaluation")
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    model.eval()
    predictions, references = [], []
    count = 0
    smoothing = SmoothingFunction().method1

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if batch is None:
                continue
            if count >= num_samples:
                break

            pixel_values = batch['pixel_values'].to(device)

            # Generate captions
            generated_ids = generate_captions_conservative(model, pixel_values)

            # Process each generation in batch
            for j in range(len(generated_ids)):
                if count >= num_samples:
                    break

                pred = processor.decode(generated_ids[j], skip_special_tokens=True)

                # Get reference captions if available
                if 'all_captions' in batch and j < len(batch['all_captions']):
                    refs = batch['all_captions'][j]
                else:
                    continue

                # Check for valid prediction
                if pred and len(pred.strip()) > 0 and not all(c == pred[0] for c in pred):
                    predictions.append(pred.lower().split())
                    references.append([r.lower().split() for r in refs])
                    count += 1

    if not predictions:
        log_and_print("❌ No valid predictions for BLEU evaluation")
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    scores = {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": []}

    try:
        for pred, refs in zip(predictions, references):
            scores["bleu1"].append(sentence_bleu(refs, pred, weights=(1, 0, 0, 0), smoothing_function=smoothing))
            scores["bleu2"].append(sentence_bleu(refs, pred, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing))
            scores["bleu3"].append(sentence_bleu(refs, pred, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing))
            scores["bleu4"].append(sentence_bleu(refs, pred, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing))
    except Exception as e:
        log_and_print(f"❌ BLEU calculation failed: {e}")
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    # Handle empty scores
    final_scores = {}
    for k, v in scores.items():
        if v:
            final_scores[k] = sum(v) / len(v)
        else:
            final_scores[k] = 0.0

    log_and_print(f"📊 BLEU evaluation completed on {len(predictions)} samples")
    return final_scores

# ==============================================================================
# BLEU Score Evaluation
# ==============================================================================
def plot_training_curves(history):
    """Create training curves with error handling"""
    try:
        import matplotlib.pyplot as plt
        epochs = history['epochs']

        # ✅ Check if we have data to plot
        if not epochs:
            log_and_print("❌ No epoch data to plot")
            return

        plt.figure(figsize=(12, 5))

        # Loss Curve
        plt.subplot(1, 2, 1)
        if history['train_loss'] and history['val_loss']:
            plt.plot(epochs, history['train_loss'], label="Train Loss", marker='o')
            plt.plot(epochs, history['val_loss'], label="Val Loss", marker='s')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss Curve")
            plt.legend()
            plt.grid(True, alpha=0.3)

        # BLEU Curve
        plt.subplot(1, 2, 2)
        if any(history[k] for k in ['bleu1', 'bleu2', 'bleu3', 'bleu4']):
            for bleu_key, label in [('bleu1', 'BLEU-1'), ('bleu2', 'BLEU-2'),
                                   ('bleu3', 'BLEU-3'), ('bleu4', 'BLEU-4')]:
                if history[bleu_key]:
                    plt.plot(epochs, history[bleu_key], label=label, marker='o')
            plt.xlabel("Epoch")
            plt.ylabel("BLEU Score")
            plt.title("BLEU Score Curve")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = OUTPUT_DIR / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        log_and_print(f"📊 Training curves saved to: {plot_path}")

    except Exception as e:
        log_and_print(f"❌ Failed to create training curves: {e}")

# ==============================================================================
# Collate - CHANGED for multiple captions
# ==============================================================================

def collate_fn(batch):
    """Custom collate function that stacks tensors properly"""
    # Filter out None entries
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    # Stack all tensor fields
    collated = {}
    keys_to_stack = ['pixel_values', 'input_ids', 'attention_mask']

    for key in keys_to_stack:
        if key in batch[0]:
            collated[key] = torch.stack([item[key] for item in batch])

    # Keep metadata as lists
    if 'image_id' in batch[0]:
        collated['image_ids'] = [item['image_id'] for item in batch]
    if 'file_name' in batch[0]:
        collated['file_names'] = [item['file_name'] for item in batch]
    if 'all_captions' in batch[0]:
        collated['all_captions'] = [item['all_captions'] for item in batch]

    return collated

# ==============================================================================
# Main function - CHANGED for GitHub approach
# ==============================================================================
def main(args):
    """Main training function - CHANGED for GitHub approach"""
    # CHANGED: Initialize GradScaler for FP16
    scaler = torch.cuda.amp.GradScaler()

    # ==========================================================================
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_and_print(f"Using device: {device}")

    # STEP 1: Test pretrained model first
    if not test_pretrained_model(device):
        raise ValueError("Pretrained model test failed!")
    # ==========================================================================
    # Load model and processor
    log_and_print(f"\nLoading model: {args['model_name']}")
    processor = BlipProcessor.from_pretrained(args['model_name'])
    model = BlipForConditionalGeneration.from_pretrained(args['model_name'])
    if model.config.pad_token_id is None:
        model.config.pad_token_id = processor.tokenizer.pad_token_id
    assert model.config.pad_token_id is not None
    log_and_print(f"✅ pad_token_id = {model.config.pad_token_id}")

    # Basic model setup - no fancy dropout
    model = model.to(device)

    # CHANGED: DO NOT freeze encoder - train full model
    # for param in model.vision_model.parameters():
    #     param.requires_grad = False
    log_and_print("🔓 Training FULL model (encoder + decoder)")

    # ==========================================================================
    # Prepare resume logic BEFORE optimizer
    resume_path = OUTPUT_DIR / "checkpoints/last_checkpoint.pth"
    start_epoch = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'bleu1': [],
        'bleu2': [],
        'bleu3': [],
        'bleu4': [],
        'epochs': []
    }
    patience_counter = 0
    optimizer_state = None
    scheduler_state = None

    if resume_path.exists() and args.get('resume', False):
        log_and_print(f"🔁 Found checkpoint at {resume_path}, resuming...")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer_state = checkpoint['optimizer_state']
        scheduler_state = checkpoint['scheduler_state']
        history = checkpoint['history']
        start_epoch = checkpoint['epoch']
        patience_counter = checkpoint['patience_counter']
    else:
        if resume_path.exists():
            log_and_print(f"⚠️ Found old checkpoint but starting fresh (different optimizer config)")
            # Optionally rename old checkpoint to avoid confusion
            import shutil
            backup_path = OUTPUT_DIR / "checkpoints/old_last_checkpoint.pth"
            if resume_path.exists():
                shutil.move(str(resume_path), str(backup_path))
                log_and_print(f"📁 Moved old checkpoint to: {backup_path}")
    # ==========================================================================
    # Dataset setup
    train_dataset = RSICDDatasetMinimal(args['train_json'], args['image_dir'], processor, 'train')
    val_dataset = RSICDDatasetMinimal(args['val_json'], args['image_dir'], processor, 'val')

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,  # CHANGED: Parallel data loading
        pin_memory=True,  # CHANGED: Faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,  # CHANGED: Parallel data loading
        pin_memory=True,  # CHANGED: Faster GPU transfer
    )

    # ==========================================================================

    # Optimizer - CHANGED: all parameters now
    optimizer = torch.optim.AdamW(
        model.parameters(),  # CHANGED: removed filter
        lr=args['learning_rate'],
        eps=1e-8,
        weight_decay=args['weight_decay']  # Added weight decay
    )

    # CHANGED: Use ReduceLROnPlateau instead of linear schedule
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
        verbose=True
    )

    # Restore optimizer/scheduler if resuming
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        log_and_print(f"🔄 Optimizer and scheduler state restored")

    # ==========================================================================
    # Training loop
    log_and_print("\n🚀 Starting minimal training...")
    log_and_print("="*60)

    best_val_loss = float('inf')
    for epoch in range(start_epoch, args['num_epochs']):
        epoch_start = time.time()

        # Train with monitoring - with FP16
        train_loss = train_epoch_with_monitoring(model, train_loader, optimizer, scheduler, device, epoch, args['num_epochs'], scaler)
        # Validate
        val_loss = validate_simple(model, val_loader, device)

        # CHANGED: Step scheduler with val_loss
        scheduler.step(val_loss)

        # Generation and BLEU
        sample_captions = evaluate_mini_simple(model, processor, val_loader, device, epoch, num_samples=10)
        for i, caption in enumerate(sample_captions[:3]):
            log_and_print(f"  Sample {i+1}: {caption}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch + 1)
        # ======================================================================
        log_and_print(f"\n📊 Computing BLEU scores for epoch {epoch+1}...")
        try:
            # 📊 Metric for Accuracy Curve
            bleu_scores = evaluate_mini_bleu(model, processor, val_loader, device, num_samples=50)
            history['bleu1'].append(bleu_scores['bleu1'])
            history['bleu2'].append(bleu_scores['bleu2'])
            history['bleu3'].append(bleu_scores['bleu3'])
            history['bleu4'].append(bleu_scores['bleu4'])

            log_and_print(f"BLEU-1: {bleu_scores['bleu1']:.4f}")
            log_and_print(f"BLEU-2: {bleu_scores['bleu2']:.4f}")
            log_and_print(f"BLEU-3: {bleu_scores['bleu3']:.4f}")
            log_and_print(f"BLEU-4: {bleu_scores['bleu4']:.4f}")
            log_and_print(f"Time: {time.time() - epoch_start:.1f}s")
        except Exception as e:
            log_and_print(f"❌ BLEU evaluation failed: {e}")
            history['bleu1'].append(0.0)
            history['bleu2'].append(0.0)
            history['bleu3'].append(0.0)
            history['bleu4'].append(0.0)
        # ======================================================================
        # Save best model if val_loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_path = OUTPUT_DIR / "best_model"
            model_path.mkdir(exist_ok=True)
            model.save_pretrained(model_path)
            processor.save_pretrained(model_path)
            log_and_print("✅ Saved best model (val_loss improved)")
        else:
            patience_counter += 1
            log_and_print(f"⏳ EarlyStopping counter: {patience_counter} / {args['early_stopping_patience']}")
            if patience_counter > args['early_stopping_patience']:
                log_and_print("⏹️ Early stopping triggered due to no val_loss improvement")
                break
        # ======================================================================
        # Save history
        history_file = OUTPUT_DIR / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

        # 💾 Save checkpoint
        checkpoint_dir = OUTPUT_DIR / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'history': history,
            'epoch': epoch + 1,
            'patience_counter': patience_counter
        }
        torch.save(checkpoint, checkpoint_dir / "last_checkpoint.pth")
        log_and_print(f"💾 Checkpoint saved: epoch {epoch+1}")
    return model, processor, history


# ==============================================================================
# Run the minimal training
# ==============================================================================

# if __name__ == "__main__":
try:
    model, processor, history = main(training_args)

    # Plot Curves
    plot_training_curves(history)

except Exception as e:
    log_and_print(f"\n❌ Training failed with error: {e}")
    raise