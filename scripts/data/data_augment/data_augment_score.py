#!/usr/bin/env python3

import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

# Folder where all the images are
IMAGE_DIR = "./images"  # Adjust this path if needed

# Get all image filenames
all_files = sorted(f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg"))

# Group base names
base_names = sorted(set(
    f.replace(".jpg", "").split("_v")[0] for f in all_files if "_v" not in f
))


print(f"\nEvaluating SSIM + MSE for {len(base_names)} image groups...\n")
print(f"{'Image':<25} | {'SSIM v1':<8} {'MSE v1':<10} | {'SSIM v2':<8} {'MSE v2':<10}")
print("-" * 70)

for base in base_names:
    try:
        # Load images in grayscale
        img_o = np.array(Image.open(os.path.join(IMAGE_DIR, f"{base}.jpg")).convert("L"))
        img_v1 = np.array(Image.open(os.path.join(IMAGE_DIR, f"{base}_v1.jpg")).convert("L"))
        img_v2 = np.array(Image.open(os.path.join(IMAGE_DIR, f"{base}_v2.jpg")).convert("L"))

        # Compute SSIM and MSE
        ssim_v1 = ssim(img_o, img_v1)
        mse_v1 = mean_squared_error(img_o.flatten(), img_v1.flatten())
        ssim_v2 = ssim(img_o, img_v2)
        mse_v2 = mean_squared_error(img_o.flatten(), img_v2.flatten())

        print(f"{base:<25} | {ssim_v1:.3f}    {mse_v1:<10.2f} | {ssim_v2:.3f}    {mse_v2:<10.2f}")
    except Exception as e:
        print(f"{base:<25} | Error: {e}")

