import os
import random
from PIL import Image
import numpy as np

# Paths
input_folder = "preprocessed"
clean_output_folder = "noiseprint_clean"   # clean images
noisy_output_folder = "noiseprint_noisy"   # noisy images
mask_folder = "noiseprint_gt"              # ground truth masks

# Create folders if they don't exist
os.makedirs(clean_output_folder, exist_ok=True)
os.makedirs(noisy_output_folder, exist_ok=True)
os.makedirs(mask_folder, exist_ok=True)

# Parameters
num_samples = 50
noise_std = 30  # Gaussian noise standard deviation

# Get all images
images = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
          if f.lower().endswith(".png")]

if len(images) < 2:
    raise ValueError("Need at least 2 images for patch swapping.")

for i in range(num_samples):
    # Pick two random images
    img1_path, img2_path = random.sample(images, 2)
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    w, h = img1.size

    # Crop a random rectangle (1/5th size)
    crop_w, crop_h = int(w / 5), int(h / 5)
    x1 = random.randint(0, w - crop_w)
    y1 = random.randint(0, h - crop_h)
    crop_box = (x1, y1, x1 + crop_w, y1 + crop_h)

    # Take a random patch from the second image
    w2, h2 = img2.size
    x2 = random.randint(0, w2 - crop_w)
    y2 = random.randint(0, h2 - crop_h)
    patch = img2.crop((x2, y2, x2 + crop_w, y2 + crop_h))

    # Paste the patch into the first image
    img1.paste(patch, crop_box)

    # Save **clean image**
    clean_path = os.path.join(clean_output_folder, f"sample_{i+1:03d}.png")
    img1.save(clean_path)

    # Add Gaussian noise
    img_np = np.array(img1).astype(np.float32)
    noise = np.random.normal(0, noise_std, img_np.shape)
    img_noisy = img_np + noise
    img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)

    # Save **noisy image**
    noisy_path = os.path.join(noisy_output_folder, f"sample_{i+1:03d}.png")
    Image.fromarray(img_noisy).save(noisy_path)

    # Create ground truth mask
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y1+crop_h, x1:x1+crop_w] = 1
    mask_img = Image.fromarray(mask * 255)  # scale 0/1 → 0/255
    mask_path = os.path.join(mask_folder, f"mask_{i+1:03d}.png")
    mask_img.save(mask_path)

print(f"Dataset ready:")
print(f"- Clean images: {clean_output_folder}")
print(f"- Noisy images: {noisy_output_folder}")
print(f"- Ground truth masks: {mask_folder}")
