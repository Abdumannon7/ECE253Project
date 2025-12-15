import os
import random
from PIL import Image
import numpy as np

# Paths
input_folder = "dehazed"   # folder containing *_A.png and *_B.png

out = {
    "hazy": {
        "suffix": "_A.png",
        "clean": "noiseprint_clean_hazy",
        "mask":  "noiseprint_gt_hazy",
    },
    "dehazed": {
        "suffix": "_B.png",
        "clean": "noiseprint_clean_dehazed",
        "mask":  "noiseprint_gt_dehazed",
    }
}

num_samples = 25

# Create output folders
for cfg in out.values():
    os.makedirs(cfg["clean"], exist_ok=True)
    os.makedirs(cfg["mask"], exist_ok=True)

# -------------------------------------------------
# Build paired image dictionary
# -------------------------------------------------
pairs = {}

for f in os.listdir(input_folder):
    if f.endswith("_A.png"):
        key = f.replace("_A.png", "")
        pairs.setdefault(key, {})["A"] = f
    elif f.endswith("_B.png"):
        key = f.replace("_B.png", "")
        pairs.setdefault(key, {})["B"] = f

# Keep only complete pairs
pairs = {k: v for k, v in pairs.items() if "A" in v and "B" in v}

keys = sorted(pairs.keys())

if len(keys) < 2:
    raise ValueError("Need at least 2 paired images")

# -------------------------------------------------
# Dataset generation
# -------------------------------------------------
for i in range(num_samples):

    # Choose base and donor (same for A & B)
    base_key, donor_key = random.sample(keys, 2)

    base_A = Image.open(os.path.join(input_folder, pairs[base_key]["A"])).convert("RGB")
    base_B = Image.open(os.path.join(input_folder, pairs[base_key]["B"])).convert("RGB")

    donor_A = Image.open(os.path.join(input_folder, pairs[donor_key]["A"])).convert("RGB")
    donor_B = Image.open(os.path.join(input_folder, pairs[donor_key]["B"])).convert("RGB")

    w, h = base_A.size
    crop_w, crop_h = w // 5, h // 5

    # SAME random crop location
    x1 = random.randint(0, w - crop_w)
    y1 = random.randint(0, h - crop_h)
    crop_box = (x1, y1, x1 + crop_w, y1 + crop_h)

    # SAME random patch location
    x2 = random.randint(0, w - crop_w)
    y2 = random.randint(0, h - crop_h)
    patch_box = (x2, y2, x2 + crop_w, y2 + crop_h)

    patch_A = donor_A.crop(patch_box)
    patch_B = donor_B.crop(patch_box)

    # Paste patches
    base_A.paste(patch_A, crop_box)
    base_B.paste(patch_B, crop_box)

    # Save images
    base_A.save(os.path.join(out["hazy"]["clean"], "sample_{:03d}.png".format(i + 1)))
    base_B.save(os.path.join(out["dehazed"]["clean"], "sample_{:03d}.png".format(i + 1)))

    # Ground-truth mask (shared)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y1 + crop_h, x1:x1 + crop_w] = 1
    mask_img = Image.fromarray(mask * 255)

    mask_img.save(os.path.join(out["hazy"]["mask"], "mask_{:03d}.png".format(i + 1)))
    mask_img.save(os.path.join(out["dehazed"]["mask"], "mask_{:03d}.png".format(i + 1)))

print("Paired hazy/dehazed NoisePrint datasets created successfully.")
