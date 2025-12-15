"""
Prepare nighttime dehazing dataset for CycleGAN training.

This script:
1. Combines hazy images from: my_hazy + Internet_night_fog
2. Combines clean images from: night_clean + Internet_night_clean1 + Internet_night_clean2
3. Resizes all images to 512x512
4. Splits into train/test sets (80/20)
5. Creates CycleGAN folder structure: trainA, trainB, testA, testB

Usage:
    python prepare_dehaze_dataset.py
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import random

# Configuration
SOURCE_DIR = Path("../datasets")
OUTPUT_DIR = Path("./datasets/nighttime_dehaze_512")
TARGET_SIZE = (512, 512)
TRAIN_SPLIT = 0.8  # 80% train, 20% test

# Source folders
HAZY_FOLDERS = [
    SOURCE_DIR / "my_hazy",
    SOURCE_DIR / "Internet_night_fog"
]

CLEAN_FOLDERS = [
    SOURCE_DIR / "night_clean",
    SOURCE_DIR / "Internet_night_clean1",
    SOURCE_DIR / "Internet_night_clean2"
]

# Image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


def get_image_files(folder):
    """Get all image files from a folder."""
    folder = Path(folder)
    if not folder.exists():
        print(f"Warning: Folder {folder} does not exist")
        return []

    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(folder.glob(f"*{ext}"))
        images.extend(folder.glob(f"*{ext.upper()}"))

    return sorted(images)


def resize_and_save(src_path, dst_path, size=(512, 512)):
    """
    Resize image to target size and save.

    Args:
        src_path: Source image path
        dst_path: Destination path
        size: Target size (width, height)
    """
    try:
        # Open image
        img = Image.open(src_path).convert('RGB')

        # Resize with high-quality resampling
        img_resized = img.resize(size, Image.Resampling.LANCZOS)

        # Save as JPEG with high quality
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img_resized.save(dst_path, 'JPEG', quality=95)

        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False


def main():
    print("Nighttime Dehazing Dataset Preparation")

    # Create output directories
    trainA_dir = OUTPUT_DIR / "trainA"
    trainB_dir = OUTPUT_DIR / "trainB"
    testA_dir = OUTPUT_DIR / "testA"
    testB_dir = OUTPUT_DIR / "testB"

    for dir_path in [trainA_dir, trainB_dir, testA_dir, testB_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Target image size: {TARGET_SIZE}")
    print(f"Train/Test split: {TRAIN_SPLIT*100:.0f}% / {(1-TRAIN_SPLIT)*100:.0f}%\n")

    # ========== Process HAZY images (Domain A) ==========
    print("Processing HAZY images (Domain A)")

    hazy_images = []
    for folder in HAZY_FOLDERS:
        images = get_image_files(folder)
        print(f"{folder.name}: {len(images)} images")
        hazy_images.extend(images)

    print(f"\nTotal hazy images: {len(hazy_images)}")

    # Shuffle and split
    random.seed(42)
    random.shuffle(hazy_images)

    n_train_hazy = int(len(hazy_images) * TRAIN_SPLIT)
    train_hazy = hazy_images[:n_train_hazy]
    test_hazy = hazy_images[n_train_hazy:]

    print(f"Train hazy: {len(train_hazy)}")
    print(f"Test hazy: {len(test_hazy)}")

    # Process train hazy
    print("\nResizing and copying train hazy images...")
    success_count = 0
    for i, src_path in enumerate(train_hazy, 1):
        dst_path = trainA_dir / f"hazy_{i:04d}.jpg"
        if resize_and_save(src_path, dst_path, TARGET_SIZE):
            success_count += 1

        if i % 50 == 0:
            print(f"  Processed {i}/{len(train_hazy)}...")

    print(f"✓ Successfully processed {success_count}/{len(train_hazy)} train hazy images")

    # Process test hazy
    print("\nResizing and copying test hazy images...")
    success_count = 0
    for i, src_path in enumerate(test_hazy, 1):
        dst_path = testA_dir / f"hazy_{i:04d}.jpg"
        if resize_and_save(src_path, dst_path, TARGET_SIZE):
            success_count += 1

    print(f"✓ Successfully processed {success_count}/{len(test_hazy)} test hazy images")

    # ========== Process CLEAN images (Domain B) ==========
    print("\n" + "-" * 60)
    print("Processing CLEAN images (Domain B)")
    print("-" * 60)

    clean_images = []
    for folder in CLEAN_FOLDERS:
        images = get_image_files(folder)
        print(f"{folder.name}: {len(images)} images")
        clean_images.extend(images)

    print(f"\nTotal clean images: {len(clean_images)}")

    # Shuffle and split
    random.shuffle(clean_images)

    n_train_clean = int(len(clean_images) * TRAIN_SPLIT)
    train_clean = clean_images[:n_train_clean]
    test_clean = clean_images[n_train_clean:]

    print(f"Train clean: {len(train_clean)}")
    print(f"Test clean: {len(test_clean)}")

    # Process train clean
    print("\nResizing and copying train clean images...")
    success_count = 0
    for i, src_path in enumerate(train_clean, 1):
        dst_path = trainB_dir / f"clean_{i:04d}.jpg"
        if resize_and_save(src_path, dst_path, TARGET_SIZE):
            success_count += 1

        if i % 50 == 0:
            print(f"  Processed {i}/{len(train_clean)}...")

    print(f"✓ Successfully processed {success_count}/{len(train_clean)} train clean images")

    # Process test clean
    print("\nResizing and copying test clean images...")
    success_count = 0
    for i, src_path in enumerate(test_clean, 1):
        dst_path = testB_dir / f"clean_{i:04d}.jpg"
        if resize_and_save(src_path, dst_path, TARGET_SIZE):
            success_count += 1

    print(f"✓ Successfully processed {success_count}/{len(test_clean)} test clean images")

    # ========== Summary ==========
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"\nDataset location: {OUTPUT_DIR.absolute()}")
    print("\nFinal counts:")
    print(f"  trainA (hazy):  {len(list(trainA_dir.glob('*.jpg')))} images")
    print(f"  trainB (clean): {len(list(trainB_dir.glob('*.jpg')))} images")
    print(f"  testA (hazy):   {len(list(testA_dir.glob('*.jpg')))} images")
    print(f"  testB (clean):  {len(list(testB_dir.glob('*.jpg')))} images")

    print("\n" + "=" * 60)
    print("Ready to train! Use this command:")
    print("=" * 60)
    print(f"""
python train.py \\
  --dataroot ./datasets/nighttime_dehaze_512 \\
  --name nighttime_dehaze_512 \\
  --model cycle_gan \\
  --netG resnet_9blocks \\
  --load_size 512 \\
  --crop_size 512 \\
  --use_custom_losses \\
  --lambda_A 5.0 \\
  --lambda_B 5.0 \\
  --lambda_tv 0.1 \\
  --batch_size 1 \\
  --gpu_ids 0
""")


if __name__ == "__main__":
    main()
