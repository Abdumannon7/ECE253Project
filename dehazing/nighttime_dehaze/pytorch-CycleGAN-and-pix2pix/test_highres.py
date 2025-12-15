"""
High-resolution testing script for nighttime dehazing.
Processes large images in patches to avoid downsampling artifacts.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from options.test_options import TestOptions
from models import create_model
from util import util
import argparse


def process_image_patches(model, image, patch_size=256, overlap=32):
    """
    Process a large image in overlapping patches.

    Args:
        model: Trained CycleGAN model
        image: PIL Image
        patch_size: Size of each patch (should match training crop_size)
        overlap: Overlap between patches to avoid seam artifacts

    Returns:
        Processed PIL Image
    """
    # Convert to tensor
    img_np = np.array(image).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW

    h, w = img_np.shape[1], img_np.shape[2]

    # Prepare output
    output = np.zeros_like(img_np)
    weight_map = np.zeros((h, w), dtype=np.float32)

    stride = patch_size - overlap

    # Process patches
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Extract patch
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)

            patch = img_np[:, y:y_end, x:x_end]

            # Pad if needed
            pad_h = patch_size - patch.shape[1]
            pad_w = patch_size - patch.shape[2]

            if pad_h > 0 or pad_w > 0:
                patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

            # Process patch
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(model.device)
            patch_tensor = patch_tensor * 2 - 1  # Normalize to [-1, 1]

            with torch.no_grad():
                output_patch = model.netG_A(patch_tensor)

            output_patch = output_patch.squeeze(0).cpu().numpy()
            output_patch = (output_patch + 1) / 2  # Denormalize to [0, 1]

            # Remove padding
            if pad_h > 0 or pad_w > 0:
                output_patch = output_patch[:, :patch.shape[1]-pad_h, :patch.shape[2]-pad_w]

            # Create weight mask (higher weight in center, lower at edges)
            patch_weight = np.ones((output_patch.shape[1], output_patch.shape[2]), dtype=np.float32)

            if overlap > 0:
                # Create smooth blending weights
                for i in range(overlap):
                    alpha = i / overlap
                    if y > 0:  # Top edge
                        patch_weight[i, :] *= alpha
                    if x > 0:  # Left edge
                        patch_weight[:, i] *= alpha
                    if y_end < h:  # Bottom edge
                        patch_weight[-(i+1), :] *= alpha
                    if x_end < w:  # Right edge
                        patch_weight[:, -(i+1)] *= alpha

            # Accumulate
            actual_h = output_patch.shape[1]
            actual_w = output_patch.shape[2]
            output[:, y:y+actual_h, x:x+actual_w] += output_patch * patch_weight[np.newaxis, :, :]
            weight_map[y:y+actual_h, x:x+actual_w] += patch_weight

    # Normalize by weight
    output = output / (weight_map[np.newaxis, :, :] + 1e-8)
    output = np.clip(output, 0, 1)

    # Convert back to PIL
    output = (output.transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(output)


def main():
    parser = argparse.ArgumentParser(description='High-resolution nighttime dehazing')
    parser.add_argument('--input', type=str, required=True, help='Input image or folder')
    parser.add_argument('--output', type=str, required=True, help='Output folder')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/nighttime_dehaze_combined', help='Checkpoint directory')
    parser.add_argument('--epoch', type=str, default='200', help='Which epoch to load')
    parser.add_argument('--patch_size', type=int, default=256, help='Patch size (should match training crop_size)')
    parser.add_argument('--overlap', type=int, default=32, help='Overlap between patches')
    parser.add_argument('--gpu_ids', type=str, default='1', help='GPU ids: e.g. 0  0,1,2  -1 for CPU')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}, epoch {args.epoch}...")

    # Create test options by modifying sys.argv
    checkpoint_name = os.path.basename(args.checkpoint)
    checkpoint_dir = os.path.dirname(args.checkpoint)

    # Backup original sys.argv
    original_argv = sys.argv.copy()

    # Set sys.argv for TestOptions parser
    sys.argv = [
        'test_highres.py',
        '--dataroot', 'dummy',  # Won't be used
        '--name', checkpoint_name,
        '--model', 'cycle_gan',
        '--checkpoints_dir', checkpoint_dir,
        '--epoch', args.epoch,
        '--no_dropout',
        '--num_test', '1'
    ]

    opt = TestOptions().parse()

    # Manually set GPU IDs after parsing
    str_ids = args.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)

    # Set device attribute
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
        opt.device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))
    else:
        opt.device = torch.device('cpu')

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    # Restore sys.argv
    sys.argv = original_argv

    print(f"Model loaded successfully on device: {model.device}")

    # Get input files
    if os.path.isfile(args.input):
        input_files = [args.input]
    else:
        input_files = [os.path.join(args.input, f) for f in os.listdir(args.input)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    print(f"Found {len(input_files)} images to process")

    # Process each image
    for i, img_path in enumerate(input_files):
        print(f"\n[{i+1}/{len(input_files)}] Processing: {os.path.basename(img_path)}")

        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            orig_size = image.size
            print(f"  Original size: {orig_size[0]}x{orig_size[1]}")

            # Process
            print(f"  Processing in {args.patch_size}x{args.patch_size} patches with {args.overlap}px overlap...")
            output_image = process_image_patches(
                model,
                image,
                patch_size=args.patch_size,
                overlap=args.overlap
            )

            # Save
            output_path = os.path.join(args.output, os.path.basename(img_path))
            output_image.save(output_path, quality=95)
            print(f"  Saved to: {output_path}")

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            continue

    print("\n✅ Done!")


if __name__ == '__main__':
    main()
