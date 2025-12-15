"""
Simple high-resolution testing script for nighttime dehazing.
Processes entire images by resizing to a reasonable resolution.
"""

import os
import torch
import numpy as np
from PIL import Image
import argparse
import sys


def load_model(checkpoint_path, epoch, gpu_id):
    """Load the trained CycleGAN model."""
    from options.test_options import TestOptions
    from models import create_model

    checkpoint_name = os.path.basename(checkpoint_path)
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Backup and modify sys.argv
    original_argv = sys.argv.copy()
    sys.argv = [
        'test_simple.py',
        '--dataroot', 'dummy',
        '--name', checkpoint_name,
        '--model', 'cycle_gan',
        '--checkpoints_dir', checkpoint_dir,
        '--epoch', epoch,
        '--no_dropout',
        '--num_test', '1'
    ]

    opt = TestOptions().parse()

    # Set GPU
    opt.gpu_ids = [int(gpu_id)] if int(gpu_id) >= 0 else []
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
        opt.device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))
    else:
        opt.device = torch.device('cpu')

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    sys.argv = original_argv

    return model, opt


def process_image(model, image_path, max_size=1024, keep_resized=True):
    """
    Process a single image.

    Args:
        model: Trained CycleGAN model
        image_path: Path to input image
        max_size: Maximum dimension (maintains aspect ratio)
        keep_resized: If True, output stays at resized resolution. If False, upscale back to original.

    Returns:
        Processed PIL Image
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    original_size = img.size

    # Resize if needed (maintain aspect ratio)
    w, h = img.size
    if max(w, h) > max_size:
        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)

        # Make dimensions divisible by 4 (for network compatibility)
        new_w = (new_w // 4) * 4
        new_h = (new_h // 4) * 4

        img = img.resize((new_w, new_h), Image.BICUBIC)
        print(f"  Resized: {original_size[0]}x{original_size[1]} -> {new_w}x{new_h}")
    else:
        # Still ensure divisible by 4
        new_w = (w // 4) * 4
        new_h = (h // 4) * 4
        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h), Image.BICUBIC)

    # Convert to tensor
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0)
    img_tensor = img_tensor * 2 - 1  # Normalize to [-1, 1]
    img_tensor = img_tensor.to(model.device)

    # Process
    with torch.no_grad():
        output_tensor = model.netG_A(img_tensor)

    # Convert back to image
    output_np = output_tensor.squeeze(0).cpu().numpy()
    output_np = (output_np + 1) / 2  # Denormalize to [0, 1]
    output_np = np.clip(output_np.transpose(1, 2, 0), 0, 1)
    output_img = Image.fromarray((output_np * 255).astype(np.uint8))

    # Resize back to original size if requested
    if not keep_resized and output_img.size != original_size:
        output_img = output_img.resize(original_size, Image.BICUBIC)
        print(f"  Resized back to original: {original_size[0]}x{original_size[1]}")
    else:
        print(f"  Output size: {output_img.size[0]}x{output_img.size[1]}")

    return output_img


def main():
    parser = argparse.ArgumentParser(description='Simple high-resolution nighttime dehazing')
    parser.add_argument('--input', type=str, required=True, help='Input image or folder')
    parser.add_argument('--output', type=str, required=True, help='Output folder')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/nighttime_dehaze_combined',
                       help='Checkpoint directory')
    parser.add_argument('--epoch', type=str, default='200', help='Which epoch to load')
    parser.add_argument('--max_size', type=int, default=1024,
                       help='Maximum dimension for processing (maintains aspect ratio)')
    parser.add_argument('--gpu_ids', type=str, default='1', help='GPU id: e.g. 0, 1, or -1 for CPU')
    parser.add_argument('--upscale', action='store_true',
                       help='Upscale output back to original size (default: keep at resized resolution)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}, epoch {args.epoch}...")
    model, opt = load_model(args.checkpoint, args.epoch, args.gpu_ids)
    print(f"Model loaded successfully on device: {model.device}")

    # Get input files
    if os.path.isfile(args.input):
        input_files = [args.input]
    else:
        input_files = [os.path.join(args.input, f) for f in os.listdir(args.input)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    print(f"Found {len(input_files)} images to process\n")

    # Process each image
    for i, img_path in enumerate(input_files):
        print(f"[{i+1}/{len(input_files)}] Processing: {os.path.basename(img_path)}")

        try:
            output_img = process_image(model, img_path, max_size=args.max_size, keep_resized=not args.upscale)

            # Save
            output_path = os.path.join(args.output, os.path.basename(img_path))
            output_img.save(output_path, quality=95)
            print(f"  Saved to: {output_path}\n")

        except Exception as e:
            print(f"  ERROR: {str(e)}\n")
            import traceback
            traceback.print_exc()
            continue

    print("✅ Done!")


if __name__ == '__main__':
    main()
