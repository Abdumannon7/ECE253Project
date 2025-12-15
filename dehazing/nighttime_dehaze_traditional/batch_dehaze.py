"""
Batch process nighttime dehazing on a folder of images.
"""

import os
import sys
from pathlib import Path
from improved_nighttime_dehaze import nighttime_dehaze_improved

def batch_dehaze(input_folder, output_folder,
                 denoise_sigma=0.06,
                 guided_radius=40,
                 guided_eps=0.002,
                 t_min=0.15,
                 gamma=1.2):
    """
    Batch process all images in a folder.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # Create output folder
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_path.glob(f'*{ext}')))
        image_files.extend(list(input_path.glob(f'*{ext.upper()}')))

    image_files = sorted(image_files)

    # Process each image
    for i, img_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {img_file.name}")

        # Output filename: keep same name but add _dehazed suffix
        output_file = output_path / f"{img_file.stem}_dehazed{img_file.suffix}"

        try:
            nighttime_dehaze_improved(
                str(img_file),
                str(output_file),
                denoise_sigma=denoise_sigma,
                guided_radius=guided_radius,
                guided_eps=guided_eps,
                t_min=t_min,
                gamma=gamma,
                visualize=False  # No visualization for batch processing
            )
            print(f"Saved to: {output_file.name}")
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")
            continue


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Batch Nighttime Dehazing')
    parser.add_argument('--input', type=str, required=True, help='Input folder with hazy images')
    parser.add_argument('--output', type=str, required=True, help='Output folder for dehazed images')
    parser.add_argument('--denoise_sigma', type=float, default=0.06, help='Denoising strength (0.05-0.08)')
    parser.add_argument('--guided_radius', type=int, default=40, help='Guided filter radius (20-60)')
    parser.add_argument('--guided_eps', type=float, default=0.002, help='Guided filter epsilon (1e-3 to 5e-3)')
    parser.add_argument('--t_min', type=float, default=0.15, help='Transmission floor (0.15-0.20)')
    parser.add_argument('--gamma', type=float, default=1.2, help='Gamma correction (1.0-1.5)')

    args = parser.parse_args()

    batch_dehaze(
        args.input,
        args.output,
        denoise_sigma=args.denoise_sigma,
        guided_radius=args.guided_radius,
        guided_eps=args.guided_eps,
        t_min=args.t_min,
        gamma=args.gamma
    )
