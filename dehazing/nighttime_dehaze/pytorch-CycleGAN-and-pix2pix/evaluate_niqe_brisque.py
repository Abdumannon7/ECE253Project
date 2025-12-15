"""
Calculate NIQE and BRISQUE (no-reference quality metrics) for dehazing results.
Compares foggy input images vs dehazed output images.
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import csv

# Use scikit-image for NIQE
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage import io, img_as_float
    import warnings
    warnings.filterwarnings('ignore')
    SKIMAGE_AVAILABLE = True
except:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available for NIQE")

# Try to import imquality for BRISQUE
try:
    import imquality.brisque as brisque
    from PIL import Image
    BRISQUE_AVAILABLE = True
except:
    BRISQUE_AVAILABLE = False
    print("Warning: imquality not available for BRISQUE, will try opencv")

# Try opencv brisque
try:
    import cv2
    # Check if cv2.quality is available
    OPENCV_BRISQUE_AVAILABLE = hasattr(cv2, 'quality')
except:
    OPENCV_BRISQUE_AVAILABLE = False


def calculate_niqe_skimage(image_path):
    """Calculate NIQE using scikit-image."""
    try:
        from skimage.metrics import niqe
        img = io.imread(image_path)
        if img.ndim == 3:
            # Convert to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
        score = niqe(img_gray)
        return score
    except Exception as e:
        print(f"Error calculating NIQE: {e}")
        return None


def calculate_brisque_imquality(image_path):
    """Calculate BRISQUE using imquality library."""
    try:
        img = Image.open(image_path)
        score = brisque.score(img)
        return score
    except Exception as e:
        print(f"Error calculating BRISQUE: {e}")
        return None


def calculate_brisque_opencv(image_path):
    """Calculate BRISQUE using OpenCV."""
    try:
        img = cv2.imread(image_path)
        brisque_model = cv2.quality.QualityBRISQUE_create()
        score = brisque_model.compute(img)[0]
        return score
    except Exception as e:
        print(f"Error calculating BRISQUE with OpenCV: {e}")
        return None


def calculate_entropy(image_path):
    """Calculate image entropy (information content)."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
    except Exception as e:
        print(f"Error calculating entropy: {e}")
        return None


def calculate_average_gradient(image_path):
    """Calculate average gradient (edge strength)."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(gx**2 + gy**2)
        avg_gradient = np.mean(gradient)
        return avg_gradient
    except Exception as e:
        print(f"Error calculating average gradient: {e}")
        return None


def calculate_saturation(image_path):
    """Calculate average saturation."""
    try:
        img = cv2.imread(image_path)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = img_hsv[:, :, 1].mean()
        return saturation
    except Exception as e:
        print(f"Error calculating saturation: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Calculate NIQE and BRISQUE for dehazing results')
    parser.add_argument('--foggy', type=str, required=True, help='Path to foggy input images folder')
    parser.add_argument('--dehazed', type=str, required=True, help='Path to dehazed results folder')
    parser.add_argument('--output', type=str, default='niqe_brisque_results.csv', help='Output CSV file')

    args = parser.parse_args()

    foggy_path = Path(args.foggy)
    dehazed_path = Path(args.dehazed)

    # Get list of dehazed images (fake_B from CycleGAN)
    dehazed_files = sorted(list(dehazed_path.glob('*_fake_B.png')))

    if len(dehazed_files) == 0:
        print('ERROR: No fake_B images found in dehazed folder!')
        print('Make sure you ran test.py and results contain *_fake_B.png files')
        return

    print(f'Found {len(dehazed_files)} dehazed images')
    print('='*70)

    # Storage for metrics
    results = []

    niqe_foggy_list = []
    niqe_dehazed_list = []
    brisque_foggy_list = []
    brisque_dehazed_list = []
    entropy_foggy_list = []
    entropy_dehazed_list = []
    gradient_foggy_list = []
    gradient_dehazed_list = []
    saturation_foggy_list = []
    saturation_dehazed_list = []

    valid_count = 0

    for i, dehazed_file in enumerate(dehazed_files):
        # Map dehazed filename to foggy filename (use real_A from same folder)
        # e.g., "foggy_0001_fake_B.png" -> "foggy_0001_real_A.png"
        foggy_name = dehazed_file.name.replace('_fake_B.png', '_real_A.png')
        foggy_file = dehazed_path / foggy_name

        if not foggy_file.exists():
            print(f'Warning: No foggy image found for {dehazed_file.name}, skipping...')
            continue

        print(f'[{i+1}/{len(dehazed_files)}] Processing: {foggy_name}')

        # Calculate metrics for foggy image
        print(f'  Computing metrics for foggy image...')
        niqe_foggy = calculate_niqe_skimage(str(foggy_file)) if SKIMAGE_AVAILABLE else None

        if BRISQUE_AVAILABLE:
            brisque_foggy = calculate_brisque_imquality(str(foggy_file))
        elif OPENCV_BRISQUE_AVAILABLE:
            brisque_foggy = calculate_brisque_opencv(str(foggy_file))
        else:
            brisque_foggy = None

        entropy_foggy = calculate_entropy(str(foggy_file))
        gradient_foggy = calculate_average_gradient(str(foggy_file))
        saturation_foggy = calculate_saturation(str(foggy_file))

        # Calculate metrics for dehazed image
        print(f'  Computing metrics for dehazed image...')
        niqe_dehazed = calculate_niqe_skimage(str(dehazed_file)) if SKIMAGE_AVAILABLE else None

        if BRISQUE_AVAILABLE:
            brisque_dehazed = calculate_brisque_imquality(str(dehazed_file))
        elif OPENCV_BRISQUE_AVAILABLE:
            brisque_dehazed = calculate_brisque_opencv(str(dehazed_file))
        else:
            brisque_dehazed = None

        entropy_dehazed = calculate_entropy(str(dehazed_file))
        gradient_dehazed = calculate_average_gradient(str(dehazed_file))
        saturation_dehazed = calculate_saturation(str(dehazed_file))

        # Print results
        if niqe_foggy is not None and niqe_dehazed is not None:
            print(f'  NIQE: {niqe_foggy:.4f} -> {niqe_dehazed:.4f} (Δ: {niqe_foggy - niqe_dehazed:.4f})')
        if brisque_foggy is not None and brisque_dehazed is not None:
            print(f'  BRISQUE: {brisque_foggy:.4f} -> {brisque_dehazed:.4f} (Δ: {brisque_foggy - brisque_dehazed:.4f})')
        if entropy_foggy is not None and entropy_dehazed is not None:
            print(f'  Entropy: {entropy_foggy:.4f} -> {entropy_dehazed:.4f} (Δ: {entropy_dehazed - entropy_foggy:.4f})')
        if gradient_foggy is not None and gradient_dehazed is not None:
            print(f'  Avg Gradient: {gradient_foggy:.4f} -> {gradient_dehazed:.4f} (Δ: {gradient_dehazed - gradient_foggy:.4f})')
        if saturation_foggy is not None and saturation_dehazed is not None:
            print(f'  Saturation: {saturation_foggy:.4f} -> {saturation_dehazed:.4f} (Δ: {saturation_dehazed - saturation_foggy:.4f})')

        # Store results
        results.append({
            'filename': foggy_name,
            'niqe_foggy': niqe_foggy,
            'niqe_dehazed': niqe_dehazed,
            'brisque_foggy': brisque_foggy,
            'brisque_dehazed': brisque_dehazed,
            'entropy_foggy': entropy_foggy,
            'entropy_dehazed': entropy_dehazed,
            'gradient_foggy': gradient_foggy,
            'gradient_dehazed': gradient_dehazed,
            'saturation_foggy': saturation_foggy,
            'saturation_dehazed': saturation_dehazed,
        })

        if niqe_foggy is not None:
            niqe_foggy_list.append(niqe_foggy)
            niqe_dehazed_list.append(niqe_dehazed)
        if brisque_foggy is not None:
            brisque_foggy_list.append(brisque_foggy)
            brisque_dehazed_list.append(brisque_dehazed)
        if entropy_foggy is not None:
            entropy_foggy_list.append(entropy_foggy)
            entropy_dehazed_list.append(entropy_dehazed)
        if gradient_foggy is not None:
            gradient_foggy_list.append(gradient_foggy)
            gradient_dehazed_list.append(gradient_dehazed)
        if saturation_foggy is not None:
            saturation_foggy_list.append(saturation_foggy)
            saturation_dehazed_list.append(saturation_dehazed)

        valid_count += 1
        print()

    # Summary statistics
    print(f'Summary Statistics ({valid_count} images)')

    if len(niqe_foggy_list) > 0:
        print(f'  Foggy:    {np.mean(niqe_foggy_list):.4f} ± {np.std(niqe_foggy_list):.4f}')
        print(f'  Dehazed:  {np.mean(niqe_dehazed_list):.4f} ± {np.std(niqe_dehazed_list):.4f}')
        print(f'  Average Improvement: {np.mean(niqe_foggy_list) - np.mean(niqe_dehazed_list):.4f}')

    if len(brisque_foggy_list) > 0:
        print(f'  Foggy:    {np.mean(brisque_foggy_list):.4f} ± {np.std(brisque_foggy_list):.4f}')
        print(f'  Dehazed:  {np.mean(brisque_dehazed_list):.4f} ± {np.std(brisque_dehazed_list):.4f}')
        print(f'  Average Improvement: {np.mean(brisque_foggy_list) - np.mean(brisque_dehazed_list):.4f}')

    if len(entropy_foggy_list) > 0:
        print(f'  Foggy:    {np.mean(entropy_foggy_list):.4f} ± {np.std(entropy_foggy_list):.4f}')
        print(f'  Dehazed:  {np.mean(entropy_dehazed_list):.4f} ± {np.std(entropy_dehazed_list):.4f}')
        print(f'  Average Improvement: {np.mean(entropy_dehazed_list) - np.mean(entropy_foggy_list):.4f}')

    if len(gradient_foggy_list) > 0:
        print(f'  Foggy:    {np.mean(gradient_foggy_list):.4f} ± {np.std(gradient_foggy_list):.4f}')
        print(f'  Dehazed:  {np.mean(gradient_dehazed_list):.4f} ± {np.std(gradient_dehazed_list):.4f}')
        print(f'  Average Improvement: {np.mean(gradient_dehazed_list) - np.mean(gradient_foggy_list):.4f}')

    if len(saturation_foggy_list) > 0:
        print(f'  Foggy:    {np.mean(saturation_foggy_list):.4f} ± {np.std(saturation_foggy_list):.4f}')
        print(f'  Dehazed:  {np.mean(saturation_dehazed_list):.4f} ± {np.std(saturation_dehazed_list):.4f}')
        print(f'  Average Improvement: {np.mean(saturation_dehazed_list) - np.mean(saturation_foggy_list):.4f}')

    print(f'Saving results to: {args.output}')

    with open(args.output, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'niqe_foggy', 'niqe_dehazed', 'niqe_improvement',
                      'brisque_foggy', 'brisque_dehazed', 'brisque_improvement',
                      'entropy_foggy', 'entropy_dehazed', 'entropy_improvement',
                      'gradient_foggy', 'gradient_dehazed', 'gradient_improvement',
                      'saturation_foggy', 'saturation_dehazed', 'saturation_improvement']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow({
                'filename': r['filename'],
                'niqe_foggy': f"{r['niqe_foggy']:.4f}" if r['niqe_foggy'] is not None else 'N/A',
                'niqe_dehazed': f"{r['niqe_dehazed']:.4f}" if r['niqe_dehazed'] is not None else 'N/A',
                'niqe_improvement': f"{r['niqe_foggy'] - r['niqe_dehazed']:.4f}" if r['niqe_foggy'] is not None else 'N/A',
                'brisque_foggy': f"{r['brisque_foggy']:.4f}" if r['brisque_foggy'] is not None else 'N/A',
                'brisque_dehazed': f"{r['brisque_dehazed']:.4f}" if r['brisque_dehazed'] is not None else 'N/A',
                'brisque_improvement': f"{r['brisque_foggy'] - r['brisque_dehazed']:.4f}" if r['brisque_foggy'] is not None else 'N/A',
                'entropy_foggy': f"{r['entropy_foggy']:.4f}" if r['entropy_foggy'] is not None else 'N/A',
                'entropy_dehazed': f"{r['entropy_dehazed']:.4f}" if r['entropy_dehazed'] is not None else 'N/A',
                'entropy_improvement': f"{r['entropy_dehazed'] - r['entropy_foggy']:.4f}" if r['entropy_foggy'] is not None else 'N/A',
                'gradient_foggy': f"{r['gradient_foggy']:.4f}" if r['gradient_foggy'] is not None else 'N/A',
                'gradient_dehazed': f"{r['gradient_dehazed']:.4f}" if r['gradient_dehazed'] is not None else 'N/A',
                'gradient_improvement': f"{r['gradient_dehazed'] - r['gradient_foggy']:.4f}" if r['gradient_foggy'] is not None else 'N/A',
                'saturation_foggy': f"{r['saturation_foggy']:.4f}" if r['saturation_foggy'] is not None else 'N/A',
                'saturation_dehazed': f"{r['saturation_dehazed']:.4f}" if r['saturation_dehazed'] is not None else 'N/A',
                'saturation_improvement': f"{r['saturation_dehazed'] - r['saturation_foggy']:.4f}" if r['saturation_foggy'] is not None else 'N/A',
            })

    print(f'Done!')
    print('='*70)


if __name__ == '__main__':
    main()
