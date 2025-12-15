"""
Calculate PSNR and SSIM for GTA5 test results.
Adapted from the paper's evaluation script.
"""

import os
import cv2
import numpy as np
import argparse


def calculate_psnr(img1, img2, test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio)."""
    assert img1.shape == img2.shape, f'Image shapes are different: {img1.shape}, {img2.shape}.'
    assert img1.shape[2] == 3
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, test_y_channel=False):
    """Calculate SSIM (structural similarity)."""
    assert img1.shape == img2.shape, f'Image shapes are different: {img1.shape}, {img2.shape}.'
    assert img1.shape[2] == 3
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


def to_y_channel(img):
    """Change to Y channel of YCbCr."""
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def _convert_input_type_range(img):
    """Convert the type and range of the input image."""
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError(f'The img type should be np.float32 or np.uint8, but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type."""
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'The dst_type should be np.float32 or np.uint8, but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image."""
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def main():
    parser = argparse.ArgumentParser(description='Calculate PSNR and SSIM for dehazing results')
    parser.add_argument('--results', type=str, required=True, help='Path to results folder (fake_B images)')
    parser.add_argument('--gt', type=str, required=True, help='Path to ground truth folder (clean images)')
    parser.add_argument('--test_y_channel', action='store_true', help='Test on Y channel (luminance) like the paper')

    args = parser.parse_args()

    # Get list of result images - ONLY fake_B (dehazed outputs)
    results_list = sorted([f for f in os.listdir(args.results) if f.endswith('_fake_B.png')])
    print(f'Found {len(results_list)} fake_B (dehazed) images')

    if len(results_list) == 0:
        return

    cumulative_psnr = 0
    cumulative_ssim = 0
    count = 0

    for i, img_name in enumerate(results_list):
        # Map fake_B to ground truth: remove _fake_B suffix
        # e.g., "123_fake_B.png" -> "123.png"
        gt_name = img_name.replace('_fake_B.png', '.png')

        result_path = os.path.join(args.results, img_name)
        gt_path = os.path.join(args.gt, gt_name)

        if not os.path.exists(gt_path):
            print(f'Warning: GT not found for {img_name} (looking for {gt_name}), skipping...')
            continue

        res = cv2.imread(result_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)

        if res is None or gt is None:
            print(f'Warning: Failed to load {img_name}, skipping...')
            continue

        # Resize result to match GT size if different
        if res.shape != gt.shape:
            print(f'Info: Resizing {img_name} from {res.shape[:2]} to {gt.shape[:2]} to match GT')
            res = cv2.resize(res, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)

        count += 1
        cur_psnr = calculate_psnr(res, gt, test_y_channel=args.test_y_channel)
        cur_ssim = calculate_ssim(res, gt, test_y_channel=args.test_y_channel)

        cumulative_psnr += cur_psnr
        cumulative_ssim += cur_ssim

        if count % 10 == 0:
            print(f'Processed {count} images - Avg PSNR: {cumulative_psnr/count:.4f}, Avg SSIM: {cumulative_ssim/count:.4f}')

    if count > 0:
        avg_psnr = cumulative_psnr / count
        avg_ssim = cumulative_ssim / count
        print(f'Final Results ({count} images):')
        print(f'  PSNR: {avg_psnr:.4f}')
        print(f'  SSIM: {avg_ssim:.4f}')
        print(f'\nPaper reference (GTA5): PSNR=30.5298, SSIM=0.9060')
    else:
        print('error: No valid image pairs found!')


if __name__ == '__main__':
    main()
