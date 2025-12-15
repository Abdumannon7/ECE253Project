"""
Improved Nighttime Dehazing with 3 Key Fixes:
1. Pre-denoise on Y channel before estimation
2. Guided filter on transmission/alpha map
3. Clamp transmission floor to prevent noise amplification

Based on pixel-wise alpha blending for nighttime images.
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path


def denoise_Y_with_BM3D(img_bgr, sigma=0.06):
    """
    Fix1: Pre-denoise the luminance (Y) channel using BM3D or fastNlMeans.
    """
    # Convert to YCrCb
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)

    # Try BM3D first (requires cv2.xphoto), fallback to fastNlMeans
    try:
        import cv2.xphoto as xphoto
        # BM3D denoising on Y channel
        # sigma in [0,255] range for uint8
        y_denoised = xphoto.bm3dDenoising(y, h=sigma*255)
    except:
        # Fallback to fastNlMeansDenoisingColored
        # h parameter: filter strength (recommended: 10-15 for night images)
        h = int(sigma * 150)  # Convert sigma to h range
        y_denoised = cv2.fastNlMeansDenoising(y, None, h=h, templateWindowSize=7, searchWindowSize=21)

    # Reconstruct BGR
    img_ycrcb_denoised = cv2.merge([y_denoised, cr, cb])
    img_denoised = cv2.cvtColor(img_ycrcb_denoised, cv2.COLOR_YCrCb2BGR)

    return img_denoised


def guided_filter(guide, src, radius=40, eps=2e-3):
    """
    Fix 2: Guided filter for edge-preserving smoothing of transmission map.
    """
    # OpenCV's ximgproc guided filter
    try:
        import cv2.ximgproc as ximgproc
        # Convert to uint8 for OpenCV guided filter
        guide_u8 = (guide * 255).astype(np.uint8)
        src_u8 = (src * 255).astype(np.uint8)

        filtered_u8 = ximgproc.guidedFilter(guide_u8, src_u8, radius=radius, eps=eps)
        filtered = filtered_u8.astype(np.float32) / 255.0
        return filtered
    except:
        # Fallback to bilateral filter
        src_u8 = (src * 255).astype(np.uint8)
        filtered_u8 = cv2.bilateralFilter(src_u8, d=2*radius+1, sigmaColor=75, sigmaSpace=75)
        filtered = filtered_u8.astype(np.float32) / 255.0
        return filtered


def estimate_atmospheric_light(img, percentile=0.001):
    """
    Estimate atmospheric light A using brightest pixels.
    """
    h, w, c = img.shape
    n_pixels = h * w
    n_bright = max(int(n_pixels * percentile), 1)

    # Get intensity
    intensity = np.mean(img, axis=2)

    # Find brightest pixels
    flat_idx = np.argpartition(intensity.ravel(), -n_bright)[-n_bright:]

    # Get atmospheric light as mean of brightest pixels
    A = np.mean(img.reshape(-1, c)[flat_idx], axis=0)

    return A


def estimate_transmission_nighttime(img, A, window_size=15):
    """
    Estimate transmission map for NIGHTTIME images (inverted approach).

    For nighttime: darker areas have MORE haze/glow, not less.
    We use a bright channel approach instead of dark channel.
    """
    intensity = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    A_intensity = np.mean(A)
    t = 1.0 - intensity / (A_intensity + 0.1)  # Inverse relationship
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    t = cv2.dilate(t, kernel)
    t = np.clip(t, 0.0, 1.0)
    t = t * 0.5 + 0.5  # Shift range to [0.5, 1.0]
    return t


def recover_scene_radiance(img, A, t, t_min=0.15, gamma=1.2):
    """
    Fix 3: Recover scene radiance with transmission floor clamping.
    For nighttime: use modified recovery + gamma correction.
    """
    # Clamp transmission to prevent noise amplification
    t_clamped = np.clip(np.maximum(t, t_min), 0.0, 1.0)
    # Standard dehazing recovery formula: J = (I - A)/t + A
    # Expand t to 3 channels and A to image shape
    t_3ch = t_clamped[:, :, np.newaxis]  # (H, W, 1)
    A_broadcast = A[np.newaxis, np.newaxis, :]  # (1, 1, 3)

    # J = (I - A) / t + A
    J = (img - A_broadcast) / t_3ch + A_broadcast

    # Clip to valid range
    J = np.clip(J, 0.0, 1.0)

    # Apply gamma correction to brighten
    if gamma != 1.0:
        J = np.power(J, 1.0 / gamma)

    return J


def nighttime_dehaze_improved(img_path, output_path=None,
                              denoise_sigma=0.06,
                              guided_radius=40, guided_eps=2e-3,
                              t_min=0.15,
                              gamma=1.2,
                              visualize=True):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {img_path}")

    h, w = img_bgr.shape[:2]
    img_denoised = denoise_Y_with_BM3D(img_bgr, sigma=denoise_sigma)
    img_01 = img_denoised.astype(np.float32) / 255.0
    A = estimate_atmospheric_light(img_01, percentile=0.001)
    print(f"A (BGR) = [{A[0]:.3f}, {A[1]:.3f}, {A[2]:.3f}]")
    t_raw = estimate_transmission_nighttime(img_01, A, window_size=15)
    print(f"t_raw range: [{t_raw.min():.3f}, {t_raw.max():.3f}]")
    guide = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t_smooth = guided_filter(guide, t_raw, radius=guided_radius, eps=guided_eps)
    print(f"t_smooth range: [{t_smooth.min():.3f}, {t_smooth.max():.3f}]")
    J_01 = recover_scene_radiance(img_01, A, t_smooth, t_min=t_min, gamma=gamma)
    J_bgr = (J_01 * 255).astype(np.uint8)
    if output_path:
        print(f"\n Saving result to: {output_path}")
        cv2.imwrite(output_path, J_bgr)

    return J_bgr


def main():
    parser = argparse.ArgumentParser(description='Improved Nighttime Dehazing')
    parser.add_argument('--input', type=str, required=True, help='Input hazy image path')
    parser.add_argument('--output', type=str, default=None, help='Output dehazed image path')
    parser.add_argument('--denoise_sigma', type=float, default=0.06, help='Denoising strength (0.05-0.08)')
    parser.add_argument('--guided_radius', type=int, default=40, help='Guided filter radius (20-60)')
    parser.add_argument('--guided_eps', type=float, default=2e-3, help='Guided filter epsilon (1e-3 to 5e-3)')
    parser.add_argument('--t_min', type=float, default=0.15, help='Transmission floor (0.15-0.20)')
    parser.add_argument('--gamma', type=float, default=1.2, help='Gamma correction for brightness (1.0-1.5)')

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_dehazed{input_path.suffix}")

    # Run dehazing
    nighttime_dehaze_improved(
        args.input,
        args.output,
        denoise_sigma=args.denoise_sigma,
        guided_radius=args.guided_radius,
        guided_eps=args.guided_eps,
        t_min=args.t_min,
        gamma=args.gamma,
        visualize=args.viz
    )


if __name__ == '__main__':
    main()
