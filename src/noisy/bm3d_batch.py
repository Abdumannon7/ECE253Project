import argparse
import os
import cv2
import numpy as np
from bm3d import bm3d_rgb
from tqdm import tqdm


def add_gaussian_noise(img, noise_std):
    noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def bm3d_denoise(noisy_bgr, noise_std):
    # BGR → RGB
    noisy_rgb = cv2.cvtColor(noisy_bgr, cv2.COLOR_BGR2RGB)

    # uint8 → float32 [0,1]
    noisy_rgb_f = noisy_rgb.astype(np.float32) / 255.0

    # Normalize sigma for BM3D
    sigma_psd = noise_std / 255.0

    denoised_rgb_f = bm3d_rgb(noisy_rgb_f, sigma_psd=sigma_psd/2)

    # Back to uint8 BGR
    denoised_rgb = (denoised_rgb_f * 255).clip(0, 255).astype(np.uint8)
    denoised_bgr = cv2.cvtColor(denoised_rgb, cv2.COLOR_RGB2BGR)

    return denoised_bgr


def main(args):
    image_paths = [
        os.path.join(args.images, f)
        for f in os.listdir(args.images)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_paths:
        raise RuntimeError("No images found in directory")

    psnr_noisy_list = []
    psnr_denoised_list = []

    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        noisy_img = add_gaussian_noise(img, args.noise_std)
        denoised_img = bm3d_denoise(noisy_img, args.noise_std)

        psnr_noisy = cv2.PSNR(img, noisy_img)
        psnr_denoised = cv2.PSNR(img, denoised_img)

        psnr_noisy_list.append(psnr_noisy)
        psnr_denoised_list.append(psnr_denoised)

        print(
            f"{os.path.basename(img_path)} | "
            f"PSNR noisy: {psnr_noisy:.2f} dB | "
            f"PSNR BM3D: {psnr_denoised:.2f} dB"
        )

        if args.save:
            os.makedirs(args.save, exist_ok=True)
            out_path = os.path.join(args.save, os.path.basename(img_path))
            cv2.imwrite(out_path, denoised_img)

    print("\n=== Averages over dataset ===")
    print(f"Average PSNR (noisy): {np.mean(psnr_noisy_list):.2f} dB")
    print(f"Average PSNR (BM3D):  {np.mean(psnr_denoised_list):.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, required=True,
                        help="Path to image directory")
    parser.add_argument("--noise-std", type=float, default=30,
                        help="Gaussian noise sigma (0–255)")
    parser.add_argument("--save", type=str, default=None,
                        help="Optional directory to save denoised images")

    args = parser.parse_args()
    main(args)
