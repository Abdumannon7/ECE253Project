"""
Test script to verify custom loss functions work correctly.
Run this before training to ensure the implementation is correct.
"""

import torch
import sys
sys.path.append('.')
from models.custom_losses import (
    LightSourceConsistencyLoss,
    GradientLoss,
    BilateralKernelLoss,
    NighttimeDehazeLosses
)


def test_light_source_loss():
    """Test light source consistency loss."""
    print("Testing Light Source Consistency Loss...")

    loss_fn = LightSourceConsistencyLoss(threshold=0.8)

    # Create dummy images
    batch_size = 2
    channels = 3
    height, width = 256, 256

    # Input with bright regions (simulating light sources)
    input_img = torch.rand(batch_size, channels, height, width)
    input_img[:, :, 100:150, 100:150] = 0.9  # Add bright region

    # Output that preserves light sources
    output_img = input_img.clone()

    # Compute loss
    loss = loss_fn(output_img, input_img)

    print(f"  Loss value (should be near 0): {loss.item():.6f}")
    assert loss.item() < 0.1, "Light source loss too high for identical images"

    # Test with different output
    output_img_different = torch.rand(batch_size, channels, height, width)
    loss_different = loss_fn(output_img_different, input_img)
    print(f"  Loss value (different output): {loss_different.item():.6f}")
    assert loss_different.item() > loss.item(), "Loss should be higher for different images"

    print("  ✓ Light Source Consistency Loss passed!\n")


def test_gradient_loss():
    """Test gradient loss."""
    print("Testing Gradient Loss...")

    loss_fn = GradientLoss()

    # Create images with edges
    batch_size = 2
    channels = 3
    height, width = 256, 256

    # Image with vertical edge
    img1 = torch.zeros(batch_size, channels, height, width)
    img1[:, :, :, :128] = 0.0
    img1[:, :, :, 128:] = 1.0

    # Same image
    img2 = img1.clone()

    # Compute loss
    loss = loss_fn(img2, img1)
    print(f"  Loss value (identical edges): {loss.item():.6f}")
    assert loss.item() < 0.01, "Gradient loss too high for identical images"

    # Different gradient
    img3 = torch.rand(batch_size, channels, height, width)
    loss_different = loss_fn(img3, img1)
    print(f"  Loss value (different gradients): {loss_different.item():.6f}")
    assert loss_different.item() > loss.item(), "Loss should be higher for different gradients"

    print("  ✓ Gradient Loss passed!\n")


def test_bilateral_kernel_loss():
    """Test bilateral kernel loss."""
    print("Testing Bilateral Kernel Loss...")

    loss_fn = BilateralKernelLoss(kernel_size=5, sigma_spatial=1.5, sigma_intensity=0.1)

    # Create smooth images
    batch_size = 2
    channels = 3
    height, width = 256, 256

    img1 = torch.rand(batch_size, channels, height, width)
    img2 = img1.clone()

    # Compute loss
    loss = loss_fn(img2, img1)
    print(f"  Loss value (identical images): {loss.item():.6f}")
    assert loss.item() < 0.01, "Bilateral kernel loss too high for identical images"

    # Different image
    img3 = torch.rand(batch_size, channels, height, width)
    loss_different = loss_fn(img3, img1)
    print(f"  Loss value (different images): {loss_different.item():.6f}")
    assert loss_different.item() > loss.item(), "Loss should be higher for different images"

    print("  ✓ Bilateral Kernel Loss passed!\n")


def test_combined_losses():
    """Test combined nighttime dehaze losses."""
    print("Testing Combined Nighttime Dehaze Losses...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")

    combined_losses = NighttimeDehazeLosses(
        device=device,
        lambda_ls=1.0,
        lambda_g=0.5,
        lambda_k=5.0
    )

    # Create dummy images
    batch_size = 2
    channels = 3
    height, width = 256, 256

    fake_B = torch.rand(batch_size, channels, height, width).to(device)
    real_A = torch.rand(batch_size, channels, height, width).to(device)

    # Compute all losses
    loss_dict = combined_losses.compute_losses(fake_B, real_A)

    print(f"  Light source loss: {loss_dict['loss_ls'].item():.6f}")
    print(f"  Gradient loss: {loss_dict['loss_g'].item():.6f}")
    print(f"  Bilateral kernel loss: {loss_dict['loss_k'].item():.6f}")
    print(f"  Total custom loss: {loss_dict['total_custom'].item():.6f}")

    # Check that losses are computed
    assert loss_dict['loss_ls'].item() > 0, "Light source loss should be positive"
    assert loss_dict['loss_g'].item() > 0, "Gradient loss should be positive"
    assert loss_dict['loss_k'].item() > 0, "Bilateral kernel loss should be positive"

    # Check weights are applied correctly
    expected_total = loss_dict['loss_ls'] + loss_dict['loss_g'] + loss_dict['loss_k']
    assert torch.allclose(loss_dict['total_custom'], expected_total, rtol=1e-5), \
        "Total loss should be sum of weighted losses"

    print("  ✓ Combined Losses passed!\n")


def test_backward_pass():
    """Test that gradients flow correctly."""
    print("Testing Backward Pass (Gradient Flow)...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    combined_losses = NighttimeDehazeLosses(
        device=device,
        lambda_ls=1.0,
        lambda_g=0.5,
        lambda_k=5.0
    )

    # Create dummy images with gradient tracking
    batch_size = 1
    channels = 3
    height, width = 64, 64  # Smaller for faster computation

    fake_B = torch.rand(batch_size, channels, height, width,
                       requires_grad=True, device=device)
    real_A = torch.rand(batch_size, channels, height, width, device=device)

    # Forward pass
    loss_dict = combined_losses.compute_losses(fake_B, real_A)
    total_loss = loss_dict['total_custom']

    # Backward pass
    total_loss.backward()

    # Check gradients exist
    assert fake_B.grad is not None, "Gradients should exist for fake_B"
    assert fake_B.grad.abs().sum() > 0, "Gradients should be non-zero"

    print(f"  Gradient norm: {fake_B.grad.norm().item():.6f}")
    print("  ✓ Backward Pass passed!\n")


if __name__ == "__main__":
    print("="*60)
    print("Testing Custom Loss Functions for Nighttime Dehazing")
    print("="*60 + "\n")

    try:
        test_light_source_loss()
        test_gradient_loss()
        test_bilateral_kernel_loss()
        test_combined_losses()
        test_backward_pass()

        print("="*60)
        print("✓ All tests passed successfully!")
        print("="*60)
        print("\nYou can now train the model with custom losses:")
        print("python train.py --use_custom_losses --dataroot ./datasets/nighttime_haze ...")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
