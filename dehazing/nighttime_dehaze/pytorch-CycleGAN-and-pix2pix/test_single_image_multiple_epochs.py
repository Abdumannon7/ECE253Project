"""
Test a single image with multiple epoch checkpoints.
"""

import os
import sys
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
from PIL import Image
import torchvision.transforms as transforms
import argparse

def test_single_image(image_path, model_name, epochs, gpu_id=1):
    """
    Test a single image with multiple epoch checkpoints.

    Args:
        image_path: Path to input foggy image
        model_name: Name of the model (e.g., 'internet_dwt_combined')
        epochs: List of epoch numbers to test
        gpu_id: GPU ID to use
    """
    # Create output directory
    output_dir = f'./results/{model_name}/single_image_test'
    os.makedirs(output_dir, exist_ok=True)

    print('='*70)
    print(f'Testing image: {image_path}')
    print(f'Model: {model_name}')
    print(f'Epochs: {epochs}')
    print('='*70)

    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')

    # Define transforms (same as CycleGAN preprocessing)
    transform_list = []
    transform_list.append(transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform_list)

    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Test with each epoch
    for epoch in epochs:
        print(f'\n[Epoch {epoch}] Processing...')

        # Parse options for this epoch
        # Modify sys.argv to pass the right arguments to TestOptions
        original_argv = sys.argv.copy()
        sys.argv = [
            sys.argv[0],
            '--dataroot', './datasets/internet_nighttime',  # Dummy dataroot
            '--name', model_name,
            '--model', 'cycle_gan',
            '--epoch', str(epoch),
            '--gpu_ids', str(gpu_id),
            '--num_test', '1',
            '--serial_batches',
            '--no_flip'
        ]

        opt = TestOptions().parse()
        sys.argv = original_argv  # Restore original argv

        opt.num_threads = 0
        opt.batch_size = 1

        # Create model
        model = create_model(opt)
        model.setup(opt)

        if opt.eval:
            model.eval()

        # Prepare data (CycleGAN expects both A and B, use A as dummy B for testing)
        data = {
            'A': img_tensor,
            'B': img_tensor,  # Dummy B (not used during test)
            'A_paths': [image_path],
            'B_paths': [image_path]
        }

        # Run inference
        model.set_input(data)
        model.test()

        # Get results
        visuals = model.get_current_visuals()

        # Save results
        for label, image in visuals.items():
            image_numpy = image[0].cpu().float().numpy()
            image_numpy = (image_numpy + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1]
            image_numpy = image_numpy.transpose(1, 2, 0) * 255.0  # CHW to HWC and scale to [0, 255]
            image_numpy = image_numpy.astype('uint8')

            output_path = os.path.join(output_dir, f'epoch{epoch}_{label}.png')
            Image.fromarray(image_numpy).save(output_path)
            print(f'  Saved: {output_path}')

        # Clean up
        del model
        torch.cuda.empty_cache()

    print(f'\n{"="*70}')
    print(f'All results saved to: {output_dir}')
    print('='*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test single image with multiple epochs')
    parser.add_argument('--image', type=str, required=True, help='Path to input foggy image')
    parser.add_argument('--name', type=str, default='internet_dwt_combined', help='Model name')
    parser.add_argument('--epochs', type=str, default='50,100,150,195', help='Comma-separated epoch numbers')
    parser.add_argument('--gpu_ids', type=str, default='1', help='GPU ID')

    args = parser.parse_args()

    # Parse epochs
    epochs = [int(e) for e in args.epochs.split(',')]

    # Run test
    test_single_image(args.image, args.name, epochs, int(args.gpu_ids))
