import os
import sys
import argparse
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# Add CycleGAN path to system
CYCLEGAN_PATH = Path(__file__).parent.parent / "dehazing_with_GAN/nighttime_dehaze/pytorch-CycleGAN-and-pix2pix"
sys.path.insert(0, str(CYCLEGAN_PATH))

from models import networks


class CycleGANInference:
    """CycleGAN inference wrapper for nighttime dehazing."""

    def __init__(self, checkpoint_dir, epoch='latest', device='cuda'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.epoch = epoch
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print(f"Loading model from: {self.checkpoint_dir}")
        print(f"Using device: {self.device}")

        # Load model
        self.netG_A = self._load_generator()
        self.netG_A.eval()

        print("Model loaded successfully!")

    def _load_generator(self):
        """Load the generator network (G_A: hazy -> clean)."""
        # Create generator architecture (ResNet 9 blocks)
        netG = networks.ResnetGenerator(
            input_nc=3,
            output_nc=3,
            ngf=64,
            norm_layer=networks.get_norm_layer(norm_type='instance'),
            use_dropout=False,
            n_blocks=9
        )

        # Load weights
        weight_file = self.checkpoint_dir / f"{self.epoch}_net_G_A.pth"

        if not weight_file.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {weight_file}\n"
                f"Available checkpoints in {self.checkpoint_dir}:\n" +
                "\n".join([str(p.name) for p in self.checkpoint_dir.glob("*_net_G_A.pth")])
            )

        print(f"Loading weights: {weight_file.name}")
        state_dict = torch.load(weight_file, map_location=self.device)
        netG.load_state_dict(state_dict)
        netG.to(self.device)

        return netG

    def get_transform(self, size=512):
        """Get image transformation."""
        transform_list = [
            transforms.Resize((size, size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        return transforms.Compose(transform_list)

    def denormalize(self, tensor):
        """Convert from [-1, 1] to [0, 1]."""
        return (tensor + 1.0) / 2.0

    def process_image(self, image_path, output_path, size=512):
        """
        Process a single image.

        Args:
            image_path: Input hazy image path
            output_path: Output dehazed image path
            size: Image size (default 512)
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        original_size = img.size

        # Transform
        transform = self.get_transform(size)
        img_tensor = transform(img).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            dehazed_tensor = self.netG_A(img_tensor)

        # Denormalize and convert to PIL
        dehazed_tensor = self.denormalize(dehazed_tensor)
        dehazed_tensor = torch.clamp(dehazed_tensor, 0, 1)

        # Convert to PIL image
        dehazed_img = transforms.ToPILImage()(dehazed_tensor.squeeze(0).cpu())

        if original_size != (size, size):
            dehazed_img = dehazed_img.resize(original_size, Image.BICUBIC)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dehazed_img.save(output_path)

    def process_directory(self, input_dir, output_dir, size=512):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f'*{ext}'))
            image_files.extend(input_dir.glob(f'*{ext.upper()}'))
        image_files = sorted(list(set(image_files)))

        if len(image_files) == 0:
            print(f"No images found in {input_dir}")
            return
        print(f"Input:  {input_dir}")
        print(f"Output: {output_dir}")

        # Process each image
        for img_path in tqdm(image_files, desc="Dehazing"):
            output_path = output_dir / f"{img_path.stem}_dehazed{img_path.suffix}"
            try:
                self.process_image(img_path, output_path, size)
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")

        print(f"Results saved to: {output_dir}

def main():
    parser = argparse.ArgumentParser(description='CycleGAN Nighttime Dehazing Inference')

    # Paths
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory with hazy images')
    parser.add_argument('--output_dir', type=str, default='./results/cyclegan_output',
                        help='Output directory for dehazed images')
    parser.add_argument('--checkpoint_path', type=str,
                        default='../dehazing_with_GAN/nighttime_dehaze/pytorch-CycleGAN-and-pix2pix/checkpoints/nighttime_dehaze_512_combined',
                        help='Path to checkpoint directory')

    # Model
    parser.add_argument('--epoch', type=str, default='latest',
                        help='Which epoch to load (e.g., "latest", "200", "150")')
    parser.add_argument('--size', type=int, default=512,
                        help='Image size for processing (default: 512)')

    # Device
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use (-1 for CPU)')

    args = parser.parse_args()

    # Setup device
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = f'cuda:{args.gpu_id}'
        torch.cuda.set_device(args.gpu_id)
    else:
        device = 'cpu'

    dehazer = CycleGANInference(
        checkpoint_dir=args.checkpoint_path,
        epoch=args.epoch,
        device=device
    )

    dehazer.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        size=args.size
    )


if __name__ == '__main__':
    main()
