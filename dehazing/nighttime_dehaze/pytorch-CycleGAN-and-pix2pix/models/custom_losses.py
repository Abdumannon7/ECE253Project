"""
Custom loss functions for nighttime dehazing with glow suppression.
These losses are used in addition to the standard CycleGAN losses.

Reference: Enhancing Visibility in Nighttime Haze Images Using Guided APSF
and Gradient Adaptive Convolution (ACMMM 2023)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add EDGE folder to path for PIDINet import
edge_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'EDGE')
if edge_path not in sys.path:
    sys.path.insert(0, edge_path)


# ==================== DWT Implementation ====================
class DWTForward(nn.Module):
    """
    2D Discrete Wavelet Transform (DWT) - Haar wavelet.
    Decomposes image into 4 subbands: LL, LH, HL, HH
    """
    def __init__(self):
        super(DWTForward, self).__init__()
        # Haar wavelet filters
        self.register_buffer('ll_filter', torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32))
        self.register_buffer('lh_filter', torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32))
        self.register_buffer('hl_filter', torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32))
        self.register_buffer('hh_filter', torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32))

    def forward(self, x):
        """
        Args:
            x: Input image [B, C, H, W]
        Returns:
            LL: Low-frequency approximation [B, C, H/2, W/2]
            LH: Horizontal details [B, C, H/2, W/2]
            HL: Vertical details [B, C, H/2, W/2]
            HH: Diagonal details [B, C, H/2, W/2]
        """
        B, C, H, W = x.shape

        # Prepare filters for all channels
        ll = self.ll_filter.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        lh = self.lh_filter.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        hl = self.hl_filter.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        hh = self.hh_filter.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)

        # Apply convolution with stride 2 for downsampling
        LL = F.conv2d(x, ll, stride=2, groups=C)
        LH = F.conv2d(x, lh, stride=2, groups=C)
        HL = F.conv2d(x, hl, stride=2, groups=C)
        HH = F.conv2d(x, hh, stride=2, groups=C)

        return LL, LH, HL, HH


class DWTInverse(nn.Module):
    """
    2D Inverse Discrete Wavelet Transform (IDWT) - Haar wavelet.
    Reconstructs image from 4 subbands.
    """
    def __init__(self):
        super(DWTInverse, self).__init__()

    def forward(self, LL, LH, HL, HH):
        """
        Args:
            LL, LH, HL, HH: Wavelet subbands [B, C, H/2, W/2]
        Returns:
            Reconstructed image [B, C, H, W]
        """
        B, C, H, W = LL.shape

        # Upsample each subband
        LL_up = F.interpolate(LL, scale_factor=2, mode='nearest')
        LH_up = F.interpolate(LH, scale_factor=2, mode='nearest')
        HL_up = F.interpolate(HL, scale_factor=2, mode='nearest')
        HH_up = F.interpolate(HH, scale_factor=2, mode='nearest')

        # Reconstruct
        recon = LL_up + LH_up + HL_up + HH_up

        return recon


class LightSourceConsistencyLoss(nn.Module):
    """
    Light source consistency loss as described in the paper (Eq. 2 and Algorithm 1).
    """
    def __init__(self, threshold=0.8, use_matting=True, matting_eps=1e-5):
        super(LightSourceConsistencyLoss, self).__init__()
        self.threshold = threshold
        self.use_matting = use_matting
        self.matting_eps = matting_eps
        self.criterion = nn.L1Loss()

    def generate_light_source_mask(self, input_img):
        """
        Algorithm 1: Light Source Map Detection
        """
        # Convert from [-1, 1] to [0, 1] if needed
        if input_img.min() < 0:
            img = (input_img + 1) / 2.0
        else:
            img = input_img

        # Step 1: Generate initial light source mask by thresholding
        # M̂_{i,j} = 1 if max_c∈{r,g,b}(I^c_{i,j}) > 0.8, else 0
        max_intensity = torch.max(img, dim=1, keepdim=True)[0]  # [B, 1, H, W]
        initial_mask = (max_intensity > self.threshold).float()

        # Step 2: Refine mask using alpha matting (Laplacian matting)
        if self.use_matting:
            refined_mask = self.refine_mask_with_matting(img, initial_mask)
        else:
            refined_mask = initial_mask

        # Step 3: Calculate percentage of pixels in mask
        light_size = (refined_mask.sum(dim=[1, 2, 3]) / refined_mask.numel() * 100)

        # Step 4: Obtain light source image: L_s = I ⊙ M
        light_source = img * refined_mask

        return refined_mask, light_source, light_size

    def refine_mask_with_matting(self, img, initial_mask):
        """
        Refine mask using alpha matting (Laplacian matting approach).
        """
        B, C, H, W = img.shape

        # For differentiability and efficiency, we use guided filter as approximation
        # to alpha matting
        refined_masks = []

        for b in range(B):
            # Use guided filter to refine the mask
            # This is a differentiable approximation to Laplacian matting
            mask_b = initial_mask[b, 0]  # [H, W]
            img_b = img[b]  # [C, H, W]

            # Use grayscale guide image
            guide = img_b.mean(dim=0, keepdim=True)  # [1, H, W]

            # Apply guided filter
            refined = self.guided_filter(guide, mask_b.unsqueeze(0), radius=8, eps=self.matting_eps)
            refined_masks.append(refined)

        refined_mask = torch.stack(refined_masks, dim=0)  # [B, 1, H, W]

        # Clip to [0, 1] range
        refined_mask = torch.clamp(refined_mask, 0, 1)

        return refined_mask

    def guided_filter(self, guide, src, radius=8, eps=1e-5):
        """
        Differentiable guided filter implementation.
        """
        # Box filter (mean filter)
        def box_filter(x, r):
            return F.avg_pool2d(x.unsqueeze(0), kernel_size=2*r+1, stride=1, padding=r).squeeze(0)

        mean_I = box_filter(guide, radius)
        mean_p = box_filter(src, radius)
        mean_Ip = box_filter(guide * src, radius)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = box_filter(guide * guide, radius)
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = box_filter(a, radius)
        mean_b = box_filter(b, radius)

        output = mean_a * guide + mean_b

        return output

    def forward(self, output, input):
        """
        Compute light source consistency loss (Eq. 2):
        L_ls = |O_c ⊙ M - L_s|_1
        """
        # Generate light source mask and extract light sources
        mask, light_source, light_size = self.generate_light_source_mask(input)

        # Convert output to [0, 1] range if needed
        if output.min() < 0:
            output_01 = (output + 1) / 2.0
        else:
            output_01 = output

        # Compute L_ls = |O_c ⊙ M - L_s|_1
        # This ensures output preserves light source regions from input
        output_light_region = output_01 * mask
        loss = self.criterion(output_light_region, light_source)

        return loss


class GradientLoss(nn.Module):
    """
    Gradient loss using PIDINet (Pixel Difference Network) for edge detection.
    """
    def __init__(self, pidinet_weights_path=None, freeze_pidinet=True):
        super(GradientLoss, self).__init__()
        self.criterion = nn.L1Loss()

        # Try to import PIDINet (now copied to local models folder)
        try:
            # Import from the same models directory
            from . import pidinet
            from .convert_pidinet import convert_pidinet

            # Create PIDINet model
            class Args:
                sa = True  # Use CSAM (Compact Spatial Attention Module)
                dil = True  # Use CDCM (Compact Dilation Convolution Module)
                config = 'carv4'

            args = Args()
            self.edge_detector = pidinet.pidinet(args)

            # Load pre-trained weights if provided
            if pidinet_weights_path is not None and os.path.exists(pidinet_weights_path):
                checkpoint = torch.load(pidinet_weights_path, map_location='cpu', weights_only=False)
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # Load weights directly without conversion (conversion changes architecture)
                self.edge_detector.load_state_dict(state_dict, strict=False)
                print(f"[GradientLoss] Loaded PIDINet weights from {pidinet_weights_path}")
            else:
                print(f"PIDINet weights not found at {pidinet_weights_path}")
                print("Using randomly initialized PIDINet (training from scratch)")

            # Freeze PIDINet weights if specified (typical for loss network)
            if freeze_pidinet:
                for param in self.edge_detector.parameters():
                    param.requires_grad = False
                self.edge_detector.eval()

            self.use_pidinet = True

        except Exception as e:
            print(f"[GradientLoss] Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            self.use_pidinet = False

    def extract_edges_pidinet(self, img):
        """
        Extract edge features using PIDINet.
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            # PIDINet expects RGB images
            # If input is normalized to [-1, 1], convert to [0, 1]
            if img.min() < 0:
                img_input = (img + 1) / 2.0
            else:
                img_input = img

            # PIDINet forward pass returns list of edge maps at different scales
            edge_maps = self.edge_detector(img_input)

            # Use the final (finest) edge map
            if isinstance(edge_maps, list):
                edge_map = edge_maps[-1]
            else:
                edge_map = edge_maps

            return edge_map

    def extract_edges_sobel(self, img):
        """
        Fallback: Extract edges using Sobel filters.
        """
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=img.dtype, device=img.device).view(1, 1, 3, 3)

        # Convert to grayscale
        if img.shape[1] == 3:
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img

        # Compute gradients
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)

        # Gradient magnitude
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        return grad

    def forward(self, output, target):
        """
        Compute gradient loss as per paper:
        L_g = |D(O_c) - D(I_h)|_1
        """
        if self.use_pidinet:
            edges_output = self.extract_edges_pidinet(output)
            edges_target = self.extract_edges_pidinet(target)
        else:
            edges_output = self.extract_edges_sobel(output)
            edges_target = self.extract_edges_sobel(target)

        loss = self.criterion(edges_output, edges_target)

        return loss


class BilateralKernelLoss(nn.Module):
    """
    Bilateral kernel loss for texture capture using adaptive convolution (Eq. 7 & 8).
    Adaptive Convolution (Eq. 7):
        v'_i = Σ_{j∈Ω(i)} K(f_i, f_j) * w[p_i - p_j] * v_j

    where:
        K(f_i, f_j) = exp(-1/(2α1) ||f_i - f_j||^2)  [range/color kernel]
        w[p_i - p_j] = exp(-1/(2α2) ||p_i - p_j||^2) [spatial kernel]
        f = (r, g, b) color features

    Bilateral Kernel Loss (Eq. 8):
        L_k = |K(O_c) - K(I_h)|_1
    """
    def __init__(self, kernel_size=5, alpha1=0.1, alpha2=1.5):
        """
        Args:
            kernel_size: Size of the bilateral filter window (default: 5)
            alpha1: Range kernel parameter (controls color similarity)
            alpha2: Spatial kernel parameter (controls spatial distance)
        """
        super(BilateralKernelLoss, self).__init__()
        self.kernel_size = kernel_size
        self.alpha1 = alpha1  # For range kernel K(f_i, f_j)
        self.alpha2 = alpha2  # For spatial kernel w[p_i - p_j]
        self.criterion = nn.L1Loss()
        self.pad = kernel_size // 2

    def apply_bilateral_filter(self, img):
        """
        Apply bilateral filtering using adaptive convolution (Eq. 7).

        The bilateral filter combines:
        1. Spatial kernel: w[p_i - p_j] = exp(-||p_i - p_j||^2 / (2*alpha2))
        2. Range kernel: K(f_i, f_j) = exp(-||f_i - f_j||^2 / (2*alpha1))
        """
        B, C, H, W = img.shape

        # Convert from [-1, 1] to [0, 1] if needed for color features
        if img.min() < 0:
            img_01 = (img + 1) / 2.0
        else:
            img_01 = img
        img_pad = F.pad(img_01, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        coords = torch.arange(-self.pad, self.pad + 1, dtype=img.dtype, device=img.device)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
        spatial_dist_sq = grid_x ** 2 + grid_y ** 2
        spatial_kernel = torch.exp(-spatial_dist_sq / (2 * self.alpha2))
        filtered_img = []

        for b in range(B):
            filtered_channels = []

            for c in range(C):
                # Extract patches for this channel
                channel_pad = img_pad[b, c:c+1, :, :]  # [1, H+2p, W+2p]

                # Unfold to get all neighborhoods [H*W, k*k]
                patches = F.unfold(channel_pad.unsqueeze(0),
                                 kernel_size=self.kernel_size,
                                 padding=0).squeeze(0)  # [k*k, H*W]
                patches = patches.view(self.kernel_size * self.kernel_size, H, W)  # [k*k, H, W]

                # Get center pixel values for range kernel computation
                center_vals = img_01[b, c, :, :].unsqueeze(0)  # [1, H, W]

                # Compute range kernel K(f_i, f_j) for each pixel
                # Color difference: ||f_i - f_j||^2
                color_diff_sq = (patches - center_vals) ** 2  # [k*k, H, W]

                # Range kernel: K(f_i, f_j) = exp(-1/(2*alpha1) * ||f_i - f_j||^2)
                range_kernel = torch.exp(-color_diff_sq / (2 * self.alpha1))  # [k*k, H, W]

                # Combined bilateral kernel: K(f_i, f_j) * w[p_i - p_j]
                bilateral_weights = range_kernel * spatial_kernel.view(-1, 1, 1)  # [k*k, H, W]

                # Normalize weights
                weight_sum = bilateral_weights.sum(dim=0, keepdim=True) + 1e-8  # [1, H, W]
                bilateral_weights = bilateral_weights / weight_sum  # [k*k, H, W]

                # Apply bilateral filtering: v'_i = Σ K(f_i, f_j) * w[p_i - p_j] * v_j
                filtered_channel = (bilateral_weights * patches).sum(dim=0)  # [H, W]

                filtered_channels.append(filtered_channel)

            # Stack channels
            filtered_img.append(torch.stack(filtered_channels, dim=0))  # [C, H, W]

        # Stack batch
        filtered_img = torch.stack(filtered_img, dim=0)  # [B, C, H, W]

        return filtered_img

    def forward(self, output, target):
        """
        Compute bilateral kernel loss (Eq. 8):
        L_k = |K(O_c) - K(I_h)|_1
        This enforces consistency between input and output after bilateral filtering,
        helping to extract high-frequency texture details that are less affected by
        haze and glow.
        """
        # Apply bilateral filtering to both images
        # K(O_c): bilateral filtered output
        output_filtered = self.apply_bilateral_filter(output)

        # K(I_h): bilateral filtered input
        target_filtered = self.apply_bilateral_filter(target)

        # Compute L1 loss: L_k = |K(O_c) - K(I_h)|_1
        loss = self.criterion(output_filtered, target_filtered)

        return loss


class TotalVariationLoss(nn.Module):
    """
    Total Variation (TV) Loss for reducing checkerboard artifacts and promoting smoothness.
    """
    def __init__(self, weight=1.0):
        super(TotalVariationLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        """
        Compute total variation loss.
        """
        h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        v_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        tv_loss = (h_tv.mean() + v_tv.mean()) * self.weight
        return tv_loss


class DWTDetailLoss(nn.Module):
    """
    DWT-based Detail Preservation Loss.

    Uses Discrete Wavelet Transform to decompose images into multi-scale frequency
    components and enforces consistency in high-frequency details (LH, HL, HH).
    """
    def __init__(self, use_ll=False, ll_weight=0.1, num_levels=1):
        super(DWTDetailLoss, self).__init__()
        self.dwt = DWTForward()
        self.use_ll = use_ll
        self.ll_weight = ll_weight
        self.num_levels = num_levels
        self.criterion = nn.L1Loss()

    def forward(self, output, target):
        """
        Compute DWT-based detail preservation loss.
        """
        # Convert from [-1, 1] to [0, 1] if needed
        if output.min() < 0:
            output = (output + 1) / 2.0
        if target.min() < 0:
            target = (target + 1) / 2.0

        total_loss = 0.0

        # Level 1 DWT decomposition
        LL_out, LH_out, HL_out, HH_out = self.dwt(output)
        LL_tar, LH_tar, HL_tar, HH_tar = self.dwt(target)

        # High-frequency detail losses (preserve fine details)
        loss_lh = self.criterion(LH_out, LH_tar)  # Horizontal details
        loss_hl = self.criterion(HL_out, HL_tar)  # Vertical details
        loss_hh = self.criterion(HH_out, HH_tar)  # Diagonal details

        total_loss += loss_lh + loss_hl + loss_hh

        # Optional: Low-frequency loss (preserve overall structure)
        if self.use_ll:
            loss_ll = self.criterion(LL_out, LL_tar)
            total_loss += self.ll_weight * loss_ll

        # Optional: Level 2 DWT for multi-scale details
        if self.num_levels >= 2:
            LL2_out, LH2_out, HL2_out, HH2_out = self.dwt(LL_out)
            LL2_tar, LH2_tar, HL2_tar, HH2_tar = self.dwt(LL_tar)

            loss_lh2 = self.criterion(LH2_out, LH2_tar)
            loss_hl2 = self.criterion(HL2_out, HL2_tar)
            loss_hh2 = self.criterion(HH2_out, HH2_tar)

            # Weight level 2 details less (coarser scale)
            total_loss += 0.5 * (loss_lh2 + loss_hl2 + loss_hh2)

        return total_loss


class CombinedDetailLoss(nn.Module):
    """
    Combined detail preservation loss using both DWT and gradient loss.
    """
    def __init__(self, gradient_loss, dwt_loss, lambda_dwt=1.0, lambda_grad=0.5):
        super(CombinedDetailLoss, self).__init__()
        self.gradient_loss = gradient_loss
        self.dwt_loss = dwt_loss
        self.lambda_dwt = lambda_dwt
        self.lambda_grad = lambda_grad

    def forward(self, output, target):
        loss_dwt = self.dwt_loss(output, target)
        loss_grad = self.gradient_loss(output, target)

        total_loss = self.lambda_dwt * loss_dwt + self.lambda_grad * loss_grad

        return total_loss, loss_dwt, loss_grad


class NighttimeDehazeLosses:
    """
    Combined loss functions for nighttime dehazing.
    Wraps all custom losses with their respective weights.
    """
    def __init__(self, device,
                 lambda_ls=1.0,       # Light source consistency weight
                 lambda_g=0.5,        # Gradient loss weight
                 lambda_k=5.0,        # Bilateral kernel loss weight
                 lambda_dwt=1.0,      # DWT detail loss weight
                 lambda_tv=0.1,       # Total variation loss weight (for artifact reduction)
                 pidinet_path=None,   # Path to PIDINet weights
                 use_dwt=False,       # Enable DWT loss
                 dwt_mode='replace',  # 'replace', 'combine', or 'add'
                 dwt_levels=1):       # Number of DWT decomposition levels
        self.device = device
        self.lambda_ls = lambda_ls
        self.lambda_g = lambda_g
        self.lambda_k = lambda_k
        self.lambda_dwt = lambda_dwt
        self.lambda_tv = lambda_tv
        self.use_dwt = use_dwt
        self.dwt_mode = dwt_mode

        # Set default PIDINet path if not provided
        if pidinet_path is None:
            # Try to find PIDINet weights in the EDGE folder
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            pidinet_path = os.path.join(current_dir, 'EDGE', 'trained_models', 'table5_pidinet.pth')

        # Initialize loss functions
        self.loss_ls = LightSourceConsistencyLoss().to(device)
        self.loss_k = BilateralKernelLoss().to(device)
        self.loss_tv = TotalVariationLoss(weight=1.0).to(device)  # TV loss for artifact reduction

        # Initialize gradient and/or DWT losses based on mode
        if use_dwt:
            self.loss_dwt = DWTDetailLoss(use_ll=False, num_levels=dwt_levels).to(device)

            if dwt_mode == 'replace':
                # Use DWT instead of gradient loss
                self.loss_g = None
            elif dwt_mode == 'combine':
                # Use combined DWT + gradient loss
                self.loss_g = GradientLoss(pidinet_weights_path=pidinet_path, freeze_pidinet=True).to(device)
            elif dwt_mode == 'add':
                # Use both separately
                self.loss_g = GradientLoss(pidinet_weights_path=pidinet_path, freeze_pidinet=True).to(device)
        else:
            # Original paper implementation
            self.loss_g = GradientLoss(pidinet_weights_path=pidinet_path, freeze_pidinet=True).to(device)
            self.loss_dwt = None

    def compute_losses(self, fake_B, real_A):
        """
        Compute all custom losses.
        """
        losses_dict = {}

        # Light source consistency loss
        loss_ls = self.loss_ls(fake_B, real_A) * self.lambda_ls
        losses_dict['loss_ls'] = loss_ls

        # Bilateral kernel loss
        loss_k = self.loss_k(fake_B, real_A) * self.lambda_k
        losses_dict['loss_k'] = loss_k

        # Total variation loss (for reducing checkerboard artifacts)
        loss_tv = self.loss_tv(fake_B) * self.lambda_tv
        losses_dict['loss_tv'] = loss_tv

        # Detail preservation losses (gradient and/or DWT)
        if self.use_dwt:
            if self.dwt_mode == 'replace':
                # Use DWT instead of gradient
                loss_dwt = self.loss_dwt(fake_B, real_A) * self.lambda_dwt
                losses_dict['loss_dwt'] = loss_dwt
                total_loss = loss_ls + loss_dwt + loss_k + loss_tv

            elif self.dwt_mode == 'combine':
                # Combined DWT + gradient with shared weight
                loss_g = self.loss_g(fake_B, real_A)
                loss_dwt = self.loss_dwt(fake_B, real_A)
                combined_detail = (self.lambda_g * loss_g + self.lambda_dwt * loss_dwt)
                losses_dict['loss_g'] = loss_g * self.lambda_g
                losses_dict['loss_dwt'] = loss_dwt * self.lambda_dwt
                total_loss = loss_ls + combined_detail + loss_k + loss_tv

            elif self.dwt_mode == 'add':
                # Both as separate losses
                loss_g = self.loss_g(fake_B, real_A) * self.lambda_g
                loss_dwt = self.loss_dwt(fake_B, real_A) * self.lambda_dwt
                losses_dict['loss_g'] = loss_g
                losses_dict['loss_dwt'] = loss_dwt
                total_loss = loss_ls + loss_g + loss_dwt + loss_k + loss_tv
        else:
            # Original paper implementation (gradient only)
            loss_g = self.loss_g(fake_B, real_A) * self.lambda_g
            losses_dict['loss_g'] = loss_g
            total_loss = loss_ls + loss_g + loss_k + loss_tv

        losses_dict['total_custom'] = total_loss

        return losses_dict
