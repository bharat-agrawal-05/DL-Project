import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import torchvision.models as models
from PIL import Image
import time
import os
import glob
import random
import argparse
from torch.cuda.amp import GradScaler, autocast

# --- Configuration (Consolidated & Expanded) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# General
NUM_EPOCHS = 50
WATERMARK_DIM = 128  # Dimension of the float watermark vector
MODEL_SAVE_DIR = "trained_watermarking_models_float_wm" # New dir for float models
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Embedder (U-Net) Config
EMBEDDER_LR = 1e-4
EMBEDDER_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "unet_embedder_float_wm.pth")
N_CHANNELS_IN = 3
N_CHANNELS_OUT = 3
BILINEAR_UPSAMPLE = True
NORMALIZE_WATERMARK_SPATIAL = True
WATERMARK_BOTTLENECK_FIXED_RES = 16
DIVISOR_FOR_PADDING = 16

# Extractor (ViT) Config
EXTRACTOR_LR = 1e-4
EXTRACTOR_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "vit_extractor_float_wm.pth")
VIT_IMG_SIZE = 224
# Set EXTRACTOR_OUTPUT_ACTIVATION based on the range of your float watermarks:
# 'sigmoid' for [0,1], 'tanh' for [-1,1], None for unbounded floats
EXTRACTOR_OUTPUT_ACTIVATION = 'sigmoid' # EXAMPLE: if your float watermarks are in [0,1]
# EXTRACTOR_OUTPUT_ACTIVATION = None # EXAMPLE: if your floats are unbounded

# Training & Loss Weights
BATCH_SIZE = 1 # START WITH 1 due to OOM, then try to increase if possible
LAMBDA_IMG_RECON = 1.0
LAMBDA_MSG_EXTRACT = 1.0 # Weight for MSE loss of float watermark extraction

# Data
TRAIN_IMAGE_DIR = "dataset/" # Your image directory
WATERMARK_FILE = "distinct.pt" # Ensure this contains (N, WATERMARK_DIM) float tensors

# Attack Layer Config
APPLY_ATTACKS = True
ATTACK_PROBABILITY = 0.7

# --- U-Net Model Definitions ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels_deep, in_channels_skip, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            target_up_channels = in_channels_deep
        else:
            target_up_channels = in_channels_deep // 2 if in_channels_deep > 1 else in_channels_deep
            self.up = nn.ConvTranspose2d(in_channels_deep, target_up_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(target_up_channels + in_channels_skip, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]; diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__(); self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)

class UNetWatermarkEmbedder(nn.Module):
    def __init__(self, n_channels_in=N_CHANNELS_IN, n_channels_out=N_CHANNELS_OUT, watermark_dim=WATERMARK_DIM,
                 bilinear=BILINEAR_UPSAMPLE, normalize_watermark_spatial=NORMALIZE_WATERMARK_SPATIAL,
                 watermark_bottleneck_fixed_res=WATERMARK_BOTTLENECK_FIXED_RES):
        super(UNetWatermarkEmbedder, self).__init__()
        self.n_channels_in, self.n_channels_out, self.watermark_dim = n_channels_in, n_channels_out, watermark_dim
        self.bilinear, self.normalize_watermark_spatial = bilinear, normalize_watermark_spatial
        self.watermark_bottleneck_fixed_res = watermark_bottleneck_fixed_res
        self.inc = DoubleConv(n_channels_in, 64); self.down1 = Down(64, 128); self.down2 = Down(128, 256)
        self.down3 = Down(256, 512); self.down4 = Down(512, 1024)
        self.bottleneck_process = DoubleConv(1024, 1024)
        self.bottleneck_to_rgb_conv = nn.Conv2d(1024, 3, kernel_size=1, padding=0)
        self.watermark_fc = nn.Linear(self.watermark_dim, self.watermark_bottleneck_fixed_res * self.watermark_bottleneck_fixed_res)
        self.up1 = Up(4, 512, 512, self.bilinear); self.up2 = Up(512, 256, 256, self.bilinear)
        self.up3 = Up(256, 128, 128, self.bilinear); self.up4 = Up(128, 64, 64, self.bilinear)
        self.outc = OutConv(64, n_channels_out)

    def forward(self, x_img, watermark_vec):
        x1=self.inc(x_img); x2=self.down1(x1); x3=self.down2(x2); x4=self.down3(x3); x5=self.down4(x4)
        bottleneck_features_processed = self.bottleneck_process(x5)
        img_bottleneck_3ch = self.bottleneck_to_rgb_conv(bottleneck_features_processed)
        B, _, H_bottleneck, W_bottleneck = img_bottleneck_3ch.shape
        watermark_spatial_flat = self.watermark_fc(watermark_vec)
        watermark_spatial_fixed = watermark_spatial_flat.view(B, 1, self.watermark_bottleneck_fixed_res, self.watermark_bottleneck_fixed_res)
        watermark_spatial_resized = F.interpolate(watermark_spatial_fixed, size=(H_bottleneck, W_bottleneck), mode='bilinear', align_corners=False)
        if self.normalize_watermark_spatial: watermark_spatial_resized = torch.tanh(watermark_spatial_resized)
        combined_bottleneck_4ch = torch.cat([img_bottleneck_3ch, watermark_spatial_resized], dim=1)
        x_decoded = self.up1(combined_bottleneck_4ch, x4); x_decoded = self.up2(x_decoded, x3)
        x_decoded = self.up3(x_decoded, x2); x_decoded = self.up4(x_decoded, x1)
        logits = self.outc(x_decoded)
        return torch.sigmoid(logits)

# --- ViT Extractor Definition ---
class ViTWatermarkExtractor(nn.Module):
    def __init__(self, watermark_output_dim=WATERMARK_DIM, vit_model_name='vit_b_16', pretrained=True,
                 output_activation=EXTRACTOR_OUTPUT_ACTIVATION): # Use configured activation
        super(ViTWatermarkExtractor, self).__init__()
        self.vit_model_name = vit_model_name
        if vit_model_name == 'vit_b_16':
            self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, watermark_output_dim)
        # Add other ViT model options if needed
        else: raise ValueError(f"Unsupported ViT model name: {vit_model_name}")

        self.output_activation_fn = None
        if output_activation == 'sigmoid': self.output_activation_fn = nn.Sigmoid()
        elif output_activation == 'tanh': self.output_activation_fn = nn.Tanh()

        try:
            self.preprocess = T.Compose([
                T.Resize((VIT_IMG_SIZE, VIT_IMG_SIZE), interpolation=InterpolationMode.BICUBIC, antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        except TypeError: # Older torchvision
             self.preprocess = T.Compose([
                T.Resize((VIT_IMG_SIZE, VIT_IMG_SIZE)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def forward(self, x_img_watermarked):
        x_processed = self.preprocess(x_img_watermarked)
        predicted_values = self.vit(x_processed)
        if self.output_activation_fn: predicted_values = self.output_activation_fn(predicted_values)
        return predicted_values

# --- Attack Layer (Example) ---
class AttackLayer(nn.Module):
    def __init__(self, attack_probability=ATTACK_PROBABILITY, jpeg_quality_range=(30, 90), noise_std_range=(0.01, 0.1)):
        super(AttackLayer, self).__init__()
        self.attack_probability = attack_probability; self.jpeg_quality_range = jpeg_quality_range
        self.noise_std_range = noise_std_range
        self.gaussian_blur = T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.5) # Increased sigma range

    def add_gaussian_noise(self, img_tensor):
        std = random.uniform(self.noise_std_range[0], self.noise_std_range[1])
        return torch.clamp(img_tensor + torch.randn_like(img_tensor) * std, 0, 1)

    def jpeg_compress_approx(self, img_tensor_batch): # Batch-wise approximation
        # This is a very simplified placeholder for JPEG artifacts
        # For more realistic JPEG, consider libraries like diffjpeg or manual PIL save/load loops (slow)
        if random.random() < 0.5: # Apply one of two simple "artifact" types
            # Slight blur
            k_size = random.choice([3, 5])
            sigma = random.uniform(0.1, 0.8)
            blurred = T.GaussianBlur(kernel_size=k_size, sigma=sigma)(img_tensor_batch)
            return blurred
        else:
            # Add a bit of blockiness by downsampling and upsampling
            B, C, H, W = img_tensor_batch.shape
            scale = random.uniform(0.7, 0.95)
            down_h, down_w = int(H * scale), int(W * scale)
            if down_h < 1 or down_w < 1: return img_tensor_batch # Avoid too small
            downsampled = F.interpolate(img_tensor_batch, size=(down_h, down_w), mode='bilinear', align_corners=False, antialias=True)
            upsampled = F.interpolate(downsampled, size=(H,W), mode='nearest') # Nearest for blockiness
            return upsampled


    def forward(self, x): # x is a batch of images (B, C, H, W) in [0,1]
        if not self.training or random.random() > self.attack_probability: return x

        x_attacked = x.clone() # Don't modify original in-place if it's used elsewhere

        # Apply a sequence of attacks to the whole batch
        if random.random() < 0.7: # Prob of applying noise
            x_attacked = self.add_gaussian_noise(x_attacked)
        if random.random() < 0.5: # Prob of applying blur
            x_attacked = self.gaussian_blur(x_attacked)
        if random.random() < 0.4: # Prob of applying "JPEG" approx
            x_attacked = self.jpeg_compress_approx(x_attacked)
        
        return torch.clamp(x_attacked, 0, 1)

# --- Helper Functions for Padding & Cropping ---
def pad_to_divisible(tensor, divisor=DIVISOR_FOR_PADDING):
    is_batched = tensor.ndim == 4
    if not is_batched: tensor = tensor.unsqueeze(0)
    B, C, H, W = tensor.shape
    pad_h = (divisor - (H % divisor)) % divisor; pad_w = (divisor - (W % divisor)) % divisor
    if pad_h == 0 and pad_w == 0:
        if not is_batched: tensor = tensor.squeeze(0)
        return tensor, (0, 0, 0, 0)
    padding_tuple = (0, pad_w, 0, pad_h)
    padded_tensor = F.pad(tensor, padding_tuple, mode='replicate')
    if not is_batched: padded_tensor = padded_tensor.squeeze(0)
    return padded_tensor, padding_tuple

def crop_from_padding(tensor, original_H, original_W):
    return tensor[:, :, :original_H, :original_W]

# --- Custom Dataset ---
class ImageWatermarkDataset(Dataset):
    def __init__(self, image_dir, watermark_file_path, base_transform=None, watermark_dim=WATERMARK_DIM, divisor_for_padding=DIVISOR_FOR_PADDING):
        self.image_dir = image_dir
        self.base_transform = base_transform if base_transform is not None else T.ToTensor()
        self.watermark_dim = watermark_dim
        self.divisor_for_padding = divisor_for_padding
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*/*.*')))
        if not self.image_paths: self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*.*')))
        if not self.image_paths: raise FileNotFoundError(f"No images in {self.image_dir}")
        print(f"Found {len(self.image_paths)} images in {self.image_dir}")
        try:
            self.all_watermarks = torch.load(watermark_file_path) # Should be (N, WATERMARK_DIM) floats
            if not isinstance(self.all_watermarks, torch.Tensor): raise TypeError("WM not a Tensor.")
            if self.all_watermarks.ndim != 2 or self.all_watermarks.shape[1] != self.watermark_dim:
                raise ValueError(f"WM shape mismatch. Expected (N, {self.watermark_dim}), got {self.all_watermarks.shape}")
            # Ensure your float watermarks are in the range expected by EXTRACTOR_OUTPUT_ACTIVATION
            # e.g., if 'sigmoid', they should be [0,1]. If 'tanh', [-1,1].
            # Add normalization here if needed:
            # if EXTRACTOR_OUTPUT_ACTIVATION == 'sigmoid':
            #     self.all_watermarks = torch.clamp(self.all_watermarks, 0, 1) # Example
            # elif EXTRACTOR_OUTPUT_ACTIVATION == 'tanh':
            #     self.all_watermarks = torch.clamp(self.all_watermarks, -1, 1) # Example
            print(f"Loaded {self.all_watermarks.shape[0]} float watermarks from {watermark_file_path}")
        except Exception as e: print(f"Error loading watermarks: {e}"); raise
        if len(self.image_paths) > self.all_watermarks.shape[0]: print("Warning: Reusing watermarks.")

    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try: image_pil = Image.open(img_path).convert('RGB')
        except Exception as e: raise IOError(f"Could not read image {img_path}: {e}")
        image_tensor_original = self.base_transform(image_pil)
        image_tensor_padded_for_unet, _ = pad_to_divisible(image_tensor_original, self.divisor_for_padding)
        watermark_idx = idx % self.all_watermarks.shape[0]
        watermark_message = self.all_watermarks[watermark_idx]
        return image_tensor_original, image_tensor_padded_for_unet, watermark_message

# --- Custom Collate Function ---
def custom_collate_fn_dual_img(batch):
    original_images = [item[0] for item in batch]; padded_images_unet = [item[1] for item in batch]
    watermarks = [item[2] for item in batch]
    max_h_orig = max(img.shape[1] for img in original_images); max_w_orig = max(img.shape[2] for img in original_images)
    collated_original_images = []
    for img in original_images:
        pad_w = max_w_orig - img.shape[2]; pad_h = max_h_orig - img.shape[1]
        collated_original_images.append(F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0))
    max_h_unet = max(img.shape[1] for img in padded_images_unet); max_w_unet = max(img.shape[2] for img in padded_images_unet)
    collated_padded_images_unet = []
    for img in padded_images_unet:
        pad_w = max_w_unet - img.shape[2]; pad_h = max_h_unet - img.shape[1]
        collated_padded_images_unet.append(F.pad(img, (0, pad_w, 0, pad_h), mode='replicate'))
    return torch.stack(collated_original_images), torch.stack(collated_padded_images_unet), torch.stack(watermarks)

# --- Main Training Script ---
def train_system():
    print(f"Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        # For OOM with fragmentation, try setting this before running the script:
        # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    train_base_transform = T.ToTensor()
    try:
        train_dataset = ImageWatermarkDataset(
            image_dir=TRAIN_IMAGE_DIR, watermark_file_path=WATERMARK_FILE,
            base_transform=train_base_transform, watermark_dim=WATERMARK_DIM,
            divisor_for_padding=DIVISOR_FOR_PADDING
        )
    except Exception as e: print(f"Dataset creation failed: {e}"); return
    if len(train_dataset) == 0: print("Dataset empty."); return

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False, # START WITH num_workers=0, pin_memory=False for stability
        collate_fn=custom_collate_fn_dual_img
    )
    print(f"DataLoader created with {len(train_dataloader)} batches.")

    embedder = UNetWatermarkEmbedder().to(DEVICE)
    extractor = ViTWatermarkExtractor(output_activation=EXTRACTOR_OUTPUT_ACTIVATION).to(DEVICE)
    attack_layer = AttackLayer().to(DEVICE) if APPLY_ATTACKS else nn.Identity().to(DEVICE)

    optimizer_embedder = optim.Adam(embedder.parameters(), lr=EMBEDDER_LR)
    optimizer_extractor = optim.Adam(extractor.parameters(), lr=EXTRACTOR_LR)

    criterion_img_recon = nn.L1Loss()
    criterion_msg_extract = nn.MSELoss() # Changed to MSELoss for float watermarks

    scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))
    print(f"Embedder params: {sum(p.numel() for p in embedder.parameters() if p.requires_grad):,}")
    print(f"Extractor params: {sum(p.numel() for p in extractor.parameters() if p.requires_grad):,}")
    print(f"Starting training for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        embedder.train(); extractor.train()
        if APPLY_ATTACKS: attack_layer.train()
        total_loss_embedder_epoch, total_loss_extractor_epoch = 0.0, 0.0
        total_loss_img_recon_epoch, total_loss_msg_extract_epoch = 0.0, 0.0
        start_time = time.time()

        for batch_idx, (original_images, padded_images_for_unet, target_watermarks) in enumerate(train_dataloader):
            original_images = original_images.to(DEVICE)
            padded_images_for_unet = padded_images_for_unet.to(DEVICE)
            target_watermarks = target_watermarks.to(DEVICE)

            optimizer_embedder.zero_grad()
            optimizer_extractor.zero_grad()

            with autocast(enabled=(DEVICE.type == 'cuda')):
                output_watermarked_padded = embedder(padded_images_for_unet, target_watermarks)
                
                # For L1 loss, comparing with original_images (which are batch-padded originals)
                # We need to crop/resize output_watermarked_padded to match original_images's shape
                # This assumes output_watermarked_padded (from U-Net on unet-padded) can be mapped to original_images (batch-padded originals)
                # For simplicity, let's stick to U-Net reconstructing its input's padded space.
                # If `original_images` and `padded_images_for_unet` have different batch-padded shapes due to collate,
                # this needs careful handling.
                # The current collate_fn makes original_images and padded_images_for_unet have shapes
                # determined by max within their respective types.
                # Let's use padded_images_for_unet as target for U-Net reconstruction to keep it simple.
                loss_img_recon = criterion_img_recon(output_watermarked_padded, padded_images_for_unet)

                attacked_watermarked_images = attack_layer(output_watermarked_padded) # Pass through attack
                predicted_watermark_values = extractor(attacked_watermarked_images)
                loss_msg_extract = criterion_msg_extract(predicted_watermark_values, target_watermarks)
                loss_embedder = LAMBDA_IMG_RECON * loss_img_recon + LAMBDA_MSG_EXTRACT * loss_msg_extract
                loss_extractor = loss_msg_extract # Extractor only cares about this

            # Scaler logic for joint training:
            # Backward pass for embedder (includes gradients from reconstruction and indirectly from extraction)
            scaler.scale(loss_embedder).backward(retain_graph=True if LAMBDA_MSG_EXTRACT > 0 else False)
            scaler.step(optimizer_embedder)

            # Backward pass specifically for extractor (if its loss isn't fully captured or needs separate scaling)
            # However, since loss_msg_extract (which is loss_extractor) is part of loss_embedder,
            # the gradients for the extractor should already be populated by the above backward call.
            # We can just step its optimizer.
            # If LAMBDA_MSG_EXTRACT = 0, then the extractor isn't trained via the embedder's loss.
            # To ensure extractor always trains on its direct loss:
            if LAMBDA_MSG_EXTRACT == 0: # If extractor isn't part of embedder's loss
                 optimizer_extractor.zero_grad() # Need to zero grads specifically for it
                 scaler.scale(loss_extractor).backward()
            # else, gradients for extractor are already there from loss_embedder.backward()

            scaler.step(optimizer_extractor)
            scaler.update()

            total_loss_embedder_epoch += loss_embedder.item()
            total_loss_extractor_epoch += loss_extractor.item()
            total_loss_img_recon_epoch += loss_img_recon.item()
            total_loss_msg_extract_epoch += loss_msg_extract.item()

            if (batch_idx + 1) % 1 == 0:
                print(f"  E{epoch+1} B{batch_idx+1}/{len(train_dataloader)} | "
                      f"L_Emb: {loss_embedder.item():.4f} (Rec: {loss_img_recon.item():.4f}, Msg: {loss_msg_extract.item():.4f}) | "
                      f"L_Ext: {loss_extractor.item():.4f}")
        
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] T: {epoch_time:.2f}s | Avg L_Emb: {total_loss_embedder_epoch/len(train_dataloader):.4f} | Avg L_Ext: {total_loss_extractor_epoch/len(train_dataloader):.4f}")
        if (epoch + 1) % 1 == 0:
            torch.save(embedder.state_dict(), os.path.join(MODEL_SAVE_DIR, f"embedder_epoch_{epoch+1}.pth"))
            torch.save(extractor.state_dict(), os.path.join(MODEL_SAVE_DIR, f"extractor_epoch_{epoch+1}.pth"))
            print(f"Ckpts saved @ epoch {epoch+1}")

    print("Training finished.")
    torch.save(embedder.state_dict(), EMBEDDER_MODEL_SAVE_PATH)
    torch.save(extractor.state_dict(), EXTRACTOR_MODEL_SAVE_PATH)
    print(f"Final models saved.")

# --- Inference Function (Placeholder) ---
def infer_system(embedder_path, extractor_path, image_path, output_dir, watermark_float_vector=None):
    print("Inference function placeholder. Needs implementation similar to training logic "
          "for loading, padding, embedding, attacking, extracting, and evaluating.")
    # Key steps:
    # 1. Load models (embedder, extractor, attack_layer if used in eval)
    # 2. Load image, preprocess (to tensor, pad for embedder)
    # 3. If watermark_float_vector is None, generate/load one (e.g., from WATERMARK_FILE)
    # 4. embedder_output = embedder(padded_image_for_unet, watermark_float_vector)
    # 5. Crop embedder_output to original image size, save this watermarked_image.
    # 6. (Optional) attacked_image = attack_layer(embedder_output) # Apply consistent attacks
    # 7. extracted_values = extractor(attacked_image_or_embedder_output)
    # 8. Calculate MSE or other metrics between watermark_float_vector and extracted_values.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Embedder-Extractor Watermarking System with Float WM.")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer'], help="Mode.")
    parser.add_argument('--input_image', type=str, help="Path to input image for inference.")
    parser.add_argument('--output_dir', type=str, default="inference_results_float_wm", help="Output dir for inference.")
    parser.add_argument('--model_embedder_path', type=str, default=EMBEDDER_MODEL_SAVE_PATH, help="Path to embedder model.")
    parser.add_argument('--model_extractor_path', type=str, default=EXTRACTOR_MODEL_SAVE_PATH, help="Path to extractor model.")
    parser.add_argument('--watermark_idx_infer', type=int, default=0, help="Index of watermark from WATERMARK_FILE for inference.")

    args = parser.parse_args()
    if args.mode == 'train':
        if not os.path.isdir(TRAIN_IMAGE_DIR): print(f"Dir '{TRAIN_IMAGE_DIR}' not found."); exit()
        if not os.path.isfile(WATERMARK_FILE): print(f"File '{WATERMARK_FILE}' not found."); exit()
        train_system()
    elif args.mode == 'infer':
        # Basic setup for inference call
        if not args.input_image: print("Need --input_image for inference"); exit()
        os.makedirs(args.output_dir, exist_ok=True)
        # Load a specific watermark for inference
        # test_watermarks = torch.load(WATERMARK_FILE)
        # selected_wm = test_watermarks[args.watermark_idx_infer].unsqueeze(0) # Add batch dim
        print(f"Inference mode selected. Calling placeholder with args: {args.model_embedder_path}, {args.model_extractor_path}, {args.input_image}, {args.output_dir}")
        # infer_system(args.model_embedder_path, args.model_extractor_path, args.input_image, args.output_dir, watermark_float_vector=selected_wm)
    else: print(f"Unknown mode: {args.mode}")