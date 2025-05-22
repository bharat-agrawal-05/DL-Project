import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import time
import os
import glob
import argparse

# --- UNet Model Definitions (Same as before) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

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
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNetWatermark_Corrected(nn.Module):
    def __init__(self, n_channels_in=3, n_channels_out=3, watermark_dim=128,
                 bilinear=True, normalize_watermark=True,
                 watermark_bottleneck_fixed_res=16):
        super(UNetWatermark_Corrected, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.watermark_dim = watermark_dim
        self.bilinear = bilinear
        self.normalize_watermark = normalize_watermark
        self.watermark_bottleneck_fixed_res = watermark_bottleneck_fixed_res
        self.inc = DoubleConv(n_channels_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.bottleneck_process = DoubleConv(1024, 1024)
        self.bottleneck_to_rgb_conv = nn.Conv2d(1024, 3, kernel_size=1, padding=0)
        self.watermark_fc = nn.Linear(
            self.watermark_dim,
            self.watermark_bottleneck_fixed_res * self.watermark_bottleneck_fixed_res
        )
        self.up1 = Up(4, 512, 512, self.bilinear)
        self.up2 = Up(512, 256, 256, self.bilinear)
        self.up3 = Up(256, 128, 128, self.bilinear)
        self.up4 = Up(128, 64, 64, self.bilinear)
        self.outc = OutConv(64, n_channels_out)

    def forward(self, x_img, watermark_vec):
        x1 = self.inc(x_img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        bottleneck_features_processed = self.bottleneck_process(x5)
        img_bottleneck_3ch = self.bottleneck_to_rgb_conv(bottleneck_features_processed)
        B, _, H_bottleneck, W_bottleneck = img_bottleneck_3ch.shape
        watermark_spatial_flat = self.watermark_fc(watermark_vec)
        watermark_spatial_fixed = watermark_spatial_flat.view(
            B, 1, self.watermark_bottleneck_fixed_res, self.watermark_bottleneck_fixed_res
        )
        watermark_spatial_resized = F.interpolate(
            watermark_spatial_fixed,
            size=(H_bottleneck, W_bottleneck),
            mode='bilinear',
            align_corners=False
        )
        if self.normalize_watermark:
            watermark_spatial_resized = torch.tanh(watermark_spatial_resized)
        combined_bottleneck_4ch = torch.cat([img_bottleneck_3ch, watermark_spatial_resized], dim=1)
        x_decoded = self.up1(combined_bottleneck_4ch, x4)
        x_decoded = self.up2(x_decoded, x3)
        x_decoded = self.up3(x_decoded, x2)
        x_decoded = self.up4(x_decoded, x1)
        logits = self.outc(x_decoded)
        return torch.sigmoid(logits)
# --- End of UNet Model Definitions ---


# --- Helper Functions for Padding and Cropping ---
def pad_to_divisible(tensor, divisor=16): # Expects (B,C,H,W) or (C,H,W)
    is_batched = tensor.ndim == 4
    if not is_batched:
        tensor = tensor.unsqueeze(0) # Add batch dim

    B, C, H, W = tensor.shape
    pad_h = (divisor - (H % divisor)) % divisor
    pad_w = (divisor - (W % divisor)) % divisor
    if pad_h == 0 and pad_w == 0:
        if not is_batched:
            tensor = tensor.squeeze(0)
        return tensor, (0, 0, 0, 0)
    padding_tuple = (0, pad_w, 0, pad_h)
    padded_tensor = F.pad(tensor, padding_tuple, mode='replicate')

    if not is_batched:
        padded_tensor = padded_tensor.squeeze(0)
    return padded_tensor, padding_tuple

def crop_from_padding(tensor, original_H, original_W): # Expects (B,C,H,W)
    return tensor[:, :, :original_H, :original_W]
# --- End of Helper Functions ---


# --- Custom Dataset for Images and Preloaded Watermarks ---
class ImageWatermarkDataset(Dataset):
    def __init__(self, image_dir, watermark_file_path, base_transform=None, watermark_dim=128, divisor_for_padding=16):
        self.image_dir = image_dir
        self.base_transform = base_transform if base_transform is not None else T.ToTensor()
        self.watermark_dim = watermark_dim
        self.divisor_for_padding = divisor_for_padding
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*/*.*')))
        if not self.image_paths:
             self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*.*')))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir} with patterns used.")
        print(f"Found {len(self.image_paths)} images in {self.image_dir}")
        try:
            self.all_watermarks = torch.load(watermark_file_path)
            if not isinstance(self.all_watermarks, torch.Tensor):
                raise TypeError("Watermark file did not contain a PyTorch Tensor.")
            if self.all_watermarks.ndim != 2 or self.all_watermarks.shape[1] != self.watermark_dim:
                raise ValueError(f"Watermarks tensor shape mismatch. Expected (N, {self.watermark_dim}), got {self.all_watermarks.shape}")
            print(f"Loaded {self.all_watermarks.shape[0]} watermarks from {watermark_file_path}")
        except Exception as e:
            print(f"Error loading watermarks from {watermark_file_path}: {e}")
            raise
        if len(self.image_paths) > self.all_watermarks.shape[0]:
            print(f"Warning: Number of images ({len(self.image_paths)}) is greater than number of watermarks ({self.all_watermarks.shape[0]}). Watermarks will be reused.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image_pil = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            raise IOError(f"Could not read image: {img_path}")
        image_tensor_original = self.base_transform(image_pil)
        image_tensor_padded, _ = pad_to_divisible(image_tensor_original, self.divisor_for_padding) # Pass (C,H,W)
        watermark_idx = idx % self.all_watermarks.shape[0]
        watermark = self.all_watermarks[watermark_idx]
        return image_tensor_padded, watermark
# --- End of Custom Dataset ---

# --- Custom Collate Function ---
def custom_collate_fn(batch):
    # batch is a list of tuples: [(image_tensor_padded_1, watermark_1), (image_tensor_padded_2, watermark_2), ...]
    # image_tensor_padded_i are individually padded by __getitem__

    images = [item[0] for item in batch]  # List of (C, H_pad_i, W_pad_i) tensors
    watermarks = [item[1] for item in batch] # List of (watermark_dim,) tensors

    # Pad images in the batch to the max H and W of this batch
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded_images_batch = []
    for img in images:
        # F.pad expects (pad_left, pad_right, pad_top, pad_bottom)
        # We want to pad on the right and bottom
        pad_w_needed = max_w - img.shape[2]
        pad_h_needed = max_h - img.shape[1]
        # Ensure no negative padding if an image is already max size
        current_padded_img = F.pad(img, (0, pad_w_needed, 0, pad_h_needed), mode='replicate')
        padded_images_batch.append(current_padded_img)

    # Stack the uniformly padded images and the watermarks
    images_collated = torch.stack(padded_images_batch, 0)
    watermarks_collated = torch.stack(watermarks, 0)

    return images_collated, watermarks_collated
# --- End of Custom Collate Function ---


# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
NUM_EPOCHS = 50
N_CHANNELS_IN = 3
N_CHANNELS_OUT = 3
WATERMARK_DIM = 128
MODEL_SAVE_PATH = "unet_watermark_embedder_dynamic.pth"
BILINEAR_UPSAMPLE = True
NORMALIZE_WATERMARK = True
WATERMARK_BOTTLENECK_FIXED_RES = 16
DIVISOR_FOR_PADDING = 16
TRAIN_IMAGE_DIR = "test/"
WATERMARK_FILE = "distinct.pt"
# --- End of Configuration ---


# --- Main Training Script ---
def train_model():
    print(f"Using device: {DEVICE}")
    train_base_transform = T.Compose([T.ToTensor()])
    try:
        train_dataset = ImageWatermarkDataset(
            image_dir=TRAIN_IMAGE_DIR,
            watermark_file_path=WATERMARK_FILE,
            base_transform=train_base_transform,
            watermark_dim=WATERMARK_DIM,
            divisor_for_padding=DIVISOR_FOR_PADDING
        )
    except Exception as e:
        print(f"Failed to create dataset: {e}")
        return
    if len(train_dataset) == 0:
        print("Dataset is empty. Check image paths and watermark file.")
        return

    # Use the custom_collate_fn in DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2, # Can set to 0 for debugging Dataloader issues
        pin_memory=True,
        collate_fn=custom_collate_fn # <<< HERE
    )
    print(f"DataLoader created with {len(train_dataloader)} batches, using custom collate_fn.")

    model = UNetWatermark_Corrected(
        n_channels_in=N_CHANNELS_IN,
        n_channels_out=N_CHANNELS_OUT,
        watermark_dim=WATERMARK_DIM,
        bilinear=BILINEAR_UPSAMPLE,
        normalize_watermark=NORMALIZE_WATERMARK,
        watermark_bottleneck_fixed_res=WATERMARK_BOTTLENECK_FIXED_RES
    ).to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        for batch_idx, (padded_images, watermarks) in enumerate(train_dataloader):
            # padded_images are now padded to max dimensions within the batch
            padded_images = padded_images.to(DEVICE)
            watermarks = watermarks.to(DEVICE)
            optimizer.zero_grad()
            output_images_padded = model(padded_images, watermarks)
            loss = criterion(output_images_padded, padded_images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_dataloader):
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] completed. Avg Loss: {avg_epoch_loss:.4f}, Time: {epoch_time:.2f}s")
        if (epoch + 1) % 10 == 0:
            chkpt_path = f"unet_watermark_dynamic_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), chkpt_path)
            print(f"Checkpoint saved to {chkpt_path}")
    print("Training finished.")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")
# --- End of Training Script ---


# --- Inference Function Example ---
def infer_single_image(model_path, image_path, output_path, watermark_file_path, watermark_idx=0):
    print(f"Using device: {DEVICE}")
    model = UNetWatermark_Corrected(
        n_channels_in=N_CHANNELS_IN, n_channels_out=N_CHANNELS_OUT, watermark_dim=WATERMARK_DIM,
        bilinear=BILINEAR_UPSAMPLE, normalize_watermark=NORMALIZE_WATERMARK,
        watermark_bottleneck_fixed_res=WATERMARK_BOTTLENECK_FIXED_RES
    ).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return
    model.eval()
    try:
        input_pil = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return
    original_H, original_W = input_pil.height, input_pil.width
    transform_to_tensor = T.ToTensor()
    input_tensor_original = transform_to_tensor(input_pil) # No .to(DEVICE) yet
    input_tensor_padded, _ = pad_to_divisible(input_tensor_original.unsqueeze(0), DIVISOR_FOR_PADDING) # Pass (B,C,H,W)
    input_tensor_padded = input_tensor_padded.to(DEVICE) # Now send to device
    try:
        all_watermarks = torch.load(watermark_file_path, map_location=DEVICE)
        if watermark_idx >= all_watermarks.shape[0]: watermark_idx = 0
        watermark_to_embed = all_watermarks[watermark_idx].unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f"Error loading watermark: {e}")
        return
    with torch.no_grad():
        output_padded_tensor = model(input_tensor_padded, watermark_to_embed)
    output_cropped_tensor = crop_from_padding(output_padded_tensor.cpu(), original_H, original_W) # CPU before ToPILImage
    output_image_pil = T.ToPILImage()(output_cropped_tensor.squeeze(0))
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        output_image_pil.save(output_path)
        print(f"Watermarked image of size {output_image_pil.size} saved to {output_path}")
    except Exception as e:
        print(f"Error saving output image: {e}")
# --- End of Inference Function Example ---


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or Infer U-Net Watermark Model.")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer'], help="Mode of operation.")
    parser.add_argument('--input_image', type=str, help="Path to input image for inference.")
    parser.add_argument('--output_image', type=str, help="Path to save watermarked image for inference.")
    parser.add_argument('--model_path', type=str, default=MODEL_SAVE_PATH, help="Path to model file.")
    parser.add_argument('--watermark_idx', type=int, default=0, help="Index of watermark for inference.")
    args = parser.parse_args()

    if args.mode == 'train':
        if not os.path.isdir(TRAIN_IMAGE_DIR): print(f"Error: Dir '{TRAIN_IMAGE_DIR}' not found."); exit()
        if not os.path.isfile(WATERMARK_FILE): print(f"Error: File '{WATERMARK_FILE}' not found."); exit()
        train_model()
    elif args.mode == 'infer':
        if not args.input_image or not args.output_image: print("Error: --input_image and --output_image must be specified for inference."); exit()
        if not os.path.isfile(args.input_image): print(f"Error: Input image '{args.input_image}' not found."); exit()
        if not os.path.isfile(args.model_path): print(f"Error: Model file '{args.model_path}' not found."); exit()
        if not os.path.isfile(WATERMARK_FILE): print(f"Error: Watermark file '{WATERMARK_FILE}' not found."); exit()
        infer_single_image(
            model_path=args.model_path, image_path=args.input_image, output_path=args.output_image,
            watermark_file_path=WATERMARK_FILE, watermark_idx=args.watermark_idx
        )
    else:
        print(f"Unknown mode: {args.mode}")