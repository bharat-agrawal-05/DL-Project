import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import os
import argparse

# --- UNet Model Definitions (MODIFIED) ---
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
    def __init__(self, in_channels_deep, in_channels_skip, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            target_up_channels = in_channels_deep
        else:
            target_up_channels = in_channels_deep // 2 if in_channels_deep > 1 else in_channels_deep
            self.up = nn.ConvTranspose2d(in_channels_deep, target_up_channels, kernel_size=2, stride=2)

        self.conv = DoubleConv(target_up_channels + in_channels_skip, out_channels)

    def forward(self, x1, x2): # x1 from deep (after upsampling), x2 from skip connection
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
    # Added img_size argument to __init__
    def __init__(self, n_channels_in=3, n_channels_out=3, watermark_dim=128, bilinear=False, normalize_watermark=True, img_size=256):
        super(UNetWatermark_Corrected, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.watermark_dim = watermark_dim
        self.bilinear = bilinear
        self.normalize_watermark = normalize_watermark
        self.img_size = img_size # Store img_size

        # Encoder
        self.inc = DoubleConv(n_channels_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.bottleneck_process = DoubleConv(1024, 1024)
        self.bottleneck_to_rgb_conv = nn.Conv2d(1024, 3, kernel_size=1, padding=0)

        # --- MODIFICATION START ---
        # Calculate bottleneck spatial dimensions based on img_size and downsampling
        downsample_factor = 2**4 # For 4 downsampling layers
        if self.img_size % downsample_factor != 0:
            raise ValueError(f"IMG_SIZE ({self.img_size}) must be divisible by {downsample_factor} for this U-Net architecture.")
        self.H_bottleneck_calc = self.img_size // downsample_factor
        self.W_bottleneck_calc = self.img_size // downsample_factor
        
        # Initialize watermark_fc here
        self.watermark_fc = nn.Linear(self.watermark_dim, self.H_bottleneck_calc * self.W_bottleneck_calc)
        # --- MODIFICATION END ---

        # Decoder
        # Input to up1 is combined_bottleneck_4ch (3 from image bottleneck, 1 from spatial watermark)
        self.up1 = Up(4, 512, 512, bilinear)
        self.up2 = Up(512, 256, 256, bilinear)
        self.up3 = Up(256, 128, 128, bilinear)
        self.up4 = Up(128, 64, 64, bilinear)

        self.outc = OutConv(64, n_channels_out)

    def forward(self, x_img, watermark_vec):
        # Encoder
        x1 = self.inc(x_img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Bottleneck
        bottleneck_features_processed = self.bottleneck_process(x5)
        img_bottleneck_3ch = self.bottleneck_to_rgb_conv(bottleneck_features_processed)

        B, _, H_bottleneck_runtime, W_bottleneck_runtime = img_bottleneck_3ch.shape

        # Optional: Sanity check if runtime dimensions match calculated ones
        assert H_bottleneck_runtime == self.H_bottleneck_calc, \
            f"Runtime H_bottleneck {H_bottleneck_runtime} != Calculated H_bottleneck {self.H_bottleneck_calc}. Input image size or model structure mismatch?"
        assert W_bottleneck_runtime == self.W_bottleneck_calc, \
            f"Runtime W_bottleneck {W_bottleneck_runtime} != Calculated W_bottleneck {self.W_bottleneck_calc}. Input image size or model structure mismatch?"

        # --- MODIFICATION START ---
        # watermark_fc is now pre-initialized, no need to create it here
        # The .to(device) for watermark_fc happens when model.to(device) is called.
        # watermark_vec should also be on the same device.
        watermark_spatial_flat = self.watermark_fc(watermark_vec)
        # --- MODIFICATION END ---
        
        watermark_spatial = watermark_spatial_flat.view(B, 1, H_bottleneck_runtime, W_bottleneck_runtime)
        if self.normalize_watermark:
            watermark_spatial = torch.tanh(watermark_spatial)

        combined_bottleneck_4ch = torch.cat([img_bottleneck_3ch, watermark_spatial], dim=1)

        # Decoder
        x_decoded = self.up1(combined_bottleneck_4ch, x4)
        x_decoded = self.up2(x_decoded, x3)
        x_decoded = self.up3(x_decoded, x2)
        x_decoded = self.up4(x_decoded, x1)

        logits = self.outc(x_decoded)
        return torch.sigmoid(logits)
# --- End of UNet Model Definitions ---


# --- Configuration (MUST MATCH TRAINING CONFIGURATION FOR THE SAVED MODEL) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256     # CRITICAL: This IMG_SIZE must match the one used when the loaded model was trained.
N_CHANNELS_IN = 3
N_CHANNELS_OUT = 3
WATERMARK_DIM = 128
BILINEAR_UPSAMPLE = False # Ensure this matches the training config of the loaded model
NORMALIZE_WATERMARK = True # Ensure this matches the training config of the loaded model

# --- Paths ---
DEFAULT_MODEL_PATH = "unet_watermark_embedder.pth"
DEFAULT_WATERMARK_FILE = "distinct.pt"

def infer_watermark(model_path, image_path, output_path, watermark_file_path, watermark_idx=0):
    print(f"Using device: {DEVICE}")

    # 1. Load Model
    model = UNetWatermark_Corrected(
        n_channels_in=N_CHANNELS_IN,
        n_channels_out=N_CHANNELS_OUT,
        watermark_dim=WATERMARK_DIM,
        bilinear=BILINEAR_UPSAMPLE,
        normalize_watermark=NORMALIZE_WATERMARK,
        img_size=IMG_SIZE # <<< --- Pass IMG_SIZE here
    ).to(DEVICE)

    try:
        # Load the state dict. strict=True is default and good for catching issues.
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    except RuntimeError as e: # Catch RuntimeError which often includes size mismatches or other load errors
        print(f"Error loading model state_dict: {e}")
        print("This might be due to a mismatch in model architecture (e.g., IMG_SIZE, WATERMARK_DIM, BILINEAR_UPSAMPLE, NORMALIZE_WATERMARK between training and inference) or a corrupted model file.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading model state_dict: {e}")
        return

    model.eval()

    # (The rest of your inference function remains the same)
    # 2. Prepare Input Image
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
    ])

    try:
        input_image = Image.open(image_path).convert('RGB')
        input_tensor = transform(input_image).unsqueeze(0).to(DEVICE)
    except FileNotFoundError:
        print(f"Error: Input image not found at {image_path}")
        return
    except Exception as e:
        print(f"Error processing input image {image_path}: {e}")
        return

    # 3. Prepare Watermark
    try:
        all_watermarks = torch.load(watermark_file_path, map_location=DEVICE)
        if not isinstance(all_watermarks, torch.Tensor) or all_watermarks.ndim != 2 or all_watermarks.shape[1] != WATERMARK_DIM:
            raise ValueError(f"Watermark file {watermark_file_path} is not a valid tensor of shape (N, {WATERMARK_DIM}).")
        
        if watermark_idx >= all_watermarks.shape[0]:
            print(f"Warning: Watermark index {watermark_idx} is out of bounds for {all_watermarks.shape[0]} watermarks. Using index 0.")
            watermark_idx = 0
        
        watermark_to_embed = all_watermarks[watermark_idx].unsqueeze(0).to(DEVICE)
        print(f"Using watermark at index {watermark_idx} from {watermark_file_path}")

    except FileNotFoundError:
        print(f"Error: Watermark file not found at {watermark_file_path}")
        return
    except Exception as e:
        print(f"Error loading or processing watermark: {e}")
        return

    # 4. Perform Inference
    with torch.no_grad():
        watermarked_tensor = model(input_tensor, watermark_to_embed)

    # 5. Post-process and Save Output Image
    output_image_pil = T.ToPILImage()(watermarked_tensor.squeeze(0).cpu())

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        output_image_pil.save(output_path)
        print(f"Watermarked image saved to {output_path}")
    except Exception as e:
        print(f"Error saving output image to {output_path}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Embed a watermark into an image using a U-Net model.")
    parser.add_argument("input_image", type=str, help="Path to the input image file.")
    parser.add_argument("output_image", type=str, help="Path to save the watermarked output image.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"Path to the trained U-Net model file (default: {DEFAULT_MODEL_PATH}).")
    parser.add_argument("--watermark_file", type=str, default=DEFAULT_WATERMARK_FILE,
                        help=f"Path to the .pt file containing watermark tensors (default: {DEFAULT_WATERMARK_FILE}).")
    parser.add_argument("--watermark_index", type=int, default=0,
                        help="Index of the watermark to use from the watermark file (default: 0).")
    # You could also add an argument for IMG_SIZE if it might vary between models
    # parser.add_argument("--img_size", type=int, default=IMG_SIZE, help="Image size model was trained with.")

    args = parser.parse_args()

    # If you add --img_size argument, you'd update IMG_SIZE from args here:
    # IMG_SIZE = args.img_size

    if not os.path.isfile(args.input_image):
        print(f"Error: Input image '{args.input_image}' not found.")
        exit()
    if not os.path.isfile(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found.")
        exit()
    if not os.path.isfile(args.watermark_file):
        print(f"Error: Watermark file '{args.watermark_file}' not found.")
        exit()

    infer_watermark(
        model_path=args.model_path,
        image_path=args.input_image,
        output_path=args.output_image,
        watermark_file_path=args.watermark_file,
        watermark_idx=args.watermark_index
    )