import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder # To load images
from PIL import Image # For ImageFolder or custom image loading
import time
import os
import glob # For finding image files if not using ImageFolder structure

# --- UNet Model Definitions (SAME AS PREVIOUSLY PROVIDED) ---
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
            target_up_channels = in_channels_deep // 2 if in_channels_deep > 1 else in_channels_deep # Avoid 0 channels
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
    def __init__(self, n_channels_in=3, n_channels_out=3, watermark_dim=128, bilinear=False, normalize_watermark=True):
        super(UNetWatermark_Corrected, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.watermark_dim = watermark_dim
        self.bilinear = bilinear
        self.normalize_watermark = normalize_watermark

        # Encoder
        self.inc = DoubleConv(n_channels_in, 64)    # x1 out: 64 ch
        self.down1 = Down(64, 128)                  # x2 out: 128 ch
        self.down2 = Down(128, 256)                 # x3 out: 256 ch
        self.down3 = Down(256, 512)                 # x4 out: 512 ch
        self.down4 = Down(512, 1024)                # x5 out: 1024 ch

        self.bottleneck_process = DoubleConv(1024, 1024) 
        self.bottleneck_to_rgb_conv = nn.Conv2d(1024, 3, kernel_size=1, padding=0)
        self.watermark_fc = None 

        # Decoder
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
        
        B, _, H_bottleneck, W_bottleneck = img_bottleneck_3ch.shape
        if self.watermark_fc is None or self.watermark_fc.out_features != H_bottleneck * W_bottleneck:
            self.watermark_fc = nn.Linear(self.watermark_dim, H_bottleneck * W_bottleneck).to(watermark_vec.device)
        
        watermark_spatial_flat = self.watermark_fc(watermark_vec)
        watermark_spatial = watermark_spatial_flat.view(B, 1, H_bottleneck, W_bottleneck)
        if self.normalize_watermark:
            watermark_spatial = torch.tanh(watermark_spatial)
        
        combined_bottleneck_4ch = torch.cat([img_bottleneck_3ch, watermark_spatial], dim=1)
        
        # Decoder
        x_decoded = self.up1(combined_bottleneck_4ch, x4)
        x_decoded = self.up2(x_decoded, x3)
        x_decoded = self.up3(x_decoded, x2)
        x_decoded = self.up4(x_decoded, x1)
        
        logits = self.outc(x_decoded)
        # If input images are normalized to [0,1], output should be too.
        # If input images are [-1,1], output might be tanh(logits)
        # For now, assume input images are [0,1] from ToTensor, so Sigmoid is good
        return torch.sigmoid(logits) # Outputting in [0,1] range
# --- End of UNet Model Definitions ---


# --- Custom Dataset for Images and Preloaded Watermarks ---
class ImageWatermarkDataset(Dataset):
    def __init__(self, image_dir, watermark_file_path, transform=None, watermark_dim=128):
        self.image_dir = image_dir
        self.transform = transform
        self.watermark_dim = watermark_dim

        # Load images
        # Option 1: Using ImageFolder (requires subdirectories in image_dir)
        # self.image_dataset = ImageFolder(root=self.image_dir)
        # self.image_paths = [item[0] for item in self.image_dataset.samples]
        
        # Option 2: Glob for image files (if images are directly in image_dir or a single subfolder)
        # Adjust pattern if your images are in a specific subfolder like 'train/all_images/'
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*/*.*'))) # For 'train/class/img.jpg'
        if not self.image_paths:
             self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*.*'))) # For 'train/img.jpg'
        
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir} with patterns used.")
        print(f"Found {len(self.image_paths)} images in {self.image_dir}")


        # Load watermarks
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
            image = Image.open(img_path).convert('RGB') # Ensure 3 channels
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            # Return a dummy image or skip, here raising error
            raise IOError(f"Could not read image: {img_path}")


        if self.transform:
            image = self.transform(image)

        # Get watermark for this image
        # Cycle through watermarks if there are fewer watermarks than images
        watermark_idx = idx % self.all_watermarks.shape[0]
        watermark = self.all_watermarks[watermark_idx]

        return image, watermark

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 4 # Adjust based on your GPU memory and image size
NUM_EPOCHS = 50    # Increase for real training
IMG_SIZE = 256     # Must be divisible by 2^4 = 16 for this U-Net
N_CHANNELS_IN = 3
N_CHANNELS_OUT = 3  # Outputting an image, so 3 channels
WATERMARK_DIM = 128 # Should match the dimension in your distinct.pt
MODEL_SAVE_PATH = "unet_watermark_embedder.pth"
BILINEAR_UPSAMPLE = False # True to use nn.Upsample, False for nn.ConvTranspose2d
NORMALIZE_WATERMARK = True # Apply tanh to spatial watermark

# --- Paths ---
TRAIN_IMAGE_DIR = "test/"  # Your folder with images
WATERMARK_FILE = "distinct.pt" # Your .pt file with watermarks

# --- Main Training Script ---
def train_model():
    print(f"Using device: {DEVICE}")

    # 1. Dataset and DataLoader
    # Images will be resized, converted to tensor (scales to [0,1])
    # You can add T.Normalize if your model expects inputs in [-1,1] range
    # Make sure this matches how your original images were if you are comparing visual quality.
    train_transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(), # Scales images to [0,1]
        # T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Uncomment if you want [-1,1]
    ])

    try:
        train_dataset = ImageWatermarkDataset(
            image_dir=TRAIN_IMAGE_DIR,
            watermark_file_path=WATERMARK_FILE,
            transform=train_transform,
            watermark_dim=WATERMARK_DIM
        )
    except Exception as e:
        print(f"Failed to create dataset: {e}")
        return

    if len(train_dataset) == 0:
        print("Dataset is empty. Check image paths and watermark file.")
        return

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    print(f"DataLoader created with {len(train_dataloader)} batches.")

    # 2. Model Initialization
    model = UNetWatermark_Corrected(
        n_channels_in=N_CHANNELS_IN,
        n_channels_out=N_CHANNELS_OUT,
        watermark_dim=WATERMARK_DIM,
        bilinear=BILINEAR_UPSAMPLE,
        normalize_watermark=NORMALIZE_WATERMARK
    ).to(DEVICE)

    # 3. Loss Function
    criterion = nn.L1Loss() # MAE for image reconstruction. Values are [0,1] due to ToTensor and sigmoid.
    # criterion = nn.MSELoss()

    # 4. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 5. Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (images, watermarks) in enumerate(train_dataloader):
            images = images.to(DEVICE)
            watermarks = watermarks.to(DEVICE)

            optimizer.zero_grad()
            output_images = model(images, watermarks)
            loss = criterion(output_images, images)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_dataloader) :
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] completed. Avg Loss: {avg_epoch_loss:.4f}, Time: {epoch_time:.2f}s")

        # (Optional) Save model checkpoint periodically
        if (epoch + 1) % 10 == 0: # Save every 10 epochs
            chkpt_path = f"unet_watermark_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), chkpt_path)
            print(f"Checkpoint saved to {chkpt_path}")


    print("Training finished.")

    # 6. Save the final trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    # Make sure the paths are correct
    if not os.path.isdir(TRAIN_IMAGE_DIR):
        print(f"Error: Training image directory '{TRAIN_IMAGE_DIR}' not found.")
        exit()
    if not os.path.isfile(WATERMARK_FILE):
        print(f"Error: Watermark file '{WATERMARK_FILE}' not found.")
        exit()
        
    train_model()