import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class BinaryStringImageDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        """
        Dataset for loading images and their corresponding binary strings.
        
        Args:
            csv_path (str): Path to CSV file containing image paths and binary strings
            transform (callable, optional): Optional transform to apply to the images
        """
        self.data = pd.read_csv(csv_path, header=None)
        self.transform = transform
        
        # No header in CSV, so assuming format: image_path,binary_string
        self.image_paths = self.data[0].values
        self.binary_strings = self.data[1].values
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        
        # Convert binary string to tensor of floats (0s and 1s)
        binary_string = self.binary_strings[idx]
        binary_tensor = torch.tensor([int(bit) for bit in binary_string], dtype=torch.float32)
        
        return image, binary_tensor
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionTransformerBinary(nn.Module):
    def __init__(
        self,
        image_size=256,
        patch_size=16,
        in_channels=3,
        emb_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        binary_length=32,  # Length of binary string output
    ):
        """
        Vision Transformer for binary string prediction.
        
        Args:
            image_size (int): Input image size
            patch_size (int): Size of each patch
            in_channels (int): Number of input channels
            emb_dim (int): Embedding dimension
            depth (int): Number of transformer layers
            num_heads (int): Number of attention heads
            mlp_ratio (float): MLP hidden dim ratio
            dropout (float): Dropout rate
            binary_length (int): Length of binary string to predict
        """
        super(VisionTransformerBinary, self).__init__()
        
        # Calculate number of patches
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding layer
        self.patch_embed = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, emb_dim))
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=int(emb_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, depth)
        
        # Binary prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, binary_length)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize weights using truncated normal distribution
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_vit_weights)
    
    @staticmethod
    def _init_vit_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Convert [B, C, H, W] to patches [B, num_patches, emb_dim]
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, emb_dim]
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Use class token for prediction
        x = x[:, 0]
        
        # Apply prediction head
        x = self.head(x)
        
        # Apply sigmoid to get values between 0 and 1
        x = torch.sigmoid(x)
        
        return x