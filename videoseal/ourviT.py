import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class VisionTransformerRegressor(nn.Module):
    def __init__(
        self,
        image_size=256,
        patch_size=16,
        in_channels=3,
        emb_dim=768,
        depth=24,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        output_dim=128,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            in_channels, emb_dim, kernel_size=patch_size, stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, emb_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=int(emb_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth, norm=nn.LayerNorm(emb_dim)
        )

        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, output_dim),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
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
        print("Doing forward pass")
        B, C, H, W = x.shape
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.transformer(x)
        cls_out = x[:, 0]
        out = self.head(cls_out)
        return out


class ImageVectorDataset(Dataset):
    def __init__(self, label_file, tensor_file, transform=None):
        self.transform = transform
        self.image_paths = []
        self.indices = []

        # Load all tensors from file once
        self.all_vectors = torch.load(tensor_file)

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) >= 2:
                    image_path = parts[0]
                    index = int(parts[1])
                    self.image_paths.append(image_path)
                    self.indices.append(index)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        vector = self.all_vectors[self.indices[idx]]
        return image, vector



if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    label_file = "/home/teaching/Desktop/Grp_22/BiometricSEAL/videoseal/Dataset.txt"
    tensor_file = "/home/teaching/Desktop/Grp_22/BiometricSEAL/videoseal/tensors.pt"
    dataset = ImageVectorDataset(label_file, tensor_file, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTransformerRegressor().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()

    num_epochs = 10
    save_path = "vit_regressor_checkpoint.pth"

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, vectors in dataloader:
            images, vectors = images.to(device), vectors.to(device)
            preds = model(images)
            loss = criterion(preds, vectors)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(dataset)

        #  Print logs for the current epoch
        print("=" * 50)
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Average Loss: {avg_loss:.6f}")
        print("=" * 50)

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, save_path)

        print(f" Checkpoint saved to {save_path}\n")