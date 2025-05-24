import os
import torch
import random
import numpy as np

# Load the tensors

tensors = torch.load('/home/teaching/Desktop/Grp_22/dfWatermarking/datasetTrial1/TokenPatternsDataset/tensorsBinary.pt')
#convert to tensor

print(f"Loaded {len(tensors)} tensors of shape {len(tensors)}")

# Create lists for images and their corresponding tensor indices
all_images = []
dataset_dir = '/home/teaching/Desktop/Grp_22/dfWatermarking/datasetTrial1/TokenPatternsDataset/'

# Process each folder
for folder_idx in range(100):
    folder_path = os.path.join(dataset_dir, f'index{folder_idx}')
    print(folder_path)
    if os.path.exists(folder_path):
        for img_file in os.listdir(folder_path):
            if img_file.endswith('.png') or img_file.endswith('.jpg'):
                # Store full image path and tensor index
                all_images.append((os.path.join(folder_path, img_file), folder_idx))

# Shuffle the dataset
random.shuffle(all_images)

# Split into train/validation (90/10)
split_idx = int(len(all_images) * 0.9)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

print(f"Total images: {len(all_images)}, Train: {len(train_images)}, Val: {len(val_images)}")

# Create manifest files
train_manifest_path = "/home/teaching/Desktop/Grp_22/BiometricSEAL/videoseal/train_manifest.csv"
val_manifest_path = "/home/teaching/Desktop/Grp_22/BiometricSEAL/videoseal/val_manifest.csv"

with open(train_manifest_path, 'w') as f:
    for img_path, tensor_idx in train_images:
        tensor = tensors[tensor_idx]
        tensor_str = tensor
        f.write(f"{img_path},{tensor_str}\n")

with open(val_manifest_path, 'w') as f:
    for img_path, tensor_idx in val_images:
        tensor = tensors[tensor_idx]
        tensor_str = tensor
        f.write(f"{img_path},{tensor_str}\n")

print(f"Manifest files created: {train_manifest_path} and {val_manifest_path}")