import os
# Fix for torchvision::nms error
os.environ["TORCH_DISPATCH_TO_FAKE"] = "0"

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import time

from newVit import BinaryStringImageDataset, VisionTransformerBinary

def print_gpu_info():
    """Print GPU information if available"""
    if torch.cuda.is_available():
        print(f"\n==== GPU INFORMATION ====")
        print(f"CUDA is available! Found {torch.cuda.device_count()} device(s).")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  CUDA Capability: {props.major}.{props.minor}")
        print("========================\n")
    else:
        print("\nWARNING: CUDA is NOT available. Training will be slow on CPU.\n")

def binary_accuracy(pred, target, threshold=0.5):
    """Calculate binary prediction accuracy"""
    binary_pred = (pred >= threshold).float()
    correct = (binary_pred == target).float().sum()
    return correct / target.numel()

def bit_accuracy(pred, target, threshold=0.5):
    """Calculate per-bit accuracy"""
    binary_pred = (pred >= threshold).float()
    correct_per_bit = (binary_pred == target).float().mean(dim=0)
    return correct_per_bit

def train_model(args):
    # Verify GPU availability
    print_gpu_info()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load datasets
    print(f"Loading datasets from {args.train_csv} and {args.val_csv}")
    train_dataset = BinaryStringImageDataset(args.train_csv, transform=transform)
    val_dataset = BinaryStringImageDataset(args.val_csv, transform=transform)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Get binary string length from data
    _, sample_binary = train_dataset[0]
    binary_length = len(sample_binary)
    print(f"Binary string length: {binary_length}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    # Initialize model
    model = VisionTransformerBinary(
        image_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=3,
        emb_dim=args.emb_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        binary_length=binary_length
    ).to(device)
    
    # Print model size
    model_size = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model has {model_size:.2f} million parameters")
    
    # Setup mixed precision training
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    if use_amp:
        print("Using mixed precision training")
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.output_dir, "training_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_size:.2f}M parameters, emb_dim={args.emb_dim}, depth={args.depth}, heads={args.num_heads}\n")
        f.write(f"Using device: {device}, AMP: {use_amp}\n")
        f.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc,Time(s)\n")
    
    best_val_acc = 0.0
    
    # Print initial GPU memory usage
    if torch.cuda.is_available():
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
    
    # Training loop
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, (images, targets) in enumerate(pbar):
            # Move data to device
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass with mixed precision if enabled
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                
                # Backward pass with gradient scaling
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                scaler.scale(loss).backward()
                
                # Gradient clipping to prevent exploding gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular training path (CPU or if not using mixed precision)
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # Update metrics
            batch_acc = binary_accuracy(outputs, targets)
            train_loss += loss.item() * images.size(0)
            train_acc += batch_acc * images.size(0)
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "acc": f"{batch_acc.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Optional: Print GPU memory usage every 50 steps
            if step % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()  # This helps free unused memory
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_acc / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        all_bit_accs = []
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * images.size(0)
                val_acc += binary_accuracy(outputs, targets) * images.size(0)
                
                # Calculate per-bit accuracy
                bit_accs = bit_accuracy(outputs, targets)
                all_bit_accs.append(bit_accs)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_acc / len(val_loader.dataset)
        
        # Compute average bit accuracies
        avg_bit_accs = torch.stack(all_bit_accs).mean(dim=0)
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print metrics
        print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Avg bit-wise accuracy: {avg_bit_accs.mean().item():.4f}")
        print(f"  Worst bit accuracy: {avg_bit_accs.min().item():.4f} (bit {avg_bit_accs.argmin().item()})")
        print(f"  Best bit accuracy: {avg_bit_accs.max().item():.4f} (bit {avg_bit_accs.argmax().item()})")
        
        # Print GPU memory usage
        if torch.cuda.is_available():
            print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
            # Empty cache to prevent memory fragmentation
            torch.cuda.empty_cache()
        
        # Log to file
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{epoch_time:.2f}\n")
        
        # Save checkpoint if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'bit_accuracies': avg_bit_accs.cpu().numpy().tolist()
            }, checkpoint_path)
            print(f"  Saved best model checkpoint to {checkpoint_path}")
        
        # Save regular checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vision Transformer for binary string prediction")
    
    # Dataset and model parameters
    parser.add_argument("--train_csv", type=str, default="train_manifest.csv", help="Path to training CSV")
    parser.add_argument("--val_csv", type=str, default="val_manifest.csv", help="Path to validation CSV")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--emb_dim", type=int, default=384, help="Embedding dimension (384 for smaller GPUs, 768 for larger)")
    parser.add_argument("--depth", type=int, default=6, help="Transformer depth (6 for smaller GPUs, 12 for larger)")
    parser.add_argument("--num_heads", type=int, default=6, help="Number of attention heads (6 for smaller GPUs, 12 for larger)")
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="MLP ratio")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (reduce to 16 or 8 if OOM errors)")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of data loader workers")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision training")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="binary_vit_output", help="Output directory")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    train_model(args)