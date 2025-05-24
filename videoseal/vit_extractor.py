import argparse
import datetime
import json
import os
import time
from typing import List, Tuple

import numpy as np
import omegaconf
import pandas as pd
from PIL import Image

import torch
import torch.distributed
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models as torchvision_models

# Assuming videoseal utils are available in the python path
# If not, you might need to copy/adapt relevant functions
import videoseal.utils as utils
import videoseal.utils.dist as udist
import videoseal.utils.logger as ulogger
import videoseal.utils.optim as uoptim
from videoseal.utils.tensorboard import CustomTensorboardWriter # Assuming this is available

# --- Configuration ---
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# --- Custom Dataset ---
class WatermarkedImageDataset(Dataset):
    """
    Dataset to load images and their corresponding binary messages from a CSV file.
    CSV format expected: column 'image_path', column 'message' (string of '0's and '1's)
    """
    def __init__(self, csv_file: str, img_transform=None, nbits: int = 32):
        self.data_frame = pd.read_csv(csv_file, header = None)
        self.data_frame.columns = ['image_path', 'message']
        self.img_transform = img_transform
        self.nbits = nbits
        print(f"Loaded dataset with {len(self.data_frame)} samples from {csv_file}")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data_frame.iloc[idx, self.data_frame.columns.get_loc('image_path')]
        # Make sure image path is absolute or relative to a known root
        if not os.path.isabs(img_path) and 'DATASET_ROOT' in os.environ:
             img_path = os.path.join(os.environ['DATASET_ROOT'], img_path)

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image not found at {img_path}")
            # Return dummy data or raise an error
            # For simplicity, returning dummy data here, but handling might need refinement
            image = Image.new('RGB', (224, 224), color = 'red') # Placeholder size
            message_str = '0' * self.nbits

        # Process message string ('0101...') into a float tensor
        message_str = str(self.data_frame.iloc[idx, self.data_frame.columns.get_loc('message')])
        if len(message_str) != self.nbits:
             raise ValueError(f"Message length mismatch in CSV row {idx}. Expected {self.nbits}, got {len(message_str)} ('{message_str}')")
        message = torch.tensor([float(bit) for bit in message_str], dtype=torch.float32)

        if self.img_transform:
            image = self.img_transform(image)

        return image, message

# --- Model Building ---
def build_extractor(model_name: str, config: omegaconf.DictConfig) -> nn.Module:
    """Builds a ViT extractor based on the config."""
    print(f"Building extractor: {model_name}")
    if not hasattr(torchvision_models, model_name):
        raise ValueError(f"Unknown torchvision model name: {model_name}")

    # Load pretrained model
    weights = torchvision_models.ViT_B_16_Weights.IMAGENET1K_V1 if config.pretrained else None
    model = getattr(torchvision_models, model_name)(weights=weights)

    # Modify the classification head
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, config.num_bits)
    print(f" -> Modified final layer to output {config.num_bits} features.")

    return model

# --- Metrics ---
@torch.no_grad()
def bit_accuracy(preds_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculates bit accuracy.
    Args:
        preds_logits: (batch_size, nbits) tensor of logits from the model.
        targets: (batch_size, nbits) tensor of target bits (0.0 or 1.0).
    Returns:
        Tensor: Scalar tensor with the mean bit accuracy.
    """
    preds_binary = (torch.sigmoid(preds_logits) > 0.5).float()
    correct = (preds_binary == targets).float()
    accuracy = correct.mean()
    return accuracy

# --- Argument Parser ---
def get_parser():
    parser = argparse.ArgumentParser("Extractor Training Script")

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Dataset parameters')
    aa("--train_csv", type=str, required=True, help="Path to the training CSV file (columns: image_path, message).")
    aa("--val_csv", type=str, required=True, help="Path to the validation CSV file (columns: image_path, message).")
    # Optionally add DATASET_ROOT if paths in CSV are relative
    # aa("--dataset_root", type=str, default="", help="Root directory for relative image paths in CSVs.")

    group = parser.add_argument_group('Experiments parameters')
    aa("--output_dir", type=str, default="output_extractor/", help="Output directory for logs and checkpoints.")

    group = parser.add_argument_group('Extractor configuration')
    aa("--model_config", type=str, default="configs/extractor_vit.yaml", help="Path to the extractor config file (YAML).")
    # Nbits is crucial and read from YAML, but can be added here for override/verification if needed
    # aa("--nbits", type=int, default=32, help="Number of bits in the message.")

    group = parser.add_argument_group('Training parameters')
    aa("--img_size", type=int, default=224, help="Size to resize input images to (must match model config).")
    aa("--batch_size", default=32, type=int, help='Batch size per GPU for training.')
    aa("--batch_size_eval", default=64, type=int, help='Batch size per GPU for evaluation.')
    aa("--epochs", default=100, type=int, help='Number of total epochs to run.')
    aa("--iter_per_epoch", default=None, type=int, help='Limit iterations per epoch (for large datasets). None means use full dataset.')
    aa("--optimizer", type=str, default="AdamW,lr=1e-4,weight_decay=0.05", help="Optimizer (default: AdamW,lr=1e-4,wd=0.05)")
    aa("--scheduler", type=str, default="CosineLRScheduler,lr_min=1e-5,t_initial=101,warmup_lr_init=1e-6,warmup_t=5", help="Scheduler (default: Cosine)")
    aa('--resume_from', default=None, type=str, help='Path to the checkpoint to resume from')
    aa('--resume_optimizer_state', type=utils.bool_inst, default=True, help='If True, also load optimizer state when resuming from checkpoint')
    aa("--loss", type=str, default="bce", choices=["bce"], help="Loss function for bit prediction.")


    group = parser.add_argument_group('Misc.')
    aa('--workers', default=4, type=int, help='Number of data loading workers')
    aa('--only_eval', type=utils.bool_inst, default=False, help='If True, only runs evaluate')
    aa('--eval_freq', default=5, type=int, help='Frequency (epochs) for evaluation')
    aa('--saveckpt_freq', default=5, type=int, help='Frequency (epochs) for saving checkpoints')
    aa('--seed', default=42, type=int, help='Random seed')
    aa('--log_freq', default=10, type=int, help='Logging frequency (iterations)')

    group = parser.add_argument_group('Distributed training parameters')
    aa('--debug_slurm', action='store_true')
    aa('--local_rank', default=-1, type=int)
    aa('--master_port', default=-1, type=int)

    return parser


# --- Training Loop ---
def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    params: omegaconf.DictConfig,
    tensorboard: CustomTensorboardWriter
) -> dict:
    model.train()
    metric_logger = ulogger.MetricLogger(delimiter="  ")
    header = f'Train - Epoch: [{epoch}/{params.epochs}]'
    print_freq = params.log_freq

    for it, (imgs, msgs) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
        if params.iter_per_epoch is not None and it >= params.iter_per_epoch:
            break

        imgs = imgs.to(device, non_blocking=True)
        msgs = msgs.to(device, non_blocking=True) # Target bits (0.0 or 1.0)

        # Forward
        optimizer.zero_grad()
        preds_logits = model(imgs) # Model outputs logits

        # Loss calculation
        loss = criterion(preds_logits, msgs)

        # Backward
        loss.backward()
        optimizer.step()

        # Log stats
        torch.cuda.synchronize()
        batch_acc = bit_accuracy(preds_logits, msgs).item()
        metric_logger.update(loss=loss.item(), bit_acc=batch_acc, lr=optimizer.param_groups[0]['lr'])

        # Add to TensorBoard more frequently if needed
        # if it % (print_freq // 2) == 0 and udist.is_main_process():
        #     step = epoch * (params.iter_per_epoch or len(train_loader)) + it
        #     tensorboard.add_scalar("TRAIN/iter_loss", loss.item(), step)
        #     tensorboard.add_scalar("TRAIN/iter_bit_acc", batch_acc, step)

    # Gather stats across all processes for logging
    metric_logger.synchronize_between_processes()
    print("Averaged train stats:", metric_logger)
    train_logs = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if udist.is_main_process():
         tensorboard.add_scalars("TRAIN/EPOCH", train_logs, epoch)

    return train_logs

# --- Evaluation Loop ---
@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    params: omegaconf.DictConfig,
    tensorboard: CustomTensorboardWriter,
) -> dict:
    model.eval()
    metric_logger = ulogger.MetricLogger(delimiter="  ")
    header = f'Eval - Epoch: [{epoch}/{params.epochs}]'
    print_freq = params.log_freq

    for it, (imgs, msgs) in enumerate(metric_logger.log_every(val_loader, print_freq, header)):
        # Limit iterations for quick eval if needed (params.iter_per_valid not implemented, use iter_per_epoch logic if desired)
        # if params.iter_per_valid is not None and it >= params.iter_per_valid:
        #     break

        imgs = imgs.to(device, non_blocking=True)
        msgs = msgs.to(device, non_blocking=True) # Target bits (0.0 or 1.0)

        # Forward
        eval_time = time.time()
        preds_logits = model(imgs) # Model outputs logits
        eval_time = (time.time() - eval_time) / imgs.shape[0]

        # Loss calculation
        loss = criterion(preds_logits, msgs)

        # Log stats
        torch.cuda.synchronize()
        batch_acc = bit_accuracy(preds_logits, msgs).item()
        metric_logger.update(loss=loss.item(), bit_acc=batch_acc, eval_time=eval_time)

    # Gather stats across all processes
    metric_logger.synchronize_between_processes()
    print("Averaged validation stats:", metric_logger)
    valid_logs = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if udist.is_main_process():
        tensorboard.add_scalars("VALID", valid_logs, epoch)
        # Optionally save first batch images/preds for visualization
        if epoch % params.saveckpt_freq == 0 and it == 0: # Save on first batch of eval on save freq
             # Can add logic here to save `imgs` and visualize `preds_logits` (e.g., as heatmap or thresholded bits)
             pass

    return valid_logs

# --- Main Function ---
def main(params):
    # Setup TensorBoard
    tensorboard = CustomTensorboardWriter(log_dir=os.path.join(params.output_dir, "tensorboard"))

    # Convert params to OmegaConf object (useful for dot notation access)
    params = omegaconf.OmegaConf.create(vars(params))
    # If dataset_root is provided, set environment variable for dataset class
    # if params.dataset_root:
    #     os.environ['DATASET_ROOT'] = params.dataset_root

    # Distributed mode setup
    udist.init_distributed_mode(params)

    # Set seeds
    seed = params.seed + udist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Deterministic training might slow down, use cautiously
    # if params.distributed:
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    # Log parameters
    print("__git__:{}".format(utils.get_sha())) # Assumes utils.get_sha() exists
    yaml_params = omegaconf.OmegaConf.to_yaml(params)
    print("__log__:\n{}".format(yaml_params))
    if udist.is_main_process():
        os.makedirs(params.output_dir, exist_ok=True)
        with open(os.path.join(params.output_dir, 'config.yaml'), 'w') as f:
             f.write(yaml_params)
        # Copy model config
        os.makedirs(os.path.join(params.output_dir, 'configs'), exist_ok=True)
        try:
            os.system(f'cp {params.model_config} {params.output_dir}/configs/extractor_model.yaml')
        except Exception as e:
            print(f"Warning: Could not copy model config: {e}")


    # Build the extractor model
    extractor_cfg_full = omegaconf.OmegaConf.load(params.model_config)
    # Assuming the actual config is under a key like 'vit_extractor' or similar
    # Find the key dynamically or adjust this line if structure is fixed
    model_key = list(extractor_cfg_full.keys())[-1] # Heuristic: take the last key
    if 'model_name' not in extractor_cfg_full:
         raise KeyError(f"'model_name' not found in {params.model_config}")
    extractor_cfg = extractor_cfg_full[model_key]
    model_name = extractor_cfg_full.model_name
    extractor = build_extractor(model_name, extractor_cfg)
    extractor = extractor.to(device)
    params.nbits = extractor_cfg.num_bits # Ensure nbits is consistent
    print(f'Extractor ({model_name}): {sum(p.numel() for p in extractor.parameters() if p.requires_grad) / 1e6:.2f}M parameters')

    # Define Loss function
    if params.loss == 'bce':
        # Numerically stable version, expects raw logits from the model
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        raise ValueError(f"Unsupported loss: {params.loss}")
    print(f"Using loss: {params.loss.upper()}")

    # Build optimizer and scheduler
    optim_params_dict = uoptim.parse_params(params.optimizer)
    optimizer = uoptim.build_optimizer(extractor.parameters(), **optim_params_dict)
    scheduler_params_dict = uoptim.parse_params(params.scheduler)
    # Adjust scheduler 't_initial' based on epochs if needed
    if 't_initial' in scheduler_params_dict and scheduler_params_dict['t_initial'] != params.epochs + 1:
         print(f"Warning: Scheduler t_initial ({scheduler_params_dict['t_initial']}) doesn't match epochs+1 ({params.epochs+1}). Adjusting...")
         scheduler_params_dict['t_initial'] = params.epochs + 1
         # Similar logic for warmup_t if needed based on iter_per_epoch
    scheduler = uoptim.build_lr_scheduler(optimizer, **scheduler_params_dict)
    print('Optimizer:', optimizer)
    print('Scheduler:', scheduler)

    # Data loading setup
    # Basic transforms - adjust normalization if needed based on model/data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) # ImageNet stats
    train_transform = transforms.Compose([
        transforms.Resize((params.img_size, params.img_size)), # Non-random resize for consistency
        # Add augmentations here if desired (e.g., RandomHorizontalFlip)
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((params.img_size, params.img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = WatermarkedImageDataset(params.train_csv, img_transform=train_transform, nbits=params.nbits)
    val_dataset = WatermarkedImageDataset(params.val_csv, img_transform=val_transform, nbits=params.nbits)

    # Create distributed samplers if needed
    if params.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=params.batch_size,
                              sampler=train_sampler, num_workers=params.workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=params.batch_size_eval,
                            sampler=val_sampler, num_workers=params.workers,
                            pin_memory=True, drop_last=False)
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")


    # Checkpoint Resuming
    start_epoch = 0
    if params.resume_from:
        print(f"Resuming from checkpoint: {params.resume_from}")
        components_to_load = {'model': extractor}
        if params.resume_optimizer_state:
            components_to_load['optimizer'] = optimizer
            components_to_load['scheduler'] = scheduler # Assuming scheduler state is saved
        to_restore = {'epoch': 0}
        uoptim.restart_from_checkpoint(
            params.resume_from,
            run_variables=to_restore,
            **components_to_load
        )
        start_epoch = to_restore["epoch"]
        print(f" -> Resumed epoch: {start_epoch}")

    # Wrap model with DDP *after* loading checkpoint and *before* passing to optimizer/training loop
    if params.distributed:
        extractor = nn.parallel.DistributedDataParallel(extractor, device_ids=[params.local_rank], find_unused_parameters=False) # Ensure find_unused_parameters is False if model is simple
        model_without_ddp = extractor.module
    else:
        model_without_ddp = extractor # Alias for saving state_dict

    # Evaluation only mode
    if params.only_eval:
        print("Running evaluation only...")
        val_stats = eval_one_epoch(extractor, val_loader, criterion, start_epoch, params, tensorboard)
        log_stats = {'epoch': start_epoch, **{f'val_{k}': v for k, v in val_stats.items()}}
        if udist.is_main_process():
            with open(os.path.join(params.output_dir, 'log_eval_only.txt'), 'w') as f:
                f.write(json.dumps(log_stats) + "\n")
        print("Evaluation finished.")
        return

    # Training loop
    print('Starting training...')
    start_time = time.time()
    for epoch in range(start_epoch, params.epochs):

        if params.distributed:
            train_loader.sampler.set_epoch(epoch) # Important for shuffling

        # Train one epoch
        train_stats = train_one_epoch(extractor, optimizer, train_loader, criterion, epoch, params, tensorboard)

        # Step the scheduler
        if scheduler is not None:
            scheduler.step(epoch + 1) # CosineLRScheduler expects epoch+1

        # Log training stats
        log_stats = {'epoch': epoch, **{f'train_{k}': v for k, v in train_stats.items()}}

        # Evaluate
        if (epoch + 1) % params.eval_freq == 0 or epoch == params.epochs - 1:
             val_stats = eval_one_epoch(extractor, val_loader, criterion, epoch, params, tensorboard)
             log_stats = {**log_stats, **{f'val_{k}': v for k, v in val_stats.items()}}
             print(f"Epoch {epoch} Val Bit Accuracy: {val_stats.get('bit_acc', 'N/A'):.4f}")

        # Save checkpoint
        if (epoch + 1) % params.saveckpt_freq == 0 or epoch == params.epochs - 1:
            save_dict = {
                'epoch': epoch + 1,
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'args': omegaconf.OmegaConf.to_container(params), # Save args as dict
                'model_config': omegaconf.OmegaConf.to_container(extractor_cfg_full) # Save model config
            }
            chkpt_path = os.path.join(params.output_dir, f'checkpoint{epoch:03}.pth')
            latest_path = os.path.join(params.output_dir, 'checkpoint_latest.pth')
            udist.save_on_master(save_dict, chkpt_path)
            udist.save_on_master(save_dict, latest_path) # Overwrite latest
            print(f"Checkpoint saved to {chkpt_path} and {latest_path}")

        # Write logs (append mode)
        if udist.is_main_process():
            with open(os.path.join(params.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + "\n")

        # Barrier if distributed to sync before next epoch
        if udist.is_dist_avail_and_initialized():
            dist.barrier()

    # End of training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Total training time: {total_time_str}')
    tensorboard.close()


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    main(params)