nohup torchrun --nproc_per_node=1 train.py --local_rank 0 \
        --image_dataset sa-1b --video_dataset none \
        --extractor_model convnext_tiny --embedder_model unet_small2_yuv_quant --hidden_size_multiplier 1 --nbits 128 \
        --scaling_w_schedule Cosine,scaling_min=0.2,epochs=500 --scaling_w 2.0 --scaling_i 1.0 --attenuation jnd_1_1 \
        --epochs 500 --iter_per_epoch 1000 --scheduler CosineLRScheduler,lr_min=1e-4,t_initial=601,warmup_lr_init=3e-4,warmup_t=20 --optimizer AdamW,lr=3e-4 \
        --lambda_dec 1.0 --lambda_d 0.05 --lambda_i 0 --perceptual_loss yuv  --num_augs 0 --disc_in_channels 1 --disc_start 50 --augmentation_config configs/no_augs.yaml --saveckpt_freq 1\
        > log.txt 2>&1 &



        #!/bin/bash

# --- Configuration ---
N_GPUS=1 # Number of GPUs per node
OUTPUT_DIR="DiffusionExtractor"
TRAIN_CSV="train_manifest.csv" # <--- CHANGE THIS
VAL_CSV="val_manifest.csv" # <--- CHANGE THIS
MODEL_CONFIG="configs/extractor_vit.yaml"
BATCH_SIZE=32 # Adjust based on GPU memory
EPOCHS=100
LR=1e-4 # Starting learning rate

# --- Ensure output directory exists ---
mkdir -p $OUTPUT_DIR

# --- Export dataset root if paths in CSV are relative ---
# export DATASET_ROOT="/path/to/image/root/directory"

# --- Run Training ---
# Use torchrun for compatibility with multi-gpu and single-gpu
# If using >1 GPU, adjust --nproc_per_node
# Set master_port if running multi-node or encountering port conflicts
torchrun --nproc_per_node=1 vit_extractor.py \
    --output_dir "DiffusionExtractor" \
    --model_config "configs/vit_extractor.yaml" \
    --train_csv "train_manifest.csv" \
    --val_csv "val_manifest.csv" \
    --batch_size 32 \
    --batch_size_eval $((32 * 2)) \
    --epochs 1 \
    --optimizer "AdamW,lr=3e-4,weight_decay=0.05" \
    --scheduler "CosineLRScheduler,lr_min=1e-4,t_initial=$((1 + 1)),warmup_lr_init=3e-4,warmup_t=5" \
    --img_size 224 \
    --workers 4 \
    --seed 42 \
    --eval_freq 5 \
    --saveckpt_freq 10 \
    --log_freq 20 \
    > "DiffusionExtractor/train_log.txt" 2>&1 & # Log output to file

echo "Training started. Output logged to $OUTPUT_DIR/train_log.txt"
echo "Tail log with: tail -f $OUTPUT_DIR/train_log.txt"
# Use `wait` if you want the script to block until training finishes
# wait