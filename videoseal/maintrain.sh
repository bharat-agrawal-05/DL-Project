nohup torchrun --nproc_per_node=1 train.py --local_rank 0 \
        --image_dataset sa-1b --video_dataset none \
        --extractor_model convnext_tiny --embedder_model unet_small2_yuv_quant --hidden_size_multiplier 1 --nbits 128 \
        --scaling_w_schedule Cosine,scaling_min=0.2,epochs=500 --scaling_w 2.0 --scaling_i 1.0 --attenuation jnd_1_1 \
        --epochs 500 --iter_per_epoch 1000 --scheduler CosineLRScheduler,lr_min=1e-4,t_initial=601,warmup_lr_init=3e-4,warmup_t=20 --optimizer AdamW,lr=3e-4 \
        --lambda_dec 1.0 --lambda_d 0.05 --lambda_i 0 --perceptual_loss yuv  --num_augs 0 --disc_in_channels 1 --disc_start 50 --augmentation_config configs/no_augs.yaml --saveckpt_freq 1\
        > log.txt 2>&1 &