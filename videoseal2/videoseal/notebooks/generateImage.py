import torch
from torch import nn
from SD.models import set_seed
import torchvision
from tqdm import tqdm
import os
import numpy as np # Added for binary message
import torch.nn.functional as F
from SD.models import setup_accelerator_and_logging, load_models_and_tokenizer \
                        , prepare_uncond_embeddings
from SD.helper import generate_class_prompts, precompute_text_embeddings

# Import for VideoSeal
from videoseal.evals.full import setup_model_from_checkpoint
from videoseal.utils.display import save_img # if needed for specific saving

# ========= VAE Class Definition (from your training script) ==========
# This VAE class is kept if you might want to switch between VAE DGM and VideoSeal DGM,
# or if other parts of your project still use it.
# For this specific request, it won't be directly used in the VideoSeal guidance path.
class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_space_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.hidden_to_miu = nn.Linear(256 * 4 * 4, latent_space_dim)
        self.hidden_to_sigma = nn.Linear(256 * 4 * 4, latent_space_dim)
        self.latent_to_hidden = nn.Linear(latent_space_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        miu = self.hidden_to_miu(h)
        sigma_logvar = self.hidden_to_sigma(h)
        temp_z_for_recon = miu
        h_decode = self.latent_to_hidden(temp_z_for_recon)
        h_decode = h_decode.view(h_decode.size(0), 256, 4, 4)
        recon_x = self.decoder(h_decode)
        return recon_x, miu, sigma_logvar
# =====================================================================


def generate_images_WIP(accelerator, class_index, class_text_embeddings, uncond_embeddings,
                      noise_scheduler, sd_vae_decoder, unet,
                      # teacher_vae_model, # Replaced by videoseal_model
                      videoseal_model, target_binary_message_tensor, # New parameters
                      generator, weight_dtype, syn_image_seed, config):
    image_save_dir_path = os.path.join(config.save_syn_data_path, str(class_index))
    os.makedirs(image_save_dir_path, exist_ok=True)
    text_embeddings = class_text_embeddings[class_index]
    syn_image_seed += 1
    generator.manual_seed(syn_image_seed)
    set_seed(syn_image_seed)

    # Trackers for VideoSeal guidance
    loss_watermark_tracker = torch.tensor(0.0, device=accelerator.device)

    torch.cuda.empty_cache()
    with accelerator.accumulate(unet):
        latents_shape = (config.batch_size_generation, unet.config.in_channels, 64, 64)
        latents = torch.randn(
            latents_shape, generator=generator, device="cpu", dtype=weight_dtype
        ).to(accelerator.device)
        latents = latents * noise_scheduler.init_noise_sigma

        noise_scheduler.set_timesteps(config.inference_nums)
        timesteps_tensor = noise_scheduler.timesteps.to(accelerator.device)

        print(f"\nStarting generation for class {class_index}, seed {syn_image_seed}. Latents dtype: {latents.dtype}")

        for timestep_idx, timesteps in enumerate(tqdm(timesteps_tensor[:-1], desc=f"Class {class_index} Steps")):
            input_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)
            latent_model_input = torch.cat([latents] * 2)
            noise_pred_unet = unet(latent_model_input, timesteps, input_embeddings).sample
            uncond_noise_pred_unet, text_noise_pred_unet = noise_pred_unet.chunk(2)
            guided_noise_pred_cfg = (uncond_noise_pred_unet + \
                                     config.guided_scale * (text_noise_pred_unet - uncond_noise_pred_unet))
            final_noise_pred_for_step = guided_noise_pred_cfg.clone()

            # --- VideoSeal DGM Guidance ---
            apply_dgm_guidance = (config.watermark_guidance_weight != 0) and \
                                 (timestep_idx >= (config.inference_nums * config.skip_dgm_initial_ratio))
            image_512_pred_x0_dgm = None

            if apply_dgm_guidance:
                current_noise_pred_for_loss_float = guided_noise_pred_cfg.float().clone().requires_grad_()
                alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps].to(latents.device)
                sqrt_alpha_prod_t = alpha_prod_t.sqrt().float()
                sqrt_one_minus_alpha_prod_t = (1 - alpha_prod_t).sqrt().float()
                if sqrt_alpha_prod_t.ndim == 0:
                    sqrt_alpha_prod_t = sqrt_alpha_prod_t.view(1,1,1,1)
                    sqrt_one_minus_alpha_prod_t = sqrt_one_minus_alpha_prod_t.view(1,1,1,1)

                pred_original_sample_float = (latents.detach().float() - \
                                        sqrt_one_minus_alpha_prod_t * current_noise_pred_for_loss_float) / \
                                       sqrt_alpha_prod_t
                
                # Decode to 512x512 image using SD VAE
                input_latents_for_sd_vae = (pred_original_sample_float / config.sd_vae_scale_factor).to(sd_vae_decoder.dtype)
                image_from_sd = sd_vae_decoder.decode(input_latents_for_sd_vae).sample
                
                # Normalize image to [0, 1] for VideoSeal detector
                image_for_videoseal_detector = (image_from_sd.float() / 2 + 0.5).clamp(0, 1)
                image_512_pred_x0_dgm = image_for_videoseal_detector # For potential saving

                if timestep_idx % config.debug_print_freq == 0:
                    print(f"\n  T{timesteps.item()}: image_for_videoseal_detector dtype: {image_for_videoseal_detector.dtype}, "
                          f"min/max: {image_for_videoseal_detector.min().item():.3f}/{image_for_videoseal_detector.max().item():.3f}, "
                          f"shape: {image_for_videoseal_detector.shape}")

                # VideoSeal detection
                # Ensure videoseal_model is on the correct device (should be done in main)
                with torch.no_grad(): # Detection itself shouldn't require grads for the detector
                    videoseal_outputs = videoseal_model.detect(image_for_videoseal_detector, is_video=False)
                
                detected_logits = videoseal_outputs["preds"] # Shape: (batch_size, num_bits + 1)
                
                # We need to enable gradients for detected_logits IF they were produced by a PyTorch module
                # that was part of the graph starting from current_noise_pred_for_loss_float.
                # However, videoseal_model.detect is likely a complex function.
                # The most straightforward way is to re-run detection with grad enabled for the image input
                # *if* the videoseal model's layers are differentiable and on the same graph.
                # For now, let's assume `detect` can be part of the graph if input requires grad.
                # A safer approach:
                # Re-evaluate image_for_videoseal_detector with requires_grad=True if its calculation
                # depends on current_noise_pred_for_loss_float. It does.
                # So, we need to ensure the graph is connected.
                # The `image_from_sd` comes from `sd_vae_decoder` which itself is a nn.Module.
                # `pred_original_sample_float` depends on `current_noise_pred_for_loss_float`.
                # So the graph *should* be connected.

                # Extract message bits logits (excluding the first confidence score if present)
                message_logits = detected_logits[:, 1:] # Assuming first is confidence, rest are bits

                if timestep_idx % config.debug_print_freq == 0:
                    print(f"  T{timesteps.item()}: message_logits shape: {message_logits.shape}, "
                          f"target_binary_message_tensor shape: {target_binary_message_tensor.shape}")
                    # Check min/max of logits
                    print(f"  T{timesteps.item()}: message_logits min/max/mean: {message_logits.min().item():.3f}/{message_logits.max().item():.3f}/{message_logits.mean().item():.3f}")


                # Expand target_binary_message_tensor to match batch size if needed
                current_batch_size = message_logits.shape[0]
                if target_binary_message_tensor.shape[0] != current_batch_size:
                    expanded_target_message = target_binary_message_tensor.repeat(current_batch_size, 1)
                else:
                    expanded_target_message = target_binary_message_tensor

                # Ensure target message is float for BCEWithLogitsLoss
                expanded_target_message = expanded_target_message.float()

                # Calculate BCE loss with logits
                # The `preds` from videoseal are logits, so use BCEWithLogitsLoss
                loss_watermark_val = F.binary_cross_entropy_with_logits(
                    message_logits,
                    expanded_target_message,
                    reduction="sum" # Sum over all bits and batch items
                )
                loss_watermark_tracker = loss_watermark_val.detach()

                total_dgm_loss = loss_watermark_val * config.watermark_guidance_weight

                if timestep_idx % config.debug_print_freq == 0:
                    print(f"  T{timesteps.item()}: loss_watermark={loss_watermark_val.item():.3e}, total_dgm_loss={total_dgm_loss.item():.3e}")

                # Get gradients
                grad_guided = torch.autograd.grad(total_dgm_loss, current_noise_pred_for_loss_float, allow_unused=True)[0]

                if grad_guided is not None:
                    if timestep_idx % config.debug_print_freq == 0:
                        print(f"  T{timesteps.item()}: grad_guided L2_norm (before clip): {grad_guided.norm().item():.3e}")
                        print(f"  T{timesteps.item()}: grad_guided_mean_abs={grad_guided.abs().mean().item():.3e}")
                    grad_guided = torch.clamp(grad_guided, -config.grad_clip, config.grad_clip)
                    final_noise_pred_for_step = (current_noise_pred_for_loss_float - grad_guided * config.lr_dgm).to(guided_noise_pred_cfg.dtype)
                else:
                    if timestep_idx % config.debug_print_freq == 0: print(f"  T{timesteps.item()}: grad_guided is None! Ensure videoseal_model.detect() is differentiable or graph is connected.")
                    # If grad is None, it means the loss is not connected to current_noise_pred_for_loss_float.
                    # This could happen if videoseal_model.detect internally detaches tensors or uses non-differentiable ops.
                    # Forcing videoseal_model to run with torch.enable_grad() context might be needed if its internal modules use torch.no_grad().
                    # One test: ensure videoseal_model parameters have requires_grad=False if it's pre-trained and fixed.

            torch.cuda.empty_cache()

            # Save intermediate images
            if (timestep_idx + 1) % config.save_intermediate_every_n_steps == 0 : # Use new config param
                if apply_dgm_guidance and image_512_pred_x0_dgm is not None:
                    image_name_512_dgm = os.path.join(image_save_dir_path,
                        f"{syn_image_seed}_{class_index}_s{timesteps.item():04d}_videoseal_pred_x0_lossW{loss_watermark_tracker.item():.2e}.jpg")
                    torchvision.utils.save_image(image_512_pred_x0_dgm.detach().cpu(), image_name_512_dgm)
                else:
                    with torch.no_grad():
                        alpha_prod_t_cfg = noise_scheduler.alphas_cumprod[timesteps].to(latents.device).float()
                        sqrt_alpha_prod_t_cfg = alpha_prod_t_cfg.sqrt()
                        sqrt_one_minus_alpha_prod_t_cfg = (1 - alpha_prod_t_cfg).sqrt()
                        if sqrt_alpha_prod_t_cfg.ndim == 0:
                            sqrt_alpha_prod_t_cfg = sqrt_alpha_prod_t_cfg.view(1,1,1,1)
                            sqrt_one_minus_alpha_prod_t_cfg = sqrt_one_minus_alpha_prod_t_cfg.view(1,1,1,1)

                        pred_original_sample_cfg = (latents.detach().float() - sqrt_one_minus_alpha_prod_t_cfg * final_noise_pred_for_step.float()) / sqrt_alpha_prod_t_cfg
                        input_latents_for_sd_vae_cfg = (pred_original_sample_cfg / config.sd_vae_scale_factor).to(sd_vae_decoder.dtype)
                        image_cfg_decoded = sd_vae_decoder.decode(input_latents_for_sd_vae_cfg).sample
                        image_to_save_intermediate_cfg = (image_cfg_decoded.float() / 2 + 0.5).clamp(0, 1)
                        
                        save_tag = "cfg_only"
                        image_name_cfg = os.path.join(image_save_dir_path,
                            f"{syn_image_seed}_{class_index}_s{timesteps.item():04d}_{save_tag}.jpg")
                        torchvision.utils.save_image(image_to_save_intermediate_cfg.detach().cpu(), image_name_cfg)
            
            with torch.no_grad():
                scheduler_output_step = noise_scheduler.step(
                    final_noise_pred_for_step.to(unet.dtype),
                    timesteps.cpu(),
                    latents.detach(),
                    generator=generator,
                    return_dict=True
                )
                latents = scheduler_output_step.prev_sample.to(weight_dtype)

            if unet.training: unet.zero_grad(set_to_none=True)
            if sd_vae_decoder.training: sd_vae_decoder.zero_grad(set_to_none=True)
            # if videoseal_model.training: videoseal_model.zero_grad(set_to_none=True) # Should be eval

            if 'current_noise_pred_for_loss_float' in locals() and hasattr(current_noise_pred_for_loss_float, 'grad') and current_noise_pred_for_loss_float.grad is not None:
                current_noise_pred_for_loss_float.grad.zero_()
            del current_noise_pred_for_loss_float # Free memory

            torch.cuda.empty_cache()

        # Save final images
        with torch.no_grad():
            final_loop_timestep = timesteps_tensor[-1].to(accelerator.device)
            final_input_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)
            final_latent_model_input = torch.cat([latents.to(device=accelerator.device, dtype=unet.dtype)] * 2)

            final_noise_preds_unet = unet(final_latent_model_input, final_loop_timestep, final_input_embeddings).sample
            final_uncond_pred, final_text_pred = final_noise_preds_unet.chunk(2)
            final_model_pred_for_saving = final_uncond_pred + config.guided_scale * (final_text_pred - final_uncond_pred)
            
            scheduler_output_final = noise_scheduler.step(
                final_model_pred_for_saving.to(unet.dtype),
                final_loop_timestep.cpu(),
                latents.to(unet.dtype),
                generator=generator,
                return_dict=True
            )
            ori_latents_final = scheduler_output_final.pred_original_sample
            
            input_latents_final_decode = (ori_latents_final / config.sd_vae_scale_factor).to(sd_vae_decoder.dtype)
            image_final_decoded = sd_vae_decoder.decode(input_latents_final_decode).sample
            image_final_save_scaled = (image_final_decoded.float() / 2 + 0.5).clamp(0, 1)
            
            for i in range(config.batch_size_generation):
                loss_w_val_fn = loss_watermark_tracker.item()
                image_name = os.path.join(image_save_dir_path,
                    f"{syn_image_seed}_{class_index}_final_lossW{loss_w_val_fn:.2e}_{i}.jpg")
                torchvision.utils.save_image(image_final_save_scaled[i].detach().cpu(), image_name)
                print(f"Saved final image: {image_name} (Size: {image_final_save_scaled[i].shape})")

    torch.cuda.empty_cache()
    return syn_image_seed


def load_teacher_vae_model(trained_dgm_path, device): # Kept for compatibility if needed elsewhere
    model = VAE()
    model.load_state_dict(torch.load(trained_dgm_path, map_location=device))
    model.eval()
    return model


class Config:
    def __init__(self):
        self.SEED = 5000
        self.name = "synthetic_cifar_videoseal"
        self.save_syn_data_path = "./synthetic_cifar_videoseal"
        self.checkpoints_dir = "./checkpoints"
        self.generate_nums = 10 # Reduced for faster testing
        self.batch_size_generation = 1 # For simplicity with VideoSeal message handling first
        self.STABLE_DIFFUSION = "runwayml/stable-diffusion-v1-5"
        self.SD_REVISION = None
        self.inference_nums = 10 # Diffusion steps
        self.guided_scale = 5.0 # CFG scale
        
        # DGM Guidance (now VideoSeal based)
        self.lr_dgm = 0.1 # Learning rate for DGM update step (might need tuning)
        self.watermark_guidance_weight = 1.0 # Weight for the VideoSeal BCE loss (tune this)
        self.skip_dgm_initial_ratio = 0.1
        self.grad_clip = 1.0
        
        self.sd_vae_scale_factor = 0.18215
        # self.trained_dgm_path = "./VAE/vae_model.pth" # Path to old VAE, not used for VideoSeal guidance
        
        # VideoSeal specific
        self.videoseal_ckpt_path = 'videoseal' # As per your notebook (e.g., 'videoseal' or full path)
        self.target_watermark_message_str = "10" * 64 # 128 bits: 101010...
        self.num_watermark_bits = 128 # Should match the length of target_watermark_message_str

        self.label_name = True
        self.data_type = "cifar10" # For prompts
        self.save_intermediate_every_n_steps = 2 # Modified from original logic for clarity
        self.debug_print_freq = 1 # Print more frequently for debugging DGM

def main(config:Config):
    accelerator = setup_accelerator_and_logging(config.SEED)
    
    tokenizer, text_encoder, noise_scheduler, sd_vae_decoder, unet, _, _, generator, weight_dtype = \
        load_models_and_tokenizer(accelerator, config.STABLE_DIFFUSION, config.SD_REVISION, config.SEED)
    
    print(f"Loaded SD models. UNet dtype: {unet.dtype}, SD VAE decoder dtype: {sd_vae_decoder.dtype}, "
          f"Text Encoder dtype: {text_encoder.dtype}, weight_dtype for latents: {weight_dtype}")

    uncond_embeddings = prepare_uncond_embeddings(tokenizer, text_encoder, unet, config.batch_size_generation)
    
    # --- Load VideoSeal Model ---
    print(f"Loading VideoSeal model from checkpoint: {config.videoseal_ckpt_path}...")
    try:
        videoseal_model = setup_model_from_checkpoint(config.videoseal_ckpt_path)
        videoseal_model.eval()
        # videoseal_model.compile() # Optional: can speed up but might be slow initially or have issues
        videoseal_model.to(accelerator.device)
        print(f"VideoSeal model loaded and moved to {accelerator.device}.")
        # Freeze VideoSeal model parameters
        for param in videoseal_model.parameters():
            param.requires_grad = False
    except Exception as e:
        print(f"Error loading VideoSeal model: {e}")
        print("Proceeding without VideoSeal guidance. Ensure config.watermark_guidance_weight is 0.")
        videoseal_model = None
        config.watermark_guidance_weight = 0 # Disable guidance if model fails to load

    # --- Prepare Target Binary Message ---
    if videoseal_model is not None and len(config.target_watermark_message_str) != config.num_watermark_bits:
        raise ValueError(f"Length of target_watermark_message_str ({len(config.target_watermark_message_str)}) "
                         f"must match num_watermark_bits ({config.num_watermark_bits}).")
    
    # Create target binary message tensor (0s and 1s)
    # This will be of shape [1, num_watermark_bits] for batch_size_generation=1
    # It will be expanded in generate_images_WIP if batch_size_generation > 1
    target_binary_message_list = [int(bit) for bit in config.target_watermark_message_str]
    target_binary_message_tensor = torch.tensor(target_binary_message_list, dtype=torch.float32, device=accelerator.device).unsqueeze(0)
    print(f"Target binary message tensor prepared: shape {target_binary_message_tensor.shape}")

    # Set models to eval mode
    unet.eval()
    sd_vae_decoder.eval()
    text_encoder.eval()
    # teacher_vae_model.eval() # If you were using the old VAE

    class_prompts = generate_class_prompts(config.label_name, config.data_type)
    class_text_embeddings, _ = precompute_text_embeddings(class_prompts, tokenizer, text_encoder, unet, config.batch_size_generation)
    
    os.makedirs(config.save_syn_data_path, exist_ok=True)
    
    syn_image_seed = config.SEED
    num_sets_per_class = config.generate_nums

    for set_idx in tqdm(range(num_sets_per_class), desc="Overall Generation Progress"):
        print(f"Generating set {set_idx + 1}/{num_sets_per_class} for each class.")
        for class_index in range(len(class_prompts)):
            if videoseal_model is None and config.watermark_guidance_weight != 0:
                print("Skipping VideoSeal guidance as model failed to load and weight is non-zero. Set weight to 0 to proceed.")
                continue

            syn_image_seed = generate_images_WIP(accelerator,
                                                class_index,
                                                class_text_embeddings,
                                                uncond_embeddings,
                                                noise_scheduler,
                                                sd_vae_decoder,
                                                unet,
                                                videoseal_model, # Pass the VideoSeal model
                                                target_binary_message_tensor, # Pass the target message
                                                generator,
                                                weight_dtype,
                                                syn_image_seed,
                                                config)
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process: # Typically accelerator.end_training() is for distributed training
        print("Generation complete.")
    # accelerator.end_training() # Usually for training loops with optimizers

if __name__ == "__main__":
    config_run = Config()
    # Example: Ensure VideoSeal checkpoint exists or adjust path
    # if not os.path.exists(config_run.videoseal_ckpt_path) and not os.path.isdir(config_run.videoseal_ckpt_path) :
    #     print(f"Warning: VideoSeal checkpoint {config_run.videoseal_ckpt_path} not found. Update Config.")
    #     # Potentially download or point to the correct one. For now, it will try to load and might fail.
    main(config_run)