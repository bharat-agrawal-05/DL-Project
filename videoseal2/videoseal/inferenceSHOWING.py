import torch
from torch import nn
from SD.models import set_seed # Assuming this exists in your SD.models
import torchvision
from tqdm import tqdm
import os
import numpy as np
import torch.nn.functional as F
from SD.models import setup_accelerator_and_logging, load_models_and_tokenizer \
                        , prepare_uncond_embeddings # Assuming these exist in your SD.models
from SD.helper import precompute_text_embeddings # generate_class_prompts removed

# Import for VideoSeal
from videoseal.evals.full import setup_model_from_checkpoint # Assuming this exists

# Helper for bit accuracy
def calculate_bit_accuracy(pred_logits, target_bits):
    preds_binary = (pred_logits > 0).float()
    correct_bits = (preds_binary == target_bits).float()
    accuracy = torch.mean(correct_bits)
    return accuracy.item()

class VAE(nn.Module): # This VAE class is part of the original code, kept as is.
    def __init__(self, input_channels=3, latent_space_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.ReLU()
        )
        self.hidden_to_miu = nn.Linear(256 * 4 * 4, latent_space_dim)
        self.hidden_to_sigma = nn.Linear(256 * 4 * 4, latent_space_dim)
        self.latent_to_hidden = nn.Linear(latent_space_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        h = self.encoder(x); h = h.view(h.size(0), -1)
        miu = self.hidden_to_miu(h); sigma_logvar = self.hidden_to_sigma(h)
        h_decode = self.latent_to_hidden(miu); h_decode = h_decode.view(h_decode.size(0), 256, 4, 4)
        return self.decoder(h_decode), miu, sigma_logvar

def generate_images_WIP(accelerator, class_index, single_prompt_text_embeddings, uncond_embeddings,
                      noise_scheduler, sd_vae_decoder, unet,
                      videoseal_model_instance, target_binary_message_tensor,
                      generator, weight_dtype, syn_image_seed, config):
    # class_index will be a fixed value (e.g., 0) passed from main, used for subfolder naming.
    image_save_dir_path = os.path.join(config.save_syn_data_path, str(class_index))
    os.makedirs(image_save_dir_path, exist_ok=True)
    
    text_embeddings = single_prompt_text_embeddings # Embeddings for the user-provided prompt
    
    syn_image_seed += 1
    generator.manual_seed(syn_image_seed)
    set_seed(syn_image_seed) # Sets seed for torch, numpy, etc.
    
    final_dgm_loss_for_image = torch.tensor(0.0, device=accelerator.device)
    torch.cuda.empty_cache()
    
    latents_shape = (config.batch_size_generation, unet.config.in_channels, 64, 64)
    latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=weight_dtype).to(accelerator.device)
    latents = latents * noise_scheduler.init_noise_sigma
    
    noise_scheduler.set_timesteps(config.inference_nums)
    timesteps_tensor = noise_scheduler.timesteps.to(accelerator.device)
    
    # Log message updated to reflect the nature of class_index
    print(f"\nStarting generation using index {class_index} for file organization (seed {syn_image_seed}).")

    interpolation_params_videoseal = {"mode": "bilinear", "align_corners": False, "antialias": True}

    num_loop_steps = len(timesteps_tensor[:-1]) 
    dgm_start_step_idx = int(num_loop_steps * config.skip_dgm_initial_ratio)
    dgm_end_step_idx = int(num_loop_steps * (1.0 - config.skip_dgm_final_ratio))
    
    dgm_steps_counter = 0

    for timestep_idx, timesteps in enumerate(tqdm(timesteps_tensor[:-1], desc=f"Index {class_index} Denoising Steps")):
        if text_embeddings.shape[0] != uncond_embeddings.shape[0] and text_embeddings.shape[0] == 1 and uncond_embeddings.shape[0] > 1:
            text_embeddings_batch = text_embeddings.repeat(uncond_embeddings.shape[0], 1, 1)
        else:
            text_embeddings_batch = text_embeddings
        
        input_embeddings = torch.cat([uncond_embeddings, text_embeddings_batch], dim=0)
        latent_model_input = torch.cat([latents] * 2)
        
        with torch.no_grad():
            noise_pred_unet_output = unet(latent_model_input, timesteps, input_embeddings).sample
        
        uncond_noise_pred_unet, text_noise_pred_unet = noise_pred_unet_output.chunk(2)
        guided_noise_pred_cfg = (uncond_noise_pred_unet + config.guided_scale * (text_noise_pred_unet - uncond_noise_pred_unet))
        final_noise_pred_for_step = guided_noise_pred_cfg.clone()
        
        is_in_dgm_window = (
            timestep_idx >= dgm_start_step_idx and
            timestep_idx < dgm_end_step_idx
        )

        apply_dgm_guidance = False
        if config.watermark_guidance_weight != 0 and is_in_dgm_window and videoseal_model_instance is not None:
            if (dgm_steps_counter % config.dgm_apply_every_n_steps) == 0:
                apply_dgm_guidance = True
            dgm_steps_counter += 1 
        
        loss_watermark_at_step = 0.0 

        if apply_dgm_guidance:
            current_noise_pred_for_dgm_loss = guided_noise_pred_cfg.float().clone().requires_grad_()
            
            alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps].to(latents.device)
            sqrt_alpha_prod_t = alpha_prod_t.sqrt(); sqrt_one_minus_alpha_prod_t = (1 - alpha_prod_t).sqrt()
            if sqrt_alpha_prod_t.ndim == 0: # Ensure dimensions for broadcasting
                sqrt_alpha_prod_t = sqrt_alpha_prod_t.view(1,1,1,1); sqrt_one_minus_alpha_prod_t = sqrt_one_minus_alpha_prod_t.view(1,1,1,1)
            
            with torch.enable_grad():
                pred_original_sample_float = (latents.detach().float() - sqrt_one_minus_alpha_prod_t.float() * current_noise_pred_for_dgm_loss) / sqrt_alpha_prod_t.float()
                input_latents_for_sd_vae = (pred_original_sample_float / config.sd_vae_scale_factor).to(sd_vae_decoder.dtype)
                image_from_sd = sd_vae_decoder.decode(input_latents_for_sd_vae).sample
                image_for_videoseal_input = (image_from_sd.float() / 2 + 0.5).clamp(0, 1)

                imgs_res_for_detector = image_for_videoseal_input.clone()
                if image_for_videoseal_input.shape[-2:] != (videoseal_model_instance.img_size, videoseal_model_instance.img_size):
                    imgs_res_for_detector = F.interpolate(
                        image_for_videoseal_input,
                        size=(videoseal_model_instance.img_size, videoseal_model_instance.img_size),
                        **interpolation_params_videoseal 
                    )
                raw_detector_output = videoseal_model_instance.detector(imgs_res_for_detector)
                if raw_detector_output.ndim == 4:
                    detected_logits_pooled = torch.mean(raw_detector_output, dim=[2,3])
                elif raw_detector_output.ndim == 2:
                    detected_logits_pooled = raw_detector_output
                else:
                    raise ValueError(f"Unexpected detector output shape: {raw_detector_output.shape}")
            
            message_logits_from_model = detected_logits_pooled[:, 1:] # Assuming first logit is 'no_message_logit'
            
            message_logits_for_loss_calc = message_logits_from_model
            if message_logits_from_model.shape[1] != config.num_watermark_bits:
                message_logits_for_loss_calc = message_logits_from_model[:, :config.num_watermark_bits]

            current_batch_size = message_logits_for_loss_calc.shape[0]
            # target_binary_message_tensor is already on accelerator.device from main()
            expanded_target_message = target_binary_message_tensor.expand(current_batch_size, -1) 
            
            loss_watermark_val = F.binary_cross_entropy_with_logits(message_logits_for_loss_calc, expanded_target_message.float(), reduction="sum")
            final_dgm_loss_for_image = loss_watermark_val.detach().clone() # Store loss from this DGM step
            loss_watermark_at_step = final_dgm_loss_for_image.item()
            total_dgm_loss = loss_watermark_val * config.watermark_guidance_weight

            current_step_bit_acc = calculate_bit_accuracy(message_logits_for_loss_calc.detach(), expanded_target_message)
            
            grad_guided = torch.autograd.grad(total_dgm_loss, current_noise_pred_for_dgm_loss, allow_unused=False)[0]
            grad_norm = grad_guided.norm().item() if grad_guided is not None else 0.0
            
            print(f"  DGM Step T={timesteps.item()}: Loss={loss_watermark_val.item():.3e}, BitAcc={current_step_bit_acc:.4f}, GradNorm={grad_norm:.3e}")

            if grad_guided is not None:
                grad_guided = torch.clamp(grad_guided, -config.grad_clip, config.grad_clip)
                final_noise_pred_for_step = (current_noise_pred_for_dgm_loss.detach() - grad_guided * config.lr_dgm).to(guided_noise_pred_cfg.dtype)
            
            if hasattr(current_noise_pred_for_dgm_loss, 'grad') and current_noise_pred_for_dgm_loss.grad is not None:
                current_noise_pred_for_dgm_loss.grad.zero_()
            del current_noise_pred_for_dgm_loss
        
        if config.save_intermediate_images and (timestep_idx + 1) % config.save_intermediate_every_n_steps == 0 :
            with torch.no_grad():
                alpha_prod_t_save = noise_scheduler.alphas_cumprod[timesteps].to(latents.device)
                sqrt_alpha_prod_t_save = alpha_prod_t_save.sqrt(); sqrt_one_minus_alpha_prod_t_save = (1 - alpha_prod_t_save).sqrt()
                if sqrt_alpha_prod_t_save.ndim == 0:
                    sqrt_alpha_prod_t_save = sqrt_alpha_prod_t_save.view(1,1,1,1); sqrt_one_minus_alpha_prod_t_save = sqrt_one_minus_alpha_prod_t_save.view(1,1,1,1)
                
                pred_original_sample_to_save = (latents.float() - sqrt_one_minus_alpha_prod_t_save.float() * final_noise_pred_for_step.float()) / sqrt_alpha_prod_t_save.float()
                input_latents_for_sd_vae_save = (pred_original_sample_to_save / config.sd_vae_scale_factor).to(sd_vae_decoder.dtype)
                image_decoded_save = sd_vae_decoder.decode(input_latents_for_sd_vae_save).sample
                image_to_save_intermediate = (image_decoded_save.float() / 2 + 0.5).clamp(0, 1)
                save_tag = "dgm_active_step" if apply_dgm_guidance else ("dgm_window_skipped" if is_in_dgm_window else "cfg_only")
                
                if apply_dgm_guidance: 
                    grad_guided_computed_this_iter = 'grad_guided' in locals() and grad_guided is not None
                    if not grad_guided_computed_this_iter: # Check if grad_guided was computed in this iteration
                        save_tag += "_nograd"

                image_name_intermediate = os.path.join(image_save_dir_path, f"{syn_image_seed}_{class_index}_s{timesteps.item():04d}_{save_tag}_lossW{loss_watermark_at_step:.2e}.jpg")
                torchvision.utils.save_image(image_to_save_intermediate.cpu(), image_name_intermediate)

        with torch.no_grad():
            scheduler_output_step = noise_scheduler.step(final_noise_pred_for_step.to(unet.dtype), timesteps.cpu(), latents.detach(), generator=generator, return_dict=True)
            latents = scheduler_output_step.prev_sample.to(weight_dtype)
        
        if hasattr(unet, 'training') and unet.training: unet.zero_grad(set_to_none=True)
        if hasattr(sd_vae_decoder, 'training') and sd_vae_decoder.training: sd_vae_decoder.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    # Final step processing and image saving
    with torch.no_grad():
        final_loop_timestep = timesteps_tensor[-1].to(accelerator.device)
        if text_embeddings.shape[0] != uncond_embeddings.shape[0] and text_embeddings.shape[0] == 1 and uncond_embeddings.shape[0] > 1:
            text_embeddings_batch_final = text_embeddings.repeat(uncond_embeddings.shape[0], 1, 1)
        else:
            text_embeddings_batch_final = text_embeddings
        
        final_input_embeddings = torch.cat([uncond_embeddings, text_embeddings_batch_final], dim=0)
        final_latent_model_input = torch.cat([latents.to(device=accelerator.device, dtype=unet.dtype)] * 2)
        final_noise_preds_unet_output = unet(final_latent_model_input, final_loop_timestep, final_input_embeddings).sample
        final_uncond_pred, final_text_pred = final_noise_preds_unet_output.chunk(2)
        final_model_pred_for_saving = final_uncond_pred + config.guided_scale * (final_text_pred - final_uncond_pred)
        
        scheduler_output_final = noise_scheduler.step(final_model_pred_for_saving.to(unet.dtype), final_loop_timestep.cpu(), latents.to(unet.dtype), generator=generator, return_dict=True)
        ori_latents_final = scheduler_output_final.pred_original_sample
        input_latents_final_decode = (ori_latents_final / config.sd_vae_scale_factor).to(sd_vae_decoder.dtype)
        image_final_decoded = sd_vae_decoder.decode(input_latents_final_decode).sample
        image_final_save_scaled = (image_final_decoded.float() / 2 + 0.5).clamp(0, 1)
        
        final_bit_accuracy = 0.0
        if videoseal_model_instance is not None and config.watermark_guidance_weight > 0:
            imgs_res_for_final_detect = image_final_save_scaled.clone()
            if image_final_save_scaled.shape[-2:] != (videoseal_model_instance.img_size, videoseal_model_instance.img_size):
                imgs_res_for_final_detect = F.interpolate(
                    image_final_save_scaled,
                    size=(videoseal_model_instance.img_size, videoseal_model_instance.img_size),
                    **interpolation_params_videoseal
                )
            final_raw_detector_output = videoseal_model_instance.detector(imgs_res_for_final_detect)
            if final_raw_detector_output.ndim == 4:
                final_detected_logits_pooled = torch.mean(final_raw_detector_output, dim=[2,3])
            elif final_raw_detector_output.ndim == 2:
                final_detected_logits_pooled = final_raw_detector_output
            else:
                raise ValueError("Unexpected final detector output shape.")
            
            final_message_logits_from_model = final_detected_logits_pooled[:, 1:]
            
            final_message_logits_for_acc = final_message_logits_from_model
            if final_message_logits_from_model.shape[1] != config.num_watermark_bits:
                final_message_logits_for_acc = final_message_logits_from_model[:, :config.num_watermark_bits]

            current_final_batch_size = final_message_logits_for_acc.shape[0]
            expanded_target_final = target_binary_message_tensor.expand(current_final_batch_size, -1)
            final_bit_accuracy = calculate_bit_accuracy(final_message_logits_for_acc, expanded_target_final)

        for i in range(config.batch_size_generation): # Assumes batch_size_generation matches image_final_save_scaled.shape[0]
            loss_w_val_fn = final_dgm_loss_for_image.item() # This is the DGM loss from the last DGM step encountered
            image_name = os.path.join(image_save_dir_path,
                f"{syn_image_seed}_{class_index}_final_lossW{loss_w_val_fn:.2e}_acc{final_bit_accuracy:.2f}_{i}.jpg")
            torchvision.utils.save_image(image_final_save_scaled[i].cpu(), image_name)
            print(f"Saved final image: {image_name} (Size: {image_final_save_scaled[i].shape}) "
                  f"Last DGM Step Loss: {loss_w_val_fn:.2e}, Final Bit Accuracy: {final_bit_accuracy:.4f}")
            
    torch.cuda.empty_cache()
    return syn_image_seed

class Config: # Configuration class remains unchanged as per request
    def __init__(self):
        self.SEED = 5000
        self.name = "TRYING" 
        self.save_syn_data_path = "./TRYING_output" # Changed default to avoid overwriting "TRYING" if it's a script/module
        self.checkpoints_dir = "./checkpoints"
        self.generate_nums = 1 # This field is no longer directly used by the main generation loop logic
        self.num_images_to_generate_total = 1 # Default to 1 image for single prompt
        self.batch_size_generation = 1
        self.STABLE_DIFFUSION = "runwayml/stable-diffusion-v1-5"; self.SD_REVISION = None
        self.inference_nums = 120 
        self.guided_scale = 7.5 
        self.lr_dgm = 3
        self.watermark_guidance_weight = 1.0 
        self.skip_dgm_initial_ratio = 0.0 # Ensure float
        self.skip_dgm_final_ratio = 0.0   # Ensure float
        self.dgm_apply_every_n_steps = 2
        self.grad_clip = 0.5 
        self.sd_vae_scale_factor = 0.18215
        
        self.videoseal_ckpt_path = 'videoseal' # Path to VideoSeal checkpoint
        
        self.target_watermark_message_str = "10" * 64; self.num_watermark_bits = 128 
        
        # These fields are no longer used for prompt generation but kept in Config
        self.label_name = True; self.data_type = "cifar10"
        self.save_intermediate_images = True 
        self.save_intermediate_every_n_steps = 10
        
def main(config:Config):
    # Ask the user for the prompt
    user_prompt = input("Please enter the prompt for the image you want to generate: ")
    if not user_prompt.strip():
        print("No prompt provided. Exiting.")
        return

    accelerator = setup_accelerator_and_logging(config.SEED)
    tokenizer, text_encoder, noise_scheduler, sd_vae_decoder, unet, _, _, generator, weight_dtype = \
        load_models_and_tokenizer(accelerator, config.STABLE_DIFFUSION, config.SD_REVISION, config.SEED)
    
    print(f"Loaded SD models. UNet dtype: {unet.dtype}, SD VAE dtype: {sd_vae_decoder.dtype}, TextEncoder dtype: {text_encoder.dtype}, Latent dtype: {weight_dtype}")
    # Ensure models are on the correct device and in evaluation mode
    unet.to(accelerator.device).eval(); text_encoder.to(accelerator.device).eval(); sd_vae_decoder.to(accelerator.device).eval()
    
    uncond_embeddings = prepare_uncond_embeddings(tokenizer, text_encoder, unet, config.batch_size_generation)
    uncond_embeddings = uncond_embeddings.to(accelerator.device, dtype=text_encoder.dtype)
    
    print(f"Loading VideoSeal model from checkpoint directory/path: {config.videoseal_ckpt_path}...")
    videoseal_model_instance = None
    if config.watermark_guidance_weight > 0: # Only attempt to load if DGM is active
        try:
            videoseal_model_instance = setup_model_from_checkpoint(config.videoseal_ckpt_path)
            videoseal_model_instance.to(accelerator.device).eval()
            print(f"VideoSeal model loaded and moved to {accelerator.device}.")
            for param in videoseal_model_instance.parameters(): 
                param.requires_grad = False # Freeze VideoSeal model
        except Exception as e:
            print(f"Error loading VideoSeal model ('{config.videoseal_ckpt_path}'): {e}")
            print("Proceeding without DGM (watermark_guidance_weight will be effectively 0).")
            # config.watermark_guidance_weight = 0 # No, let generate_images_WIP handle videoseal_model_instance being None
            videoseal_model_instance = None # Ensure it's None so DGM is skipped
    else:
        print("Watermark guidance weight is 0, VideoSeal model will not be loaded or used.")

    # Validate message length if VideoSeal model is loaded and DGM is active
    if videoseal_model_instance is not None and config.watermark_guidance_weight > 0:
        if len(config.target_watermark_message_str) != config.num_watermark_bits:
            raise ValueError(f"Config 'target_watermark_message_str' length ({len(config.target_watermark_message_str)}) "
                             f"does not match 'num_watermark_bits' ({config.num_watermark_bits}).")
    
    target_binary_message_list = [int(bit) for bit in config.target_watermark_message_str]
    target_binary_message_tensor = torch.tensor(target_binary_message_list, dtype=torch.float32, device=accelerator.device).unsqueeze(0)
    print(f"Target watermark message: {config.num_watermark_bits} bits, tensor shape on device: {target_binary_message_tensor.shape}")
    
    # Process the single user prompt
    print(f"Processing user prompt: '{user_prompt}'")
    # precompute_text_embeddings expects a list of prompts.
    # The second argument to precompute_text_embeddings is batch_size for tokenizing, typically 1 for a single prompt.
    list_of_single_prompt_embeddings, _ = precompute_text_embeddings(
        [user_prompt], tokenizer, text_encoder, unet, 1 
    )
    single_prompt_text_embeddings = list_of_single_prompt_embeddings[0].to(accelerator.device, dtype=text_encoder.dtype)
    
    # Ensure text_embeddings are [1, seq_len, hidden_size] for CFG logic in generate_images_WIP
    if single_prompt_text_embeddings.shape[0] != 1:
        print(f"Warning: Text embedding initial shape is {single_prompt_text_embeddings.shape}. Reshaping/slicing to ensure [1, seq_len, dim].")
        single_prompt_text_embeddings = single_prompt_text_embeddings[0:1] # Take the first one if batched

    os.makedirs(config.save_syn_data_path, exist_ok=True)
    syn_image_seed = config.SEED # Initial seed for the overall generation sequence
    
    images_generated_count = 0
    # Calculate how many times to call generate_images_WIP
    num_generation_iterations = (config.num_images_to_generate_total + config.batch_size_generation - 1) // config.batch_size_generation

    # Use a fixed index for directory structure, as we're handling one prompt type per run.
    # This was previously class_idx_loop.
    fixed_folder_index = 0 
    
    print(f"\nGenerating {config.num_images_to_generate_total} image(s) in {num_generation_iterations} batch(es) for the prompt: '{user_prompt}'.")

    for iter_idx in tqdm(range(num_generation_iterations), desc="Generating Image Batches"):
        if images_generated_count >= config.num_images_to_generate_total:
            break # Stop if desired number of images already met/exceeded
        
        current_batch_info = (f"Batch {iter_idx + 1}/{num_generation_iterations} "
                              f"(Image(s) {images_generated_count + 1} to "
                              f"{min(images_generated_count + config.batch_size_generation, config.num_images_to_generate_total)})")
        print(f"--- {current_batch_info} ---")
        
        syn_image_seed = generate_images_WIP(
            accelerator,
            fixed_folder_index, # Used for subfolder naming under save_syn_data_path
            single_prompt_text_embeddings,
            uncond_embeddings,
            noise_scheduler,
            sd_vae_decoder,
            unet,
            videoseal_model_instance, # Pass the (potentially None) VideoSeal model
            target_binary_message_tensor,
            generator,
            weight_dtype,
            syn_image_seed, # Current seed, will be incremented by generate_images_WIP
            config
        )
        images_generated_count += config.batch_size_generation 
            
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Report actual images generated, respecting num_images_to_generate_total
        actual_images_generated = min(images_generated_count, config.num_images_to_generate_total)
        print(f"Generation complete. Total images generated for the prompt: {actual_images_generated}")
        print(f"Images saved in: {os.path.join(config.save_syn_data_path, str(fixed_folder_index))}")


if __name__ == "__main__":
    config_run = Config()
    main(config_run)