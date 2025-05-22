import torch
from torch import nn # Added for VAE class
from SD.models import set_seed
import torchvision
from tqdm import tqdm
import os
import torch.nn.functional as F
from SD.models import setup_accelerator_and_logging, load_models_and_tokenizer \
                        , prepare_uncond_embeddings
from SD.helper import generate_class_prompts, precompute_text_embeddings

# ========= VAE Class Definition (from your training script) ==========
class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_space_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),             # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),            # 8x8 -> 4x4
            nn.ReLU()
        )
        self.hidden_to_miu = nn.Linear(256 * 4 * 4, latent_space_dim)
        self.hidden_to_sigma = nn.Linear(256 * 4 * 4, latent_space_dim) # This sigma is log_variance
        self.latent_to_hidden = nn.Linear(latent_space_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),   # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),    # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1), # 16x16 -> 32x32
            nn.Sigmoid() # Output [0, 1]
        )

    def forward(self, x): # Expects x to be float32 if VAE is float32
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        miu = self.hidden_to_miu(h)
        sigma_logvar = self.hidden_to_sigma(h)
        
        # For DGM's MSE, we use a reconstruction from miu.
        temp_z_for_recon = miu
        h_decode = self.latent_to_hidden(temp_z_for_recon)
        h_decode = h_decode.view(h_decode.size(0), 256, 4, 4)
        recon_x = self.decoder(h_decode)
        return recon_x, miu, sigma_logvar
# =====================================================================


def generate_images_WIP(accelerator, class_index, class_text_embeddings, uncond_embeddings, noise_scheduler, sd_vae_decoder, unet, teacher_vae_model,
                      generator, weight_dtype, syn_image_seed, config):
    image_save_dir_path = os.path.join(config.save_syn_data_path, str(class_index))
    os.makedirs(image_save_dir_path, exist_ok=True)
    text_embeddings = class_text_embeddings[class_index]
    syn_image_seed += 1
    generator.manual_seed(syn_image_seed)
    set_seed(syn_image_seed)

    loss_m_tracker = torch.tensor(0.0, device=accelerator.device)
    loss_kl_tracker = torch.tensor(0.0, device=accelerator.device)
    torch.cuda.empty_cache()
    with accelerator.accumulate(unet): # Not actually training unet, but sets context
        latents_shape = (config.batch_size_generation, unet.config.in_channels, 64, 64) # SD latents are 64x64 for 512x512 images
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
            apply_dgm_guidance = (config.m_dgm_weight != 0 or config.kl_dgm_weight != 0) and \
                                 (timestep_idx >= (config.inference_nums * config.skip_dgm_initial_ratio))
            image_512_pred_x0_dgm = None # To store 512x512 predicted x0 when DGM is active
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
                input_latents_for_sd_vae = (pred_original_sample_float / config.sd_vae_scale_factor).to(sd_vae_decoder.dtype)
                image_from_sd = sd_vae_decoder.decode(input_latents_for_sd_vae).sample 
                image_norm_for_teacher_float = (image_from_sd.float() / 2 + 0.5).clamp(0, 1)
                image_512_pred_x0_dgm = image_norm_for_teacher_float # This is the 512x512 [0,1] version for potential saving
                image_smol_for_teacher_float = F.interpolate(image_norm_for_teacher_float, size=(32, 32), mode='bilinear', align_corners=False)
                if timestep_idx % config.debug_print_freq == 0 :
                    print(f"\n  T{timesteps.item()}: guided_noise_pred_cfg dtype: {guided_noise_pred_cfg.dtype}")
                    print(f"  T{timesteps.item()}: image_from_sd dtype: {image_from_sd.dtype}, min/max: {image_from_sd.min().item():.3f}/{image_from_sd.max().item():.3f}")
                    print(f"  T{timesteps.item()}: image_norm_for_teacher_float (512x512 for save) dtype: {image_norm_for_teacher_float.dtype}, min/max: {image_norm_for_teacher_float.min().item():.3f}/{image_norm_for_teacher_float.max().item():.3f}")
                    print(f"  T{timesteps.item()}: image_smol_for_teacher_float (32x32 for VAE) dtype: {image_smol_for_teacher_float.dtype}, min/max: {image_smol_for_teacher_float.min().item():.3f}/{image_smol_for_teacher_float.max().item():.3f}")
                teacher_recon_image, teacher_miu, teacher_sigma_logvar = teacher_vae_model(image_smol_for_teacher_float)
                if timestep_idx % config.debug_print_freq == 0:
                    print(f"  T{timesteps.item()}: Teacher VAE mu min/max/mean: {teacher_miu.min().item():.3f}/{teacher_miu.max().item():.3f}/{teacher_miu.mean().item():.3f}")
                    print(f"  T{timesteps.item()}: Teacher VAE logvar min/max/mean: {teacher_sigma_logvar.min().item():.3f}/{teacher_sigma_logvar.max().item():.3f}/{teacher_sigma_logvar.mean().item():.3f}")
                loss_m_val = F.mse_loss(teacher_recon_image, image_smol_for_teacher_float, reduction="sum")
                loss_kl_val = 0.5 * torch.sum(torch.exp(teacher_sigma_logvar) + teacher_miu.pow(2) - 1.0 - teacher_sigma_logvar)
                loss_m_tracker = loss_m_val.detach()
                loss_kl_tracker = loss_kl_val.detach()
                total_dgm_loss = loss_m_val * config.m_dgm_weight + loss_kl_val * config.kl_dgm_weight
                if timestep_idx % config.debug_print_freq == 0:
                    print(f"  T{timesteps.item()}: loss_m={loss_m_val.item():.3e}, loss_kl={loss_kl_val.item():.3e}, total_dgm={total_dgm_loss.item():.3e}")
                grad_guided = torch.autograd.grad(total_dgm_loss, current_noise_pred_for_loss_float, allow_unused=True)[0]
                if grad_guided is not None:
                    if timestep_idx % config.debug_print_freq == 0:
                        print(f"  T{timesteps.item()}: grad_guided L2_norm (before clip): {grad_guided.norm().item():.3e}")
                        print(f"  T{timesteps.item()}: grad_guided_mean_abs={grad_guided.abs().mean().item():.3e}")
                    grad_guided = torch.clamp(grad_guided, -config.grad_clip, config.grad_clip)
                    final_noise_pred_for_step = (current_noise_pred_for_loss_float - grad_guided * config.lr_dgm).to(guided_noise_pred_cfg.dtype)
                else:
                    if timestep_idx % config.debug_print_freq == 0: print(f"  T{timesteps.item()}: grad_guided is None!")
                    # final_noise_pred_for_step remains guided_noise_pred_cfg
            torch.cuda.empty_cache()    
            
            # Save intermediate images every 2nd step (timestep_idx 1, 3, 5, ...)
            if (timestep_idx + 1) % 2 == 0 :
                if apply_dgm_guidance and image_512_pred_x0_dgm is not None:
                    # Save 512x512 predicted x0 from DGM path
                    image_name_512_dgm = os.path.join(image_save_dir_path,
                        f"{syn_image_seed}_{class_index}_s{timesteps.item():04d}_dgm_pred_x0_512_m{loss_m_tracker.item():.2e}_kl{loss_kl_tracker.item():.2e}.jpg")
                    torchvision.utils.save_image(image_512_pred_x0_dgm.detach().cpu(), image_name_512_dgm)
                    
                    # Removed saving of 32x32 input to VAE
                    # Removed saving of 32x32 teacher VAE reconstruction

                else: # Standard CFG step or DGM skipped, decode for saving
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
                        image_to_save_intermediate_cfg = (image_cfg_decoded.float() / 2 + 0.5).clamp(0, 1) # 512x512, [0,1]
                        
                        save_tag = "cfg_only"
                        image_name_cfg = os.path.join(image_save_dir_path,
                            f"{syn_image_seed}_{class_index}_s{timesteps.item():04d}_{save_tag}.jpg")
                        torchvision.utils.save_image(image_to_save_intermediate_cfg.detach().cpu(), image_name_cfg)
            
            # --- Scheduler Step ---
            with torch.no_grad():
                scheduler_output_step = noise_scheduler.step(
                    final_noise_pred_for_step.to(unet.dtype), 
                    timesteps.cpu(), # Timestep should be on CPU for some schedulers
                    latents.detach(),
                    generator=generator,
                    return_dict=True 
                )
                latents = scheduler_output_step.prev_sample.to(weight_dtype)
            
            # Clear gradients if any were accumulated (though not strictly necessary if models are in eval mode and no optimizer step)
            if unet.training: unet.zero_grad() # Should be in eval, but as a safeguard
            if sd_vae_decoder.training: sd_vae_decoder.zero_grad()
            if teacher_vae_model.training: teacher_vae_model.zero_grad()
            
            # If current_noise_pred_for_loss_float was created, ensure its grad is cleared
            if 'current_noise_pred_for_loss_float' in locals() and current_noise_pred_for_loss_float.grad is not None:
                current_noise_pred_for_loss_float.grad.zero_() # Or del current_noise_pred_for_loss_float to free memory if not needed
            
            torch.cuda.empty_cache()

        # Save final images
        with torch.no_grad():
            # The last step in DDIM/DDPM typically doesn't denoise. It uses the x0 prediction from the previous step.
            # However, diffusers schedulers often handle this. If timesteps_tensor[:-1] was used,
            # the last denoising step for T_0 might be implicitly handled by how pred_original_sample is used.
            # The original code had a final denoising step. Let's re-evaluate if this is standard.
            # Most pipelines decode latents after the loop.
            # Let's scale latents to image space using sd_vae_decoder.
            # latents = 1 / config.sd_vae_scale_factor * latents # This is if latents are x0 at the end
            # The original code performs one more unet pass and scheduler step for the final timestep.
            # This is somewhat unusual if timesteps_tensor[:-1] means all but the T=0 step.
            # If timesteps_tensor includes T=0 as the last element, then the loop ending at [:-1] is correct,
            # and then a final processing for T=0 is done. Let's assume this is the intent.

            # The original code's final step logic:
            final_loop_timestep = timesteps_tensor[-1].to(accelerator.device)
            final_input_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)
            # Ensure latents are on the correct device and dtype for unet
            final_latent_model_input = torch.cat([latents.to(device=accelerator.device, dtype=unet.dtype)] * 2)

            final_noise_preds_unet = unet(final_latent_model_input, final_loop_timestep, final_input_embeddings).sample
            final_uncond_pred, final_text_pred = final_noise_preds_unet.chunk(2)
            final_model_pred_for_saving = final_uncond_pred + config.guided_scale * (final_text_pred - final_uncond_pred)
            
            # Use scheduler's logic to get pred_original_sample from the final noise prediction
            # Some schedulers might not have noise_pred as direct input for final step if it expects x_t-1
            # The original `scheduler_output_step.pred_original_sample` seems more direct if available from the last step of the loop.
            # However, if the loop is `timesteps_tensor[:-1]`, then `latents` is x_ প্রায়0 (noisy x0) or x_T_final-1
            # The original code uses .pred_original_sample from a *new* scheduler step.
            
            # Get x0 from the final latents (which should be x_T_final where T_final is the last timestep in timesteps_tensor)
            # This step should ideally be x0 if the scheduler has processed down to the equivalent of t=0
            # If `latents` is the output from the last step of the main loop (which processed up to timesteps_tensor[-2]),
            # then we need one more step with timesteps_tensor[-1].

            # The original code uses `ori_latents_final` from a scheduler step with `final_model_pred_for_saving`.
            # This implies `latents` at this point is `x_t` for `final_loop_timestep`.
            scheduler_output_final = noise_scheduler.step(
                final_model_pred_for_saving.to(unet.dtype), # model_output (noise pred)
                final_loop_timestep.cpu(), # timestep
                latents.to(unet.dtype), # sample (current latents x_t)
                generator=generator,
                return_dict=True # ensure return_dict is True if accessing .pred_original_sample
            )
            ori_latents_final = scheduler_output_final.pred_original_sample # This should be x0 estimate
            
            # Decode the predicted original sample (x0)
            input_latents_final_decode = (ori_latents_final / config.sd_vae_scale_factor).to(sd_vae_decoder.dtype)
            image_final_decoded = sd_vae_decoder.decode(input_latents_final_decode).sample
            image_final_save_scaled = (image_final_decoded.float() / 2 + 0.5).clamp(0, 1) # Normalize to [0,1]
            
            for i in range(config.batch_size_generation):
                loss_m_val_fn = loss_m_tracker.item() 
                loss_kl_val_fn = loss_kl_tracker.item()
                image_name = os.path.join(image_save_dir_path,
                    f"{syn_image_seed}_{class_index}_final_m{loss_m_val_fn:.2e}_kl{loss_kl_val_fn:.2e}_{i}.jpg")
                torchvision.utils.save_image(image_final_save_scaled[i].detach().cpu(), image_name)
                print(f"Saved final image: {image_name} (Size: {image_final_save_scaled[i].shape})")

    torch.cuda.empty_cache()
    return syn_image_seed


def load_teacher_vae_model(trained_dgm_path, device):
    model = VAE() 
    model.load_state_dict(torch.load(trained_dgm_path, map_location=device))
    model.eval()
    return model


class Config:
    def __init__(self):
        self.SEED = 5000
        self.name = "synthetic_cifar_final" # Changed name to avoid overwriting previous runs
        self.save_syn_data_path = "./synthetic_cifar_final" # Changed path
        self.checkpoints_dir = "./checkpoints" 
        self.generate_nums = 1000 # Reduced for faster testing
        self.batch_size_generation = 1
        self.STABLE_DIFFUSION = "runwayml/stable-diffusion-v1-5"
        self.SD_REVISION = None 
        self.inference_nums = 10
        self.guided_scale = 5
        self.lr_dgm = 0.5 # CHANGED from 0.4 (and originally 0.7)
        self.m_dgm_weight = 5.0  
        self.kl_dgm_weight = 2 # CHANGED from 2.0 (and originally 20)
        self.skip_dgm_initial_ratio = 0.1 
        self.grad_clip = 1
        self.sd_vae_scale_factor = 0.18215 
        self.trained_dgm_path = "./VAE/vae_model.pth"
        self.label_name = True
        self.data_type = "cifar10"
        self.save_intermediate_every_n_steps = 5
        self.debug_print_freq = 10 # Increased frequency for more debug output

def main(config:Config):
    accelerator = setup_accelerator_and_logging(config.SEED)
    
    tokenizer, text_encoder, noise_scheduler, sd_vae_decoder, unet, _, _, generator, weight_dtype = \
        load_models_and_tokenizer(accelerator, config.STABLE_DIFFUSION, config.SD_REVISION, config.SEED)
    
    print(f"Loaded SD models. UNet dtype: {unet.dtype}, SD VAE decoder dtype: {sd_vae_decoder.dtype}, Text Encoder dtype: {text_encoder.dtype}, weight_dtype for latents: {weight_dtype}")

    uncond_embeddings = prepare_uncond_embeddings(tokenizer, text_encoder, unet, config.batch_size_generation)
    
    if not os.path.exists(config.trained_dgm_path):
        print(f"Error: Teacher VAE model not found at {config.trained_dgm_path}")
        print("Creating a dummy VAE file for placeholder purposes. DGM guidance will not be meaningful.")
        os.makedirs(os.path.dirname(config.trained_dgm_path), exist_ok=True)
        dummy_vae = VAE()
        torch.save(dummy_vae.state_dict(), config.trained_dgm_path)

    teacher_vae_model = load_teacher_vae_model(config.trained_dgm_path, accelerator.device)
    teacher_vae_model = teacher_vae_model.to(accelerator.device) 
    print(f"Teacher VAE loaded. Dtype: {next(teacher_vae_model.parameters()).dtype}")
    
    unet.eval()
    sd_vae_decoder.eval()
    text_encoder.eval()
    teacher_vae_model.eval()

    class_prompts = generate_class_prompts(config.label_name, config.data_type)
    class_text_embeddings, _ = precompute_text_embeddings(class_prompts, tokenizer, text_encoder, unet, config.batch_size_generation)
    
    # config.save_syn_data_path is already set in Config and includes the new "debug" name
    os.makedirs(config.save_syn_data_path, exist_ok=True) 
    
    syn_image_seed = config.SEED
    num_sets_per_class = config.generate_nums 

    for set_idx in tqdm(range(num_sets_per_class), desc="Overall Generation Progress"):
        print(f"Generating set {set_idx + 1}/{num_sets_per_class} for each class.")
        for class_index in range(len(class_prompts)):
            syn_image_seed = generate_images_WIP(accelerator,
                                                class_index,
                                                class_text_embeddings,
                                                uncond_embeddings,
                                                noise_scheduler,
                                                sd_vae_decoder, 
                                                unet,
                                                teacher_vae_model, 
                                                generator,
                                                weight_dtype,
                                                syn_image_seed,
                                                config)
    
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    config_run = Config()
    main(config_run)