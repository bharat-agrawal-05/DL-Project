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

# Helper for bit accuracy
def calculate_bit_accuracy(pred_logits, target_bits):
    preds_binary = (pred_logits > 0).float()
    correct_bits = (preds_binary == target_bits).float()
    accuracy = torch.mean(correct_bits)
    return accuracy.item()

class VAE(nn.Module):
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

def generate_images_WIP(accelerator, class_index, single_class_text_embeddings, uncond_embeddings,
                      noise_scheduler, sd_vae_decoder, unet,
                      videoseal_model_instance, target_binary_message_tensor,
                      generator, weight_dtype, syn_image_seed, config):
    image_save_dir_path = os.path.join(config.save_syn_data_path, str(class_index))
    os.makedirs(image_save_dir_path, exist_ok=True)
    text_embeddings = single_class_text_embeddings
    syn_image_seed += 1
    generator.manual_seed(syn_image_seed)
    set_seed(syn_image_seed)
    
    final_dgm_loss_for_image = torch.tensor(0.0, device=accelerator.device)
    torch.cuda.empty_cache()
    
    latents_shape = (config.batch_size_generation, unet.config.in_channels, 64, 64)
    latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=weight_dtype).to(accelerator.device)
    latents = latents * noise_scheduler.init_noise_sigma
    noise_scheduler.set_timesteps(config.inference_nums)
    timesteps_tensor = noise_scheduler.timesteps.to(accelerator.device)
    print(f"\nStarting generation for class {class_index}, seed {syn_image_seed}.")

    interpolation_params_videoseal = {"mode": "bilinear", "align_corners": False, "antialias": True}

    num_loop_steps = len(timesteps_tensor[:-1]) 
    dgm_start_step_idx = int(num_loop_steps * config.skip_dgm_initial_ratio)
    dgm_end_step_idx = int(num_loop_steps * (1.0 - config.skip_dgm_final_ratio))
    
    dgm_steps_counter = 0 # Counter for DGM application frequency

    for timestep_idx, timesteps in enumerate(tqdm(timesteps_tensor[:-1], desc=f"Class {class_index} Steps")):
        if text_embeddings.shape[0] != uncond_embeddings.shape[0] and text_embeddings.shape[0] ==1 and uncond_embeddings.shape[0] > 1:
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
        
        # Determine if DGM should be generally active in this window
        is_in_dgm_window = (
            timestep_idx >= dgm_start_step_idx and
            timestep_idx < dgm_end_step_idx
        )

        apply_dgm_guidance = False
        if config.watermark_guidance_weight != 0 and is_in_dgm_window:
            if (dgm_steps_counter % config.dgm_apply_every_n_steps) == 0:
                apply_dgm_guidance = True
            dgm_steps_counter += 1 # Increment only if within the general DGM window
        
        loss_watermark_at_step = 0.0 

        if apply_dgm_guidance:
            current_noise_pred_for_dgm_loss = guided_noise_pred_cfg.float().clone().requires_grad_()
            
            alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps].to(latents.device)
            sqrt_alpha_prod_t = alpha_prod_t.sqrt(); sqrt_one_minus_alpha_prod_t = (1 - alpha_prod_t).sqrt()
            if sqrt_alpha_prod_t.ndim == 0:
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
            
            message_logits_from_model = detected_logits_pooled[:, 1:] 
            
            message_logits_for_loss_calc = message_logits_from_model
            if message_logits_from_model.shape[1] != config.num_watermark_bits:
                # This warning and slicing should ideally not be needed if you load a model
                # whose bit capacity matches config.num_watermark_bits.
                # print(f"Warning (DGM Step T={timesteps.item()}): VideoSeal model produced {message_logits_from_model.shape[1]} logit bits, "
                #       f"but target is {config.num_watermark_bits} bits. Slicing model output.")
                message_logits_for_loss_calc = message_logits_from_model[:, :config.num_watermark_bits]

            current_batch_size = message_logits_for_loss_calc.shape[0]
            expanded_target_message = target_binary_message_tensor.expand(current_batch_size, -1)
            
            loss_watermark_val = F.binary_cross_entropy_with_logits(message_logits_for_loss_calc, expanded_target_message.float(), reduction="sum")
            final_dgm_loss_for_image = loss_watermark_val.detach().clone()
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
                
                # More detailed tag if DGM was attempted but grad was None
                if apply_dgm_guidance: # Only if DGM was attempted this step
                    grad_guided_computed_this_iter = 'grad_guided' in locals() and grad_guided is not None
                    if not grad_guided_computed_this_iter:
                        save_tag += "_nograd"

                image_name_intermediate = os.path.join(image_save_dir_path, f"{syn_image_seed}_{class_index}_s{timesteps.item():04d}_{save_tag}_lossW{loss_watermark_at_step:.2e}.jpg")
                torchvision.utils.save_image(image_to_save_intermediate.cpu(), image_name_intermediate)

        with torch.no_grad():
            scheduler_output_step = noise_scheduler.step(final_noise_pred_for_step.to(unet.dtype), timesteps.cpu(), latents.detach(), generator=generator, return_dict=True)
            latents = scheduler_output_step.prev_sample.to(weight_dtype)
        if unet.training: unet.zero_grad(set_to_none=True)
        if sd_vae_decoder.training: sd_vae_decoder.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

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
                # print(f"Warning (Final Eval): VideoSeal model produced {final_message_logits_from_model.shape[1]} logit bits, "
                #       f"but target is {config.num_watermark_bits} bits. Slicing model output for accuracy.")
                final_message_logits_for_acc = final_message_logits_from_model[:, :config.num_watermark_bits]

            current_final_batch_size = final_message_logits_for_acc.shape[0]
            expanded_target_final = target_binary_message_tensor.expand(current_final_batch_size, -1)
            final_bit_accuracy = calculate_bit_accuracy(final_message_logits_for_acc, expanded_target_final)

        for i in range(config.batch_size_generation):
            loss_w_val_fn = final_dgm_loss_for_image.item()
            image_name = os.path.join(image_save_dir_path,
                f"{syn_image_seed}_{class_index}_final_lossW{loss_w_val_fn:.2e}_acc{final_bit_accuracy:.2f}_{i}.jpg")
            torchvision.utils.save_image(image_final_save_scaled[i].cpu(), image_name)
            print(f"Saved final image: {image_name} (Size: {image_final_save_scaled[i].shape}) "
                  f"Last DGM Step Loss: {loss_w_val_fn:.2e}, Final Bit Accuracy: {final_bit_accuracy:.4f}")
            
    torch.cuda.empty_cache()
    return syn_image_seed

class Config:
    def __init__(self):
        self.SEED = 5000
        self.name = "TRYING" 
        self.save_syn_data_path = "./TRYING"
        self.checkpoints_dir = "./checkpoints"
        self.generate_nums = 1 
        self.num_images_to_generate_total = 5 
        self.batch_size_generation = 1
        self.STABLE_DIFFUSION = "runwayml/stable-diffusion-v1-5"; self.SD_REVISION = None
        self.inference_nums = 120 
        self.guided_scale = 7.5 
        self.lr_dgm = 3
        self.watermark_guidance_weight = 1.0 
        self.skip_dgm_initial_ratio = 0
        self.skip_dgm_final_ratio = 0   
        self.dgm_apply_every_n_steps = 2 # ADDED: Apply DGM every X steps within the active window (1 means every step)
        self.grad_clip = 0.5 
        self.sd_vae_scale_factor = 0.18215
        
        self.videoseal_ckpt_path = 'videoseal' 
        
        self.target_watermark_message_str = "10" * 64; self.num_watermark_bits = 128 
        
        self.label_name = True; self.data_type = "cifar10"
        self.save_intermediate_images = True 
        self.save_intermediate_every_n_steps = 10
        
def main(config:Config):
    accelerator = setup_accelerator_and_logging(config.SEED)
    tokenizer, text_encoder, noise_scheduler, sd_vae_decoder, unet, _, _, generator, weight_dtype = \
        load_models_and_tokenizer(accelerator, config.STABLE_DIFFUSION, config.SD_REVISION, config.SEED)
    print(f"Loaded SD models. UNet: {unet.dtype}, SD VAE: {sd_vae_decoder.dtype}, TextEnc: {text_encoder.dtype}, Latents: {weight_dtype}")
    unet.to(accelerator.device).eval(); text_encoder.to(accelerator.device).eval(); sd_vae_decoder.to(accelerator.device).eval()
    uncond_embeddings = prepare_uncond_embeddings(tokenizer, text_encoder, unet, config.batch_size_generation)
    uncond_embeddings = uncond_embeddings.to(accelerator.device, dtype=text_encoder.dtype)
    
    print(f"Loading VideoSeal model from checkpoint: {config.videoseal_ckpt_path}...")
    videoseal_model_instance = None
    try:
        videoseal_model_instance = setup_model_from_checkpoint(config.videoseal_ckpt_path)
        videoseal_model_instance.to(accelerator.device).eval()
        print(f"VideoSeal model loaded and moved to {accelerator.device}.")
        for param in videoseal_model_instance.parameters(): 
            param.requires_grad = False
    except Exception as e:
        print(f"Error loading VideoSeal model ('{config.videoseal_ckpt_path}'): {e}")
        config.watermark_guidance_weight = 0

    if videoseal_model_instance and len(config.target_watermark_message_str) != config.num_watermark_bits:
        raise ValueError(f"Config message length mismatch.")
    
    target_binary_message_list = [int(bit) for bit in config.target_watermark_message_str]
    target_binary_message_tensor = torch.tensor(target_binary_message_list, dtype=torch.float32, device=accelerator.device).unsqueeze(0)
    print(f"Target message: {config.num_watermark_bits} bits, shape {target_binary_message_tensor.shape}")
    
    class_prompts = generate_class_prompts(config.label_name, config.data_type)
    list_of_class_text_embeddings, _ = precompute_text_embeddings(class_prompts, tokenizer, text_encoder, unet, config.batch_size_generation)
    class_text_embeddings = [emb.to(accelerator.device, dtype=text_encoder.dtype) for emb in list_of_class_text_embeddings]
    
    os.makedirs(config.save_syn_data_path, exist_ok=True)
    syn_image_seed = config.SEED
    
    images_generated_count = 0
    for set_idx in tqdm(range(config.generate_nums), desc="Overall Generation Sets"):
        if images_generated_count >= config.num_images_to_generate_total: break
        print(f"\nGenerating set {set_idx + 1}/{config.generate_nums}.")
        num_classes_this_set = min(len(class_prompts), config.num_images_to_generate_total - images_generated_count)
        for class_idx_loop in range(num_classes_this_set):
            if images_generated_count >= config.num_images_to_generate_total: break
            if videoseal_model_instance is None and config.watermark_guidance_weight != 0:
                print("Skipping DGM: VideoSeal model not loaded and weight > 0."); continue
            current_class_text_embeddings = class_text_embeddings[class_idx_loop]
            print(f"--- Generating image {images_generated_count + 1}/{config.num_images_to_generate_total} (Class Index: {class_idx_loop}, Prompt: '{class_prompts[class_idx_loop]}') ---")
            syn_image_seed = generate_images_WIP(accelerator, class_idx_loop, current_class_text_embeddings,
                                                uncond_embeddings, noise_scheduler, sd_vae_decoder, unet,
                                                videoseal_model_instance,
                                                target_binary_message_tensor, generator,
                                                weight_dtype, syn_image_seed, config)
            images_generated_count += config.batch_size_generation
            
    accelerator.wait_for_everyone()
    if accelerator.is_main_process: print(f"Generation complete. Total images generated: {images_generated_count}")

if __name__ == "__main__":
    config_run = Config()
    main(config_run)