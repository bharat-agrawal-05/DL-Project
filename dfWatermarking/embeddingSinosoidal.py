import torch
import torchvision
from tqdm import tqdm
import os
import torch.nn.functional as F
from pathlib import Path
import re
import numpy as np # For message embedding
import math # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< IMPORT MATH MODULE

# Diffusers and Transformers imports
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils import check_min_version
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel

# Accelerate imports
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration

check_min_version("0.10.0.dev0") # diffusers check
logger = get_logger(__name__)

# ========= Config Class Definition ==========
class Config:
    def __init__(self, 
                 prompt: str = "A serene landscape with a hidden message",
                 output_dir: str = "./generated_images_sd_embed",
                 num_images_per_prompt: int = 1,
                 binary_message: str = "10101010101010101010101010101010", # 32-bit default
                 inference_steps: int = 100, 
                 embedding_step: int = 80, 
                 save_intermediate_every_n_steps: int = 20 ,
                 embedding_pattern_type: str = 'sinusoidal' # 'sinusoidal', 'gradient', or 'noise'
                ):
        self.SEED = 42
        self.prompt = prompt
        self.output_dir = output_dir
        self.num_images_per_prompt = num_images_per_prompt

        # Stable Diffusion parameters
        self.STABLE_DIFFUSION_MODEL_NAME = "runwayml/stable-diffusion-v1-5"
        self.SD_REVISION = None
        self.sd_vae_scale_factor = 0.18215
        self.image_size = 512

        # Generation parameters
        self.batch_size_generation = 1
        self.inference_steps = inference_steps    
        self.guidance_scale = 7.5

        # Intermediate saving
        self.save_intermediate_every_n_steps = save_intermediate_every_n_steps 

        # Technical
        self.mixed_precision = "fp16"
        self.gradient_accumulation_steps = 1

        # Message Embedding Parameters 
        self.binary_message = binary_message
        self.embedding_step = embedding_step 
        self.embedding_strength = 0.02 # Adjusted for global pattern methods
        self.embedding_pattern_type = 'sinusoidal' # 'sinusoidal', 'gradient', or 'noise'

        # Parameters for the original patch-based method (if you want to switch back)
        self.embedding_patch_size = 4 


        if len(self.binary_message) != 32:
            raise ValueError("Binary message must be 32 bits long.")


# ========= Helper Functions ==========
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str, HUGGINGFACE_HUB_TOKEN:str=None):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
        token=HUGGINGFACE_HUB_TOKEN
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        return CLIPTextModel
    else:
        raise ValueError(f"Unsupported model class: {model_class}")

def load_models_and_tokenizer(accelerator: Accelerator, model_name: str, revision: str, seed: int, HUGGINGFACE_HUB_TOKEN:str=None):
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    logger.info(f"Using weight_dtype: {weight_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, subfolder="tokenizer", revision=revision, use_fast=False, token=HUGGINGFACE_HUB_TOKEN
    )
    text_encoder_cls = import_model_class_from_model_name_or_path(model_name, revision, HUGGINGFACE_HUB_TOKEN=HUGGINGFACE_HUB_TOKEN)
    text_encoder = text_encoder_cls.from_pretrained(
        model_name, subfolder="text_encoder", revision=revision, torch_dtype=weight_dtype, token=HUGGINGFACE_HUB_TOKEN
    )
    sd_vae = AutoencoderKL.from_pretrained(
        model_name, subfolder="vae", revision=revision, torch_dtype=weight_dtype, token=HUGGINGFACE_HUB_TOKEN
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_name, subfolder="unet", revision=revision, torch_dtype=weight_dtype, token=HUGGINGFACE_HUB_TOKEN
    )
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler", token=HUGGINGFACE_HUB_TOKEN)

    sd_vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    try:
        import xformers
        pt_version_major_minor = ".".join(torch.__version__.split(".")[:2])
        if pt_version_major_minor >= "2.0":
             unet.enable_xformers_memory_efficient_attention()
             logger.info("Using xformers for UNet.")
        else:
            logger.info("PyTorch version might not be fully compatible with xformers build. Skipping xformers enabling.")
    except ImportError:
        logger.info("xformers not available. UNet will use default attention.")
    except Exception as e:
        logger.warning(f"Could not enable xformers memory efficient attention: {e}. UNet will use default attention.")

    unet, text_encoder, sd_vae, noise_scheduler = accelerator.prepare(
        unet, text_encoder, sd_vae, noise_scheduler
    )
    
    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(seed)
    return tokenizer, text_encoder, noise_scheduler, sd_vae, unet, generator, weight_dtype

def setup_accelerator_and_logging(seed: int, gradient_accumulation_steps: int, mixed_precision: str, output_dir: str):
    logging_dir = Path(output_dir, "logs")
    # For Accelerate versions where ProjectConfiguration might cause issues if not fully set up
    # or if a default project is not configured, direct project_dir might be more robust.
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        project_dir=logging_dir # Using project_dir directly
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_main_process:
        os.makedirs(logging_dir, exist_ok=True) # Ensure logging_dir exists
        try:
            # If project_dir is used, init_trackers might not need a project_name argument,
            # or it will use the name of the directory.
            accelerator.init_trackers(project_name="stable_diffusion_generation")
        except ImportError:
            logger.warning("TensorBoard (or other configured tracker) not found. Logging to trackers will be disabled.")
        except Exception as e: 
            logger.warning(f"Could not initialize trackers: {e}")

    set_seed(seed)
    return accelerator

def _encode_prompt(prompt_batch, tokenizer, text_encoder, device, num_images_per_prompt, do_classifier_free_guidance):
    text_inputs = tokenizer(
        prompt_batch, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device) 
    
    attention_mask = None
    if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(device) 

    prompt_embeds = text_encoder(text_input_ids, attention_mask=attention_mask)[0] 
    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

    if do_classifier_free_guidance:
        uncond_tokens = [""] * len(prompt_batch) 
        max_length = text_input_ids.shape[-1]
        uncond_input = tokenizer(
            uncond_tokens, padding="max_length", max_length=max_length,
            truncation=True, return_tensors="pt",
        )
        uncond_input_ids = uncond_input.input_ids.to(device) 
        uncond_attention_mask = None
        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            uncond_attention_mask = uncond_input.attention_mask.to(device) 
        
        negative_prompt_embeds = text_encoder(uncond_input_ids, attention_mask=uncond_attention_mask)[0] 
        negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0) 
    
    return prompt_embeds

# ========= Message Embedding Function (Global Patterns) ==========
def embed_message_global_pattern(image_tensor: torch.Tensor, binary_message: str,
                                 strength: float = 0.02, pattern_type: str = 'sinusoidal'):
    """
    Embeds a binary message by applying global patterns to the entire image.
    """
    if image_tensor.dim() != 4:
        raise ValueError("Image tensor must be 4D (B, C, H, W)")
    
    modified_image_batch = image_tensor.clone()
    # Operate on the first image in the batch
    img_to_modify = modified_image_batch[0] # Shape (C, H, W)
    device = img_to_modify.device
    
    msg_len = len(binary_message)
    if msg_len == 0: return image_tensor # No message to embed

    h, w = img_to_modify.shape[1], img_to_modify.shape[2] # Get H, W from the single image
    
    # Create coordinate grids once, ensure they are on the correct device
    y_coords, x_coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    y_coords = y_coords.float() / (h -1 if h > 1 else 1) # Normalize to [0,1], prevent div by zero
    x_coords = x_coords.float() / (w -1 if w > 1 else 1) # Normalize to [0,1]
    
    bits = torch.tensor([int(bit) for bit in binary_message], device=device)
    
    cumulative_pattern = torch.zeros_like(img_to_modify[0]) # Accumulate patterns on a (H,W) tensor

    for bit_idx, bit_val in enumerate(bits):
        angle = (bit_idx / msg_len) * math.pi 
        freq_x = 2 + (bit_idx % 5)  
        freq_y = 2 + ((bit_idx // 5) % 5)
        
        current_bit_pattern = torch.zeros_like(x_coords) # (H,W)

        if pattern_type == 'sinusoidal':
            if bit_val == 1:
                current_bit_pattern = torch.cos(2 * math.pi * freq_x * x_coords + angle) * torch.cos(2 * math.pi * freq_y * y_coords)
            else:
                current_bit_pattern = torch.sin(2 * math.pi * freq_x * x_coords + angle) * torch.sin(2 * math.pi * freq_y * y_coords)
                
        elif pattern_type == 'gradient':
            if bit_val == 1:
                center_y, center_x = (h-1)/2, (w-1)/2 # Center for 0-indexed
                dist_sq = ((y_coords* (h-1) - center_y)**2 + (x_coords* (w-1) - center_x)**2)
                # Avoid issues with sqrt(0) if h or w is 1 by adding small epsilon to dist_sq if it's 0
                dist = torch.sqrt(dist_sq + 1e-8 if h==1 or w==1 else dist_sq)
                current_bit_pattern = torch.cos(dist * (0.1 / (max(h,w)/100) + 0.01 * bit_idx)) # Scale factor for dist
            else:
                current_bit_pattern = x_coords * math.cos(angle) + y_coords * math.sin(angle)
                current_bit_pattern = torch.cos(current_bit_pattern * (5 + bit_idx % 3))
                
        elif pattern_type == 'noise':
            g = torch.Generator(device=device)
            g.manual_seed(bit_idx + 1) 
            noise = torch.randn(h, w, generator=g, device=device)
            if bit_val == 1:
                # Smoother noise - using a fixed small kernel for simplicity
                # Ensure kernel_size is odd and appropriate for image size
                ks = min(max(3, h // 64, w // 64) | 1, 7) # Odd, between 3 and 7
                # Pad manually for avg_pool if needed or use conv with padding
                # For avg_pool2d, it does not take padding as integer like conv2d
                # Let's use a Gaussian blur for smoother noise
                # For simplicity, use a simple box blur via convolution
                blur_kernel = torch.ones(1, 1, ks, ks, device=device) / (ks*ks)
                current_bit_pattern = F.conv2d(noise.unsqueeze(0).unsqueeze(0), blur_kernel, padding=ks//2).squeeze()

            else:
                current_bit_pattern = noise
        
        if current_bit_pattern.abs().max() > 1e-8: # Avoid division by zero for null patterns
            current_bit_pattern = current_bit_pattern / (current_bit_pattern.abs().max() + 1e-8)
        
        # Modulate strength per bit and add to cumulative pattern
        bit_specific_strength = strength * (0.5 + 1.0 * (bit_idx / msg_len)) # Vary strength
        cumulative_pattern += bit_specific_strength * current_bit_pattern

    # Apply the single cumulative_pattern to all channels of the single image
    for c in range(img_to_modify.shape[0]): # Iterate C channels
        img_to_modify[c] += cumulative_pattern # Pattern is (H,W), broadcasted addition to (H,W) channel slice
    
    modified_image_batch[0] = img_to_modify.clamp(0, 1) # Clamp and put back
    return modified_image_batch


# ========= Image Generation Function (Modified) ==========
def generate_images(
    accelerator: Accelerator, all_prompt_embeddings: torch.Tensor, noise_scheduler: DDPMScheduler,
    sd_vae: AutoencoderKL, unet: UNet2DConditionModel, generator: torch.Generator,
    weight_dtype: torch.dtype, config: Config, run_id: int = 0
):
    safe_prompt_tag = re.sub(r'[^\w_.)( -]', '', config.prompt[:50]).strip().replace(' ', '_')
    image_save_dir = os.path.join(config.output_dir, f"{safe_prompt_tag}_run{run_id}")
    os.makedirs(image_save_dir, exist_ok=True)

    num_channels_latents = unet.config.in_channels
    height, width = config.image_size, config.image_size
    total_images_to_generate_for_this_prompt = config.num_images_per_prompt
    unet_processing_batch_size = config.batch_size_generation
    num_unet_batches = (total_images_to_generate_for_this_prompt + unet_processing_batch_size - 1) // unet_processing_batch_size
    generated_image_count = 0
    do_classifier_free_guidance = config.guidance_scale > 1.0

    for batch_idx in range(num_unet_batches):
        current_unet_batch_size = min(
            unet_processing_batch_size, 
            total_images_to_generate_for_this_prompt - (batch_idx * unet_processing_batch_size)
        )
        if current_unet_batch_size <= 0: continue

        latents_shape = (current_unet_batch_size, num_channels_latents, height // 8, width // 8)
        initial_latents_for_batch = torch.randn(
            latents_shape, generator=generator, device=accelerator.device, dtype=weight_dtype
        )
        latents = initial_latents_for_batch * noise_scheduler.init_noise_sigma

        noise_scheduler.set_timesteps(config.inference_steps, device=accelerator.device)
        timesteps = noise_scheduler.timesteps
        
        start_idx_for_this_batch_prompts = batch_idx * unet_processing_batch_size
        end_idx_for_this_batch_prompts = start_idx_for_this_batch_prompts + current_unet_batch_size

        if do_classifier_free_guidance:
            uncond_batch_embeds = all_prompt_embeddings[start_idx_for_this_batch_prompts : end_idx_for_this_batch_prompts]
            cond_offset = total_images_to_generate_for_this_prompt 
            cond_batch_embeds = all_prompt_embeddings[
                cond_offset + start_idx_for_this_batch_prompts : \
                cond_offset + end_idx_for_this_batch_prompts
            ]
            batch_prompt_embeddings_for_unet = torch.cat([uncond_batch_embeds, cond_batch_embeds], dim=0)
        else:
            batch_prompt_embeddings_for_unet = all_prompt_embeddings[start_idx_for_this_batch_prompts : end_idx_for_this_batch_prompts]
        
        logger.info(f"\nBatch {batch_idx+1}/{num_unet_batches}: Generating {current_unet_batch_size} images.")
        logger.info(f"  UNet prompt embeddings shape: {batch_prompt_embeddings_for_unet.shape}")

        for i, t in enumerate(tqdm(timesteps, desc=f"Batch {batch_idx+1} Denoising")):
            current_denoising_step = config.inference_steps - i 

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

            with torch.no_grad():
                noise_pred_out = unet(latent_model_input, t, encoder_hidden_states=batch_prompt_embeddings_for_unet).sample

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred_out.chunk(2)
                noise_pred = noise_pred_uncond + config.guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = noise_pred_out

            if current_denoising_step == config.embedding_step:
                logger.info(f"  Embedding message at denoising step {current_denoising_step} (timestep {t.item()}) using '{config.embedding_pattern_type}' pattern.")
                with torch.no_grad():
                    alpha_prod_t = noise_scheduler.alphas_cumprod[t]
                    beta_prod_t = 1 - alpha_prod_t
                    pred_original_sample = (latents - beta_prod_t**(0.5) * noise_pred) / alpha_prod_t**(0.5)
                    image_latents_for_decode = (pred_original_sample / config.sd_vae_scale_factor).to(sd_vae.device, dtype=sd_vae.dtype)
                    decoded_x0_image = sd_vae.decode(image_latents_for_decode).sample
                    image_for_embedding = (decoded_x0_image / 2 + 0.5).clamp(0, 1)
                    
                    if accelerator.is_main_process:
                        pre_embed_path = os.path.join(image_save_dir, f"pre_embed_s{current_denoising_step}_b{batch_idx}_img{generated_image_count}.png")
                        torchvision.utils.save_image(image_for_embedding[0].float().cpu(), pre_embed_path)

                    # Embed message using the global pattern method
                    modified_image_for_embedding = embed_message_global_pattern(
                        image_for_embedding, 
                        config.binary_message,
                        strength=config.embedding_strength, # Use the general strength from Config
                        pattern_type=config.embedding_pattern_type # Use pattern type from Config
                    )
                    
                    if accelerator.is_main_process:
                        post_embed_path = os.path.join(image_save_dir, f"post_embed_s{current_denoising_step}_b{batch_idx}_img{generated_image_count}.png")
                        torchvision.utils.save_image(modified_image_for_embedding[0].float().cpu(), post_embed_path)

                    modified_image_scaled_back = (modified_image_for_embedding * 2 - 1).to(sd_vae.device, dtype=sd_vae.dtype)
                    modified_latents_dist = sd_vae.encode(modified_image_scaled_back).latent_dist
                    modified_x0_latents = modified_latents_dist.sample() 
                    modified_x0_latents = modified_x0_latents * config.sd_vae_scale_factor
                    noise_for_current_step = torch.randn_like(modified_x0_latents)
                    latents = noise_scheduler.add_noise(
                        modified_x0_latents.to(latents.device, dtype=latents.dtype), 
                        noise_for_current_step.to(latents.device, dtype=latents.dtype), 
                        t.unsqueeze(0) 
                    ).to(latents.dtype)
                    logger.info(f"    Message embedded. New latents shape: {latents.shape}")
            
            latents = noise_scheduler.step(noise_pred, t, latents, generator=generator).prev_sample
            
            if config.save_intermediate_every_n_steps > 0 and \
               (i + 1) % config.save_intermediate_every_n_steps == 0 and \
               (i + 1) < len(timesteps):
                with torch.no_grad():
                    alpha_prod_t_inter = noise_scheduler.alphas_cumprod[t]
                    beta_prod_t_inter = 1 - alpha_prod_t_inter
                    pred_original_sample_inter = (latents - beta_prod_t_inter**(0.5) * noise_pred) / alpha_prod_t_inter**(0.5)
                    image_latents_for_decode_inter = (pred_original_sample_inter / config.sd_vae_scale_factor).to(sd_vae.dtype)
                    image_decoded_inter = sd_vae.decode(image_latents_for_decode_inter).sample
                    image_to_save_inter = (image_decoded_inter / 2 + 0.5).clamp(0, 1)
                    
                    for img_idx_in_batch in range(image_to_save_inter.shape[0]):
                        global_img_idx = generated_image_count + img_idx_in_batch
                        intermediate_image_path = os.path.join(image_save_dir,
                            f"intermediate_s{current_denoising_step}_b{batch_idx}_img{global_img_idx}.png")
                        torchvision.utils.save_image(image_to_save_inter[img_idx_in_batch].float().cpu(), intermediate_image_path)
        
        with torch.no_grad():
            final_latents_for_decode = latents / config.sd_vae_scale_factor
            image_final_decoded = sd_vae.decode(final_latents_for_decode.to(sd_vae.dtype)).sample
            image_final_save_scaled = (image_final_decoded / 2 + 0.5).clamp(0, 1)

            for img_idx_in_batch in range(image_final_save_scaled.shape[0]):
                global_img_idx = generated_image_count + img_idx_in_batch
                final_image_name = os.path.join(image_save_dir, f"final_img{global_img_idx}_embedded_s{config.embedding_step}_{config.embedding_pattern_type}.png")
                torchvision.utils.save_image(image_final_save_scaled[img_idx_in_batch].float().cpu(), final_image_name)
                logger.info(f"Saved final image: {final_image_name}")
        
        generated_image_count += current_unet_batch_size
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"All {generated_image_count} images generated and saved in {image_save_dir}")

# ========= Main Orchestration Function ==========
def main(config: Config):
    HUGGINGFACE_HUB_TOKEN = None 
    accelerator = setup_accelerator_and_logging(config.SEED, config.gradient_accumulation_steps, config.mixed_precision, config.output_dir)
    
    logger.info("Loading models and tokenizer...")
    tokenizer, text_encoder, noise_scheduler, sd_vae, unet, generator, weight_dtype = \
        load_models_and_tokenizer(accelerator, config.STABLE_DIFFUSION_MODEL_NAME, 
                                  config.SD_REVISION, config.SEED, HUGGINGFACE_HUB_TOKEN)
    
    logger.info(f"Models loaded. UNet dtype: {unet.dtype}, SD VAE dtype: {sd_vae.dtype}, Text Encoder dtype: {text_encoder.dtype}")
    logger.info(f"Embedding a 32-bit message: '{config.binary_message}' at step {config.embedding_step} using '{config.embedding_pattern_type}' pattern.")

    unet.eval()
    text_encoder.eval()
    sd_vae.eval()

    do_classifier_free_guidance = config.guidance_scale > 1.0
    prompt_batch_for_encoder = [config.prompt] 
    
    all_prompt_embeddings = _encode_prompt(
        prompt_batch_for_encoder, tokenizer, text_encoder, accelerator.device, 
        config.num_images_per_prompt, 
        do_classifier_free_guidance
    )
    
    logger.info(f"Total prompt embeddings generated (shape): {all_prompt_embeddings.shape}")

    generate_images(
        accelerator, all_prompt_embeddings, noise_scheduler, sd_vae, unet,
        generator, weight_dtype, config, run_id=0 
    )
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        try: accelerator.end_training()
        except Exception as e: logger.warning(f"Error during accelerator.end_training(): {e}")
    logger.info("Image generation complete.")

# ========= Entry Point ==========
if __name__ == "__main__":
    config_run = Config(
        num_images_per_prompt=1, 
        inference_steps=100, 
        embedding_step=80,   
        save_intermediate_every_n_steps=10,
        embedding_pattern_type='sinusoidal' # Choose 'sinusoidal', 'gradient', or 'noise'
        # embedding_strength = 0.01 # Optionally override default strength
    )
    main(config_run)